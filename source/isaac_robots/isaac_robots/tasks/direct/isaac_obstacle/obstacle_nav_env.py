from __future__ import annotations

import math

import numpy as np
import torch
from gymnasium import spaces

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

from .obstacle_nav_env_cfg import ObstacleNavEnvCfg
import logging

logger = logging.getLogger(__name__)

class ObstacleNavDirectEnv(DirectRLEnv):
    cfg: ObstacleNavEnvCfg

    def __init__(self, cfg: ObstacleNavEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.cfg.observation_space,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self._actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["lin_vel", "ang_vel", "distance_to_goal",
                        "obstacle_proximity", "collision", "goal_reached"]
        }
        # Bounding radius in XY for a 0.3x0.3 cuboid
        obstacle_radius = 0.5 * math.sqrt(0.3**2 + 0.3**2)
        self._obstacle_radii = torch.tensor([obstacle_radius, obstacle_radius], dtype=torch.float, device=self.device)
        self._obstacle_heights = torch.tensor([1.0, 1.0], dtype=torch.float, device=self.device)
        self._obstacle_positions_w = torch.zeros(self.num_envs, 2, 3, device=self.device)

        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self) -> None:
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._obstacle_0 = RigidObject(self.cfg.obstacle_0)
        self._obstacle_1 = RigidObject(self.cfg.obstacle_1)
        self.scene.rigid_objects["obstacle_0"] = self._obstacle_0
        self.scene.rigid_objects["obstacle_1"] = self._obstacle_1

        if self.cfg.lidar is not None:
            self._lidar = self.cfg.lidar.class_type(self.cfg.lidar)
            self.scene.sensors["lidar"] = self._lidar
        else:
            self._lidar = None

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self._env_origins = self._terrain.env_origins
        
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # self._env_origins = self._terrain.env_origins

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self) -> None:
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id,
            forces=self._thrust,
            torques=self._moment,
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._desired_pos_w,
        )
        obs_parts = [
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            self._robot.data.projected_gravity_b,
            desired_pos_b,
        ]
        if self._lidar is not None:
            ray_hits_w = self._lidar.data.ray_hits_w
            lidar_origin_w = self._lidar.data.pos_w.unsqueeze(1)
            lidar_ranges = torch.linalg.norm(ray_hits_w - lidar_origin_w, dim=-1)
            lidar_ranges = torch.nan_to_num(
                lidar_ranges,
                nan=self.cfg.sensor_selection.lidar_max_distance_m,
                posinf=self.cfg.sensor_selection.lidar_max_distance_m,
                neginf=0.0,
            )
            lidar_ranges = torch.clamp(lidar_ranges / self.cfg.sensor_selection.lidar_max_distance_m, 0.0, 1.0)
            obs_parts.append(lidar_ranges.reshape(self.num_envs, -1))
        obs = torch.cat(obs_parts, dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1.0 - torch.tanh(distance_to_goal / 0.8)

        # Obstacle proximity: surface distance to nearest obstacle
        robot_xy = self._robot.data.root_pos_w[:, :2]
        obstacle_xy = self._obstacle_positions_w[:, :, :2]
        dist_to_obstacles = torch.linalg.norm(
            robot_xy.unsqueeze(1) - obstacle_xy, dim=-1
        ) - self._obstacle_radii.unsqueeze(0)
        min_clearance = dist_to_obstacles.min(dim=-1).values
        proximity_penalty = torch.clamp(1.0 - min_clearance / self.cfg.obstacle_safety_distance, min=0.0)

        # Collision: binary flag
        collision = (min_clearance < self.cfg.collision_margin).float()

        # Goal reached: binary flag
        goal_reached = (distance_to_goal < self.cfg.goal_reached_threshold).float()

        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "obstacle_proximity": proximity_penalty * self.cfg.obstacle_proximity_reward_scale * self.step_dt,
            "collision": collision * self.cfg.collision_penalty,
            "goal_reached": goal_reached * self.cfg.goal_reached_bonus,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        too_low = self._robot.data.root_pos_w[:, 2] < 0.1
        too_high = self._robot.data.root_pos_w[:, 2] > 2.5

        # Obstacle collision
        robot_xy = self._robot.data.root_pos_w[:, :2]
        obstacle_xy = self._obstacle_positions_w[:, :, :2]
        dist = torch.linalg.norm(robot_xy.unsqueeze(1) - obstacle_xy, dim=-1)
        collided = torch.any(
            dist < (self._obstacle_radii.unsqueeze(0) + self.cfg.collision_margin), dim=-1
        )

        # Goal reached (success termination)
        dist_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        reached = dist_to_goal < self.cfg.goal_reached_threshold

        died = too_low | too_high | collided
        time_out = time_out | reached
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()

        extras = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras[f"Episode_Reward/{key}"] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = {}
        self.extras["log"].update(extras)
        self.extras["log"]["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        self.extras["log"]["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"]["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()

        self._actions[env_ids] = 0.0
        self._thrust[env_ids] = 0.0
        self._moment[env_ids] = 0.0

        self._obstacle_positions_w[env_ids, 0, :] = self._obstacle_0.data.root_pos_w[env_ids]
        self._obstacle_positions_w[env_ids, 1, :] = self._obstacle_1.data.root_pos_w[env_ids]
        self._sample_goals(env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._env_origins[env_ids]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _sample_goals(self, env_ids: torch.Tensor) -> None:
        z_min, z_max = self.cfg.goal_z_range
        half_extent = self.cfg.layout_half_extent
        for env_id in env_ids.tolist():
            origin_xy = self._env_origins[env_id, :2]
            obstacle_xy = self._obstacle_positions_w[env_id, :, :2]
            goal_xy = None
            for _ in range(self.cfg.max_sampling_attempts):
                sample = origin_xy + torch.empty(2, device=self.device).uniform_(-half_extent, half_extent)
                obstacle_dist = torch.linalg.norm(obstacle_xy - sample.unsqueeze(0), dim=-1)
                obstacle_clearance = self._obstacle_radii + self.cfg.goal_clearance_margin
                far_from_obstacles = bool(torch.all(obstacle_dist > obstacle_clearance))
                far_from_spawn = bool(
                    torch.linalg.norm(sample - origin_xy) > self.cfg.goal_min_distance_from_spawn
                )
                if far_from_obstacles and far_from_spawn:
                    goal_xy = sample
                    break
            if goal_xy is None:
                goal_xy = origin_xy + torch.tensor((self.cfg.goal_min_distance_from_spawn, 0.0), device=self.device)

            self._desired_pos_w[env_id, 0:2] = goal_xy
            self._desired_pos_w[env_id, 2] = torch.empty(1, device=self.device).uniform_(z_min, z_max).item()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
