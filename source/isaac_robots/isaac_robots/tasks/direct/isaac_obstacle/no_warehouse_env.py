from __future__ import annotations

import os
import numpy as np
import torch
from gymnasium import spaces

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import ContactSensor
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
            for key in [
                "lin_vel", 
                "ang_vel", 
                "distance_to_goal", 
                "goal_reached", 
                "geo_dist_improvement"
        ]
        }

        self._goal_offsets = torch.tensor(
            [
                [0.0, -2.0, 3.0],  # GOAL0
                [-4.5, 3.5, 7.0],   # GOAL0
            ],
            device=self.device,
        )
        self._num_waypoints = len(self._goal_offsets)

        self._body_id = self._robot.find_bodies("body")[0]        
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        self._waypoints_w = torch.zeros(self.num_envs, self._num_waypoints, 3, device=self.device)
        self._waypoint_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._waypoint_markers = VisualizationMarkers(self.cfg.waypoint_markers)

        # Previous geodesic distance for progress-based reward
        self._prev_geo_dist = torch.full((self.num_envs,), float("inf"), device=self.device)

        # Stuck detection: terminate if no geodesic progress toward goal for too long
        self._best_distance_to_goal = torch.full((self.num_envs,), float("inf"), device=self.device)
        self._steps_without_progress = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._stuck_timeout_steps = int(5.0 / (self.cfg.decimation * self.cfg.sim.dt))  # 5 seconds

        # Grace period after reset: suppress collision/stuck detection for a few steps so the drone
        # has time to physically settle at the spawn position before termination checks fire.
        _grace_seconds = 0.3
        self._reset_grace_steps_remaining = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._reset_grace_duration = int(_grace_seconds / (self.cfg.decimation * self.cfg.sim.dt))

        # Termination cause tracking for separate logging
        self._last_stuck = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_collided = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Load geodesic distance field from voxelizer output
        _voxel_dir = os.path.join(os.path.dirname(__file__), "../../../../../../voxel_output")
        _data = np.load(os.path.join(_voxel_dir, "distance_field.npz"))
        # Shape: (num_goals, X, Y, Z) — one field per waypoint goal
        self._dist_fields = torch.tensor(_data["distance_fields"], dtype=torch.float32, device=self.device)
        self._grid_origin = torch.tensor(_data["origin"], dtype=torch.float32, device=self.device)
        self._grid_resolution = float(_data["resolution"])
        self._grid_shape = list(self._dist_fields.shape[1:])  # (X, Y, Z)

        self._occupancy = torch.tensor(_data["occupancy"], dtype=torch.bool, device=self.device)

        import json
        with open(os.path.join(_voxel_dir, "grid_metadata.json")) as _f:
            _meta = json.load(_f)
        self._geo_dist_tanh_scale = float(_meta["max_geodesic_dist_m"])

        # Validate that the baked goals match the configured waypoints (world frame = env-local for env_0)
        _baked_goals = torch.tensor(_data["goals_world"], dtype=torch.float32, device=self.device)
        assert _baked_goals.shape[0] == self._num_waypoints, (
            f"distance_field.npz has {_baked_goals.shape[0]} goals but env has {self._num_waypoints} waypoints. "
            "Re-run voxelizer with all goal positions."
        )
        assert torch.allclose(_baked_goals, self._goal_offsets, atol=0.01), (
            f"Baked goals {_baked_goals.tolist()} don't match waypoint offsets {self._goal_offsets.tolist()}. "
            "Re-run voxelizer with matching --goals."
        )

        self.set_debug_vis(self.cfg.debug_vis)

    def _query_distance_field(self, pos_w: torch.Tensor) -> torch.Tensor:
        """Look up geodesic distance-to-goal for each env given world-frame positions."""
        local = pos_w - self._terrain.env_origins # (num_envs, 3) — env-local frame
        idx = ((local - self._grid_origin) / self._grid_resolution).long()
        idx[:, 0].clamp_(0, self._grid_shape[0] - 1)
        idx[:, 1].clamp_(0, self._grid_shape[1] - 1)
        idx[:, 2].clamp_(0, self._grid_shape[2] - 1)
        # Index per-env field by current waypoint; multiply to convert voxel steps → meters
        return self._dist_fields[self._waypoint_idx, idx[:, 0], idx[:, 1], idx[:, 2]] * self._grid_resolution

    def _setup_scene(self) -> None:
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # if self.cfg.lidar is not None:
        #     self._lidar = self.cfg.lidar.class_type(self.cfg.lidar)
        #     self.scene.sensors["lidar"] = self._lidar
        # else:
        self._lidar = None

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]
        print(f"Action sample: {self._actions[0].cpu().numpy()}, Thrust: {self._thrust[0, 0, 2].item():.3f} N, Moment: {self._moment[0, 0].cpu().numpy()} Nm")

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
        
        obs_list = [
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            self._robot.data.projected_gravity_b,
            desired_pos_b,
        ]

        if self._lidar is not None:
            ray_hits_w = self._lidar.data.ray_hits_w
            lidar_origin_w = self._lidar.data.pos_w.unsqueeze(1)
            
            lidar_ranges = torch.linalg.norm(ray_hits_w - lidar_origin_w, dim=-1)
            max_dist = self.cfg.sensor_selection.lidar_max_distance_m
            
            lidar_ranges.div_(max_dist)  # Divide in-place first so max dist becomes 1.0
            lidar_ranges.nan_to_num_(nan=1.0, posinf=1.0, neginf=0.0)
            lidar_ranges.clamp_(0.0, 1.0)
            
            obs_list.append(lidar_ranges.view(self.num_envs, -1))

        # 5. Concatenate exactly once at the very end
        obs = torch.cat(obs_list, dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        
        # 1. Query distance and sanitize immediately to prevent Out-of-Bounds NaNs
        geo_dist = self._query_distance_field(self._robot.data.root_pos_w)
        geo_dist = torch.nan_to_num(geo_dist, nan=100.0, posinf=100.0)

        # 2. Calculate improvement
        geo_dist_improvement = torch.where(
            torch.isinf(self._prev_geo_dist),
            torch.zeros_like(geo_dist),
            self._prev_geo_dist - geo_dist,  
        )
        
        # 3. Prevent gradient explosions: A drone physically cannot progress more than 
        # a few meters in a single dt step. Clamp it to physically possible bounds.
        geo_dist_improvement = torch.clamp(geo_dist_improvement, min=-1.0, max=1.0)

        self._prev_geo_dist = geo_dist.detach()
        distance_to_goal_mapped = geo_dist_improvement 

        # Waypoint reached: advance to next and update target
        goal_reached = (distance_to_goal < self.cfg.goal_reached_threshold).float()
        reached_ids = goal_reached.bool().nonzero(as_tuple=False).squeeze(-1)
        
        if reached_ids.numel() > 0:
            self._waypoint_idx[reached_ids] = (self._waypoint_idx[reached_ids] + 1) % self._num_waypoints
            self._desired_pos_w[reached_ids] = self._waypoints_w[reached_ids, self._waypoint_idx[reached_ids]]
            
            # 4. CRITICAL FIX: Clear the tracker so the next step doesn't generate a massive penalty
            self._prev_geo_dist[reached_ids] = float("inf")

        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "goal_reached": goal_reached * self.cfg.goal_reached_bonus,
        }
        
        # 5. Final safety stack and sum
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        reward = torch.nan_to_num(reward, nan=0.0, posinf=10.0, neginf=-10.0)

        for key, value in rewards.items():
            self._episode_sums[key] += value
            
        self._episode_sums["geo_dist_improvement"] += geo_dist_improvement

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._body_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        #died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.0, self._robot.data.root_pos_w[:, 2] > 10.0)
        return died, time_out
    
    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()

        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras[f"Episode_Reward/{key}"] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = {}
        self.extras["log"].update(extras)
        self.extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        self.extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        
        # Final geodesic distance (inf → 0 for logging; -1 means unreachable)
        final_geo = self._prev_geo_dist[env_ids]
        self.extras["log"]["Metrics/final_geo_dist"] = final_geo.nan_to_num(nan=0.0, posinf=0.0).mean().item()

        #extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._thrust[env_ids] = 0.0
        self._moment[env_ids] = 0.0

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]

        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Restore these variable resets in _reset_idx
        # self._prev_geo_dist[env_ids] = float("inf")
        # self._steps_without_progress[env_ids] = 0

        # self._waypoints_w[env_ids] = self._terrain.env_origins[env_ids] + self._goal_offsets.unsqueeze(0)
        # self._waypoint_idx[env_ids] = 0
        # self._desired_pos_w[env_ids] = self._waypoints_w[env_ids, 0]

        # # Spawn validity: check if robot's current (post-reset) voxel is occupied or unreachable
        # spawn_local = self._robot.data.root_pos_w[env_ids] - self._env_origins[env_ids]
        # spawn_idx = ((spawn_local - self._grid_origin) / self._grid_resolution).long()
        # spawn_idx[:, 0] = spawn_idx[:, 0].clamp(0, self._grid_shape[0] - 1)
        # spawn_idx[:, 1] = spawn_idx[:, 1].clamp(0, self._grid_shape[1] - 1)
        # spawn_idx[:, 2] = spawn_idx[:, 2].clamp(0, self._grid_shape[2] - 1)
        # spawn_occupied = self._occupancy[spawn_idx[:, 0], spawn_idx[:, 1], spawn_idx[:, 2]].float().mean().item()
        # self.extras["log"]["Metrics/spawn_in_obstacle_frac"] = spawn_occupied
        # spawn_geo = self._dist_fields[0, spawn_idx[:, 0], spawn_idx[:, 1], spawn_idx[:, 2]] * self._grid_resolution
        # # -1.0 → unreachable (inf); 0 → at goal; positive → valid
        # self.extras["log"]["Metrics/spawn_geo_dist"] = spawn_geo.nan_to_num(nan=0.0, posinf=-1.0).mean().item()

        # self._prev_geo_dist[env_ids] = float("inf")

        # self._waypoints_w[env_ids] = self._env_origins[env_ids].unsqueeze(1) + self._goal_offsets.unsqueeze(0)
        # self._waypoint_idx[env_ids] = 0
        # self._desired_pos_w[env_ids] = self._waypoints_w[env_ids, 0]

        # spawn_pos_local = torch.tensor(self.cfg.robot.init_state.pos, device=self.device, dtype=default_root_state.dtype)
        # default_root_state[:, :3] = self._env_origins[env_ids] + spawn_pos_local.unsqueeze(0)

        # spawn_quat = torch.tensor(self.cfg.robot.init_state.rot, device=self.device, dtype=default_root_state.dtype)
        # spawn_quat = spawn_quat / torch.linalg.norm(spawn_quat).clamp_min(1e-8)
        # default_root_state[:, 3:7] = spawn_quat.unsqueeze(0)
        # default_root_state[:, 7:13] = 0.0

        # self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        # self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        # self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # # Seed best-distance with spawn geodesic distance so stuck detection works from step 1
        # spawn_local = default_root_state[:, :3] - self._env_origins[env_ids]
        # spawn_idx = ((spawn_local - self._grid_origin) / self._grid_resolution).long()
        # spawn_idx[:, 0] = spawn_idx[:, 0].clamp(0, self._grid_shape[0] - 1)
        # spawn_idx[:, 1] = spawn_idx[:, 1].clamp(0, self._grid_shape[1] - 1)
        # spawn_idx[:, 2] = spawn_idx[:, 2].clamp(0, self._grid_shape[2] - 1)
        # spawn_wp_idx = self._waypoint_idx[env_ids]
        # spawn_geo = self._dist_fields[spawn_wp_idx, spawn_idx[:, 0], spawn_idx[:, 1], spawn_idx[:, 2]] * self._grid_resolution
        # self._best_distance_to_goal[env_ids] = spawn_geo

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_pos_visualizer.set_visibility(True)
            self._waypoint_markers.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
            self._waypoint_markers.set_visibility(False)

    def _debug_vis_callback(self, event):
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
        #self._waypoint_markers.visualize(self._waypoints_w.view(-1, 3))
