from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.markers import CUBOID_MARKER_CFG

from .isaac_labyrinth_env_cfg import LabyrinthEnvCfg
from isaac_robots.scripts.labyrinth_builder import LabyrinthBuilder


class LabyrinthDirectEnv(DirectRLEnv):
    cfg: LabyrinthEnvCfg

    def __init__(self, cfg: LabyrinthEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.cfg.observation_space,), dtype=np.float32,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Action buffers
        self._actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # Goal tracking: ring waypoints (approach -> center -> exit per ring)
        n_waypoints = len(self._labyrinth.rings_env0) * 3
        self._n_waypoints = max(n_waypoints, 1)
        self._waypoints_w = torch.zeros(self.num_envs, self._n_waypoints, 3, device=self.device)
        self._waypoint_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Precompute ring waypoints (env-local, same for all envs)
        ring_wp = self._goal_sampler.sample_ring_waypoints(self._labyrinth.rings_env0)
        self._ring_waypoints_local = torch.tensor(ring_wp, dtype=torch.float32, device=self.device)

        # Geodesic distance fields (one per waypoint)
        dist_fields_np = self._goal_sampler.compute_waypoint_distance_fields(
            self._labyrinth.rings_env0,
        )
        self._dist_fields = torch.tensor(dist_fields_np, dtype=torch.float32, device=self.device)
        grid = self._goal_sampler.grid
        self._grid_half = grid._half
        self._grid_res = grid.res
        self._grid_n = grid.n
        self._prev_remaining = torch.full((self.num_envs,), float("inf"), device=self.device)

        # Episode tracking
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel", "ang_vel", "path_progress", "goal_reached",
                "tilt", "action_smoothness", "alive", "proximity",
            ]
        }
        self._full_loop_completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Robot physics
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight_scalar = (self._robot_mass * self._gravity_magnitude).item()
        # Per-env weight (for mass randomization)
        self._robot_weight = torch.full(
            (self.num_envs,), self._robot_weight_scalar, device=self.device,
        )

        # Grace period after reset (for contact termination)
        self._reset_grace_steps_remaining = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._reset_grace_duration = int(
            self.cfg.grace_seconds / (self.cfg.decimation * self.cfg.sim.dt)
        )

        # Visualization
        self._waypoint_markers = VisualizationMarkers(self.cfg.waypoint_markers)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self) -> None:
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # LiDAR sensor
        if self.cfg.lidar is not None:
            self._lidar = self.cfg.lidar.class_type(self.cfg.lidar)
            self.scene.sensors["lidar"] = self._lidar
        else:
            self._lidar = None

        # Contact sensor
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # Terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self._env_origins = self._terrain.env_origins

        # Build labyrinth (walls + rings) for all envs
        self._labyrinth = LabyrinthBuilder(self.cfg.labyrinth)
        self._labyrinth.build(self.scene, self._env_origins)
        self._goal_sampler = self._labyrinth.goal_sampler

        # Clone environments
        self.scene.clone_environments(copy_from_source=True)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ── physics / actions ────────────────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._previous_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)
        thrust = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        # Domain randomization: thrust noise
        if self.cfg.domain_rand.thrust_noise_std > 0:
            noise = torch.randn(self.num_envs, device=self.device) * self.cfg.domain_rand.thrust_noise_std
            thrust = thrust * (1.0 + noise)
        self._thrust[:, 0, 2] = thrust
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self) -> None:
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id,
            forces=self._thrust,
            torques=self._moment,
        )

    # ── observations ─────────────────────────────────────────────────────────

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
            # Domain randomization: sensor noise (raw metres, before normalization)
            if self.cfg.domain_rand.sensor_noise_std > 0:
                noise = torch.randn_like(lidar_ranges) * self.cfg.domain_rand.sensor_noise_std
                lidar_ranges = lidar_ranges + noise
            max_dist = self.cfg.sensor_selection.lidar_max_distance_m
            lidar_ranges.div_(max_dist)
            lidar_ranges.nan_to_num_(nan=1.0, posinf=1.0, neginf=0.0)
            lidar_ranges.clamp_(0.0, 1.0)
            obs_parts.append(lidar_ranges.reshape(self.num_envs, -1))

        return {"policy": torch.cat(obs_parts, dim=-1)}

    # ── geodesic distance lookup ─────────────────────────────────────────────

    def _geodesic_remaining(self) -> torch.Tensor:
        """Look up geodesic distance to current waypoint from the drone's XY position."""
        # Convert world XY to env-local XY
        pos_w = self._robot.data.root_pos_w  # (B, 3)
        local_xy = pos_w[:, :2] - self._env_origins[:, :2]
        # Convert to grid cell indices
        ci = ((local_xy[:, 0] + self._grid_half) / self._grid_res).long().clamp(0, self._grid_n - 1)
        cj = ((local_xy[:, 1] + self._grid_half) / self._grid_res).long().clamp(0, self._grid_n - 1)
        # Gather from per-waypoint distance fields
        return self._dist_fields[self._waypoint_idx, ci, cj]

    # ── rewards ──────────────────────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        dt = self.step_dt
        pos_w = self._robot.data.root_pos_w
        lin_vel_b = self._robot.data.root_lin_vel_b
        ang_vel_b = self._robot.data.root_ang_vel_b
        proj_grav = self._robot.data.projected_gravity_b

        # 1. Waypoint advancement
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - pos_w, dim=1)
        goal_reached = (distance_to_goal < self.cfg.goal_reached_threshold).float()
        reached_ids = goal_reached.bool().nonzero(as_tuple=False).squeeze(-1)
        if reached_ids.numel() > 0:
            new_idx = (self._waypoint_idx[reached_ids] + 1) % self._n_waypoints
            # Track full-loop completion (all rings traversed)
            wrapped = new_idx < self._waypoint_idx[reached_ids]
            self._full_loop_completed[reached_ids[wrapped]] = True
            self._waypoint_idx[reached_ids] = new_idx
            self._desired_pos_w[reached_ids] = self._waypoints_w[
                reached_ids, self._waypoint_idx[reached_ids]
            ]
            self._prev_remaining[reached_ids] = float("inf")

        # 2. Geodesic path progress (primary navigation signal)
        remaining = self._geodesic_remaining()
        path_progress = torch.where(
            torch.isinf(self._prev_remaining),
            torch.zeros_like(remaining),
            self._prev_remaining - remaining,
        ).clamp(-1.0, 1.0)
        self._prev_remaining = remaining.detach()

        # 3. Wall proximity penalty from ToF sensor
        proximity_penalty = torch.zeros(self.num_envs, device=self.device)
        if self._lidar is not None:
            ray_hits_w = self._lidar.data.ray_hits_w
            lidar_origin_w = self._lidar.data.pos_w.unsqueeze(1)
            lidar_ranges = torch.linalg.norm(ray_hits_w - lidar_origin_w, dim=-1)
            min_range, _ = lidar_ranges.min(dim=-1)
            proximity_penalty = torch.clamp(
                1.0 - min_range / self.cfg.wall_danger_distance, min=0.0,
            )

        # 4. Stability & smoothness penalties
        tilt_error = 1.0 + proj_grav[:, 2]
        lin_vel_sq = torch.sum(torch.square(lin_vel_b), dim=1)
        ang_vel_sq = torch.sum(torch.square(ang_vel_b), dim=1)
        action_diff_sq = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)

        rewards = {
            "path_progress": path_progress * self.cfg.distance_to_goal_reward_scale,
            "goal_reached": goal_reached * self.cfg.goal_reached_bonus,
            "alive": torch.full((self.num_envs,), self.cfg.alive_bonus * dt, device=self.device),
            "proximity": proximity_penalty * self.cfg.wall_proximity_reward_scale * dt,
            "tilt": tilt_error * self.cfg.tilt_reward_scale * dt,
            "lin_vel": lin_vel_sq * self.cfg.lin_vel_reward_scale * dt,
            "ang_vel": ang_vel_sq * self.cfg.ang_vel_reward_scale * dt,
            "action_smoothness": action_diff_sq * self.cfg.action_smoothness_scale * dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        reward = torch.nan_to_num(reward, nan=0.0, posinf=10.0, neginf=-10.0)

        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    # ── termination ──────────────────────────────────────────────────────────

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Height bounds
        too_low = self._robot.data.root_pos_w[:, 2] < 0.1
        too_high = self._robot.data.root_pos_w[:, 2] > self.cfg.labyrinth.wall_height + 0.5

        # Contact termination with grace period
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        collision = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self._body_id], dim=-1), dim=1)[0]
            > self.cfg.collision_force_threshold,
            dim=1,
        )
        grace_active = self._reset_grace_steps_remaining > 0
        self._reset_grace_steps_remaining[grace_active] -= 1

        died = too_low | too_high | (collision & ~grace_active)
        return died, time_out

    # ── reset ────────────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # ── Logging ──
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        rings_completed = (self._waypoint_idx[env_ids] // 3).float().mean()
        n_resetting = len(env_ids)
        success_count = self._full_loop_completed[env_ids].sum().item()

        extras = {}
        for key in self._episode_sums:
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras[f"Episode_Reward/{key}"] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = {}
        self.extras["log"].update(extras)
        self.extras["log"]["Episode_Termination/collision"] = torch.count_nonzero(
            self.reset_terminated[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/time_out"] = torch.count_nonzero(
            self.reset_time_outs[env_ids]
        ).item()
        self.extras["log"]["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        final_rem = self._prev_remaining[env_ids]
        self.extras["log"]["Metrics/final_path_remaining"] = (
            final_rem.nan_to_num(nan=0.0, posinf=0.0).mean().item()
        )
        self.extras["log"]["Metrics/rings_completed"] = rings_completed.item()
        self.extras["log"]["Metrics/success_rate"] = success_count / max(n_resetting, 1)
        self.extras["log"]["Metrics/episode_length"] = (
            self.episode_length_buf[env_ids].float().mean().item()
        )

        # ── Reset ──
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        # Reset action buffers
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._thrust[env_ids] = 0.0
        self._moment[env_ids] = 0.0

        # Reset path tracking
        self._prev_remaining[env_ids] = float("inf")
        self._full_loop_completed[env_ids] = False

        # Set ring waypoints in world frame (env_origin + local waypoints)
        origins = self._env_origins[env_ids]  # (B, 3)
        for i in range(self._n_waypoints):
            self._waypoints_w[env_ids, i] = origins + self._ring_waypoints_local[i]

        self._waypoint_idx[env_ids] = 0
        self._desired_pos_w[env_ids] = self._waypoints_w[env_ids, 0]

        # Grace period
        self._reset_grace_steps_remaining[env_ids] = self._reset_grace_duration

        # Domain randomization: mass
        if self.cfg.domain_rand.mass_randomization_pct > 0:
            pct = self.cfg.domain_rand.mass_randomization_pct
            scale = 1.0 + (torch.rand(len(env_ids), device=self.device) * 2 - 1) * pct
            self._robot_weight[env_ids] = self._robot_weight_scalar * scale

        # Reset robot pose
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._env_origins[env_ids]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # ── debug vis ────────────────────────────────────────────────────────────

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
        self._waypoint_markers.visualize(self._waypoints_w.view(-1, 3))
