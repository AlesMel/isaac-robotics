from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import Camera, ContactSensor
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.markers import CUBOID_MARKER_CFG

from .isaac_labyrinth_env_cfg import LabyrinthEnvCfg
from isaac_robots.scripts.labyrinth_builder import LabyrinthBuilder


class LabyrinthDirectEnv(DirectRLEnv):
    cfg: LabyrinthEnvCfg

    def __init__(self, cfg: LabyrinthEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        # Recompute obs dim from actual cfg — cfg.observation_space may be stale
        # if camera was set on the cfg after __post_init__ ran (which only saw camera=None).
        if self.cfg.camera is not None:
            # Asymmetric mode: actor gets proprio(12) + stacked frames
            cam_h, cam_w = self.cfg.camera.height, self.cfg.camera.width
            _obs_dim = 12 + self.cfg.frame_stack * cam_h * cam_w
            self._frame_buffer = torch.zeros(
                self.num_envs, self.cfg.frame_stack, cam_h, cam_w, device=self.device,
            )
        else:
            _obs_dim = 12
            if self.cfg.lidar is not None:
                _obs_dim += self.cfg.sensor_selection.lidar_flat_dim
            self._frame_buffer = None
        # Keep cfg and gym spaces in sync.
        self.cfg.observation_space = _obs_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(_obs_dim,), dtype=np.float32,
        )
        self.single_observation_space["policy"] = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(_obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Action buffers
        self._actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # ── Multi-layout setup ────────────────────────────────────────────────
        n_layouts = self.cfg.labyrinth.n_layouts
        self._env_layout_id = torch.arange(self.num_envs, device=self.device) % n_layouts

        # Canonical ring count (same for every layout, includes sentinel padding)
        canonical_n_rings = len(self._labyrinth._rings_per_layout[0])
        self._n_waypoints = max(canonical_n_rings, 1)
        self._waypoints_w = torch.zeros(self.num_envs, self._n_waypoints, 3, device=self.device)
        self._waypoint_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Per-layout count of REAL (non-sentinel) rings.  Sentinel rings are
        # below-floor padding at z=-1 used to keep tensor shapes uniform;
        # they must never be assigned as navigation targets.
        real_counts = [
            max(sum(1 for r in self._labyrinth._rings_per_layout[k] if r.pos[2] >= 0), 1)
            for k in range(n_layouts)
        ]
        self._real_wp_count = torch.tensor(real_counts, dtype=torch.long, device=self.device)
        self._env_real_wp_count = self._real_wp_count[self._env_layout_id]

        # Per-layout ring waypoints: (n_layouts, n_wp, 3)
        wp_list = [
            self._labyrinth._goal_samplers[k].sample_ring_waypoints(
                self._labyrinth._rings_per_layout[k]
            )
            for k in range(n_layouts)
        ]
        self._ring_waypoints_per_layout = torch.tensor(
            np.stack(wp_list, axis=0), dtype=torch.float32, device=self.device
        )

        # Per-layout geodesic distance fields: (n_layouts, n_wp, grid_n, grid_n)
        grid = self._labyrinth._goal_samplers[0].grid
        self._grid_half = grid._half
        self._grid_res = grid.res
        self._grid_n = grid.n
        df_list = [
            self._labyrinth._goal_samplers[k].compute_waypoint_distance_fields(
                self._labyrinth._rings_per_layout[k]
            )
            if self._labyrinth._rings_per_layout[k]
            else np.zeros((1, grid.n, grid.n), dtype=np.float32)
            for k in range(n_layouts)
        ]
        self._dist_fields = torch.tensor(
            np.stack(df_list, axis=0), dtype=torch.float32, device=self.device
        )
        self._prev_remaining = torch.full((self.num_envs,), float("inf"), device=self.device)

        # Episode tracking
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel", "ang_vel", "path_progress", "goal_reached",
                "tilt", "action_smoothness", "alive", "proximity", "low_altitude",
                "geodesic_shaping",
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

        # Lidar range cache (computed once in _get_rewards, reused in _get_observations).
        # NOTE: For just-reset envs, the first observation uses pre-reset lidar data.
        # This is a known Isaac Lab pattern — sensor data refreshes after the next
        # physics step. The impact is negligible for RL training.
        self._lidar_ranges_raw: torch.Tensor | None = None

        # Grace period after reset (for contact termination)
        self._reset_grace_steps_remaining = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._reset_grace_duration = int(
            self.cfg.grace_seconds / (self.cfg.decimation * self.cfg.sim.dt)
        )

        # Visualization: one sphere per waypoint slot, green (first) → red (last).
        n_wp = self._n_waypoints
        gradient_markers = {}
        for i in range(n_wp):
            t = i / max(n_wp - 1, 1)   # 0.0 (first) → 1.0 (last)
            gradient_markers[f"wp_{i:03d}"] = sim_utils.SphereCfg(
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(t, 1.0 - t, 0.0),   # green → red
                ),
            )
        self._waypoint_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path=f"/Visuals/ring_waypoints_{n_wp:03d}",
                markers=gradient_markers,
            )
        )
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

        # Crazyflie AI bundle HM01B0 monochrome camera (optional).
        # Uses CameraCfg (not TiledCameraCfg) to avoid OffsetCfg rotation bug.
        if self.cfg.camera is not None:
            self._camera = Camera(self.cfg.camera)
            self.scene.sensors["camera"] = self._camera
        else:
            self._camera = None

        # Terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self._env_origins = self._terrain.env_origins

        # Build labyrinth (walls + rings) for all envs
        self._labyrinth = LabyrinthBuilder(self.cfg.labyrinth)
        self._labyrinth.build(self.scene, self._env_origins)

        # Clone environments
        self.scene.clone_environments(copy_from_source=True)

        # Manual collision filtering is required when replicate_physics=False.
        # Include global terrain collisions for all envs.
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Reposition prims for non-zero layouts BEFORE sim.reset() initializes PhysX.
        # USD changes made after sim.reset() are not picked up by the physics engine.
        if self.cfg.labyrinth.n_layouts > 1:
            self._labyrinth.apply_layouts_to_envs(
                self.scene.cfg.num_envs, self._env_origins
            )

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ── physics / actions ────────────────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._previous_actions = self._actions.clone()
        if torch.isnan(actions).any():
            print(f"[WARN] NaN actions detected — model weights likely exploded. "
                  f"NaN count: {torch.isnan(actions).sum().item()}/{actions.numel()}")
        self._actions = actions.clone().nan_to_num_(nan=0.0).clamp(-1.0, 1.0)
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
        lin_vel_b = self._robot.data.root_lin_vel_b
        ang_vel_b = self._robot.data.root_ang_vel_b
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._desired_pos_w,
        )

        if self._camera is not None:
            # ── Actor obs: proprio(12) + stacked frames ──────────────────────
            rgba = self._camera.data.output["rgb"]  # (N, H, W, 4) uint8
            gray = (
                0.299 * rgba[..., 0].float()
                + 0.587 * rgba[..., 1].float()
                + 0.114 * rgba[..., 2].float()
            ) / 255.0  # (N, H, W), normalised to [0, 1]

            # Shift buffer left — .clone() required because source/dest overlap
            self._frame_buffer[:, :-1] = self._frame_buffer[:, 1:].clone()
            self._frame_buffer[:, -1] = gray

            proprio = torch.cat([
                lin_vel_b, ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b,
            ], dim=-1)  # (N, 12)
            actor_obs = torch.cat([
                proprio,
                self._frame_buffer.reshape(self.num_envs, -1),
            ], dim=-1)  # (N, 16396)

            # ── Critic obs: privileged state (11D) ───────────────────────────
            if self._lidar_ranges_raw is not None:
                min_obstacle = self._lidar_ranges_raw.min(dim=-1).values.unsqueeze(-1)
            else:
                min_obstacle = torch.full(
                    (self.num_envs, 1),
                    self.cfg.sensor_selection.lidar_max_distance_m,
                    device=self.device,
                )
            geodesic = self._geodesic_remaining().unsqueeze(-1)
            critic_obs = torch.cat([
                desired_pos_b, lin_vel_b, ang_vel_b,
                geodesic, min_obstacle,
            ], dim=-1)  # (N, 11)

            return {
                "policy": actor_obs.nan_to_num_(),
                "critic": critic_obs.nan_to_num_(),
            }

        # ── Fallback: no camera (flat MLP mode) ─────────────────────────────
        obs_parts = [
            lin_vel_b, ang_vel_b,
            self._robot.data.projected_gravity_b,
            desired_pos_b,
        ]
        if self._lidar is not None:
            if self._lidar_ranges_raw is None:
                ray_hits_w = self._lidar.data.ray_hits_w
                lidar_origin_w = self._lidar.data.pos_w.unsqueeze(1)
                self._lidar_ranges_raw = torch.linalg.norm(ray_hits_w - lidar_origin_w, dim=-1)
            lidar_ranges = self._lidar_ranges_raw.clone()
            if self.cfg.domain_rand.sensor_noise_std > 0:
                noise = torch.randn_like(lidar_ranges) * self.cfg.domain_rand.sensor_noise_std
                lidar_ranges = lidar_ranges + noise
            max_dist = self.cfg.sensor_selection.lidar_max_distance_m
            lidar_ranges.div_(max_dist)
            lidar_ranges.nan_to_num_(nan=1.0, posinf=1.0, neginf=0.0)
            lidar_ranges.clamp_(0.0, 1.0)
            obs_parts.append(lidar_ranges.reshape(self.num_envs, -1))

        policy_obs = torch.cat(obs_parts, dim=-1)

        # Always return "critic" key to avoid KeyError when state_space > 0
        if self.cfg.state_space:
            if self._lidar_ranges_raw is not None:
                min_obstacle = self._lidar_ranges_raw.min(dim=-1).values.unsqueeze(-1)
            else:
                min_obstacle = torch.full(
                    (self.num_envs, 1),
                    self.cfg.sensor_selection.lidar_max_distance_m,
                    device=self.device,
                )
            geodesic = self._geodesic_remaining().unsqueeze(-1)
            critic_obs = torch.cat([
                desired_pos_b, lin_vel_b, ang_vel_b,
                geodesic, min_obstacle,
            ], dim=-1)
            return {"policy": policy_obs.nan_to_num_(), "critic": critic_obs.nan_to_num_()}

        return {"policy": policy_obs.nan_to_num_()}

    # ── geodesic distance lookup ─────────────────────────────────────────────

    def _geodesic_remaining(self) -> torch.Tensor:
        """Look up geodesic distance to current waypoint from the drone's XY position."""
        # Convert world XY to env-local XY
        pos_w = self._robot.data.root_pos_w  # (B, 3)
        local_xy = pos_w[:, :2] - self._env_origins[:, :2]
        # Convert to grid cell indices
        ci = ((local_xy[:, 0] + self._grid_half) / self._grid_res).long().clamp(0, self._grid_n - 1)
        cj = ((local_xy[:, 1] + self._grid_half) / self._grid_res).long().clamp(0, self._grid_n - 1)
        # Gather from per-layout, per-waypoint distance fields
        dist = self._dist_fields[self._env_layout_id, self._waypoint_idx, ci, cj]
        # Cap inf (occupied/unreachable cells) to a finite max so the
        # RunningStandardScaler in the critic doesn't get corrupted.
        return dist.clamp(max=self.cfg.labyrinth.size * 2.0)

    # ── rewards ──────────────────────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        dt = self.step_dt
        pos_w = self._robot.data.root_pos_w
        lin_vel_b = self._robot.data.root_lin_vel_b
        ang_vel_b = self._robot.data.root_ang_vel_b
        proj_grav = self._robot.data.projected_gravity_b

        # 1. Waypoint advancement (wraps at per-env real count, skipping sentinels)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - pos_w, dim=1)
        goal_reached = (distance_to_goal < self.cfg.goal_reached_threshold).float()
        reached_ids = goal_reached.bool().nonzero(as_tuple=False).squeeze(-1)
        if reached_ids.numel() > 0:
            real_count = self._env_real_wp_count[reached_ids]
            new_idx = (self._waypoint_idx[reached_ids] + 1) % real_count
            # Track full-loop completion (all real rings traversed)
            wrapped = new_idx < self._waypoint_idx[reached_ids]
            self._full_loop_completed[reached_ids[wrapped]] = True
            self._waypoint_idx[reached_ids] = new_idx
            self._desired_pos_w[reached_ids] = self._waypoints_w[
                reached_ids, self._waypoint_idx[reached_ids]
            ]
            self._prev_remaining[reached_ids] = float("inf")

        # 2. Geodesic path progress + continuous shaping
        remaining = self._geodesic_remaining()
        path_progress = torch.where(
            torch.isinf(self._prev_remaining),
            torch.zeros_like(remaining),
            self._prev_remaining - remaining,
        ).clamp(-1.0, 1.0)
        self._prev_remaining = remaining.detach()

        # Continuous shaping: always positive, higher when closer to waypoint.
        # Unlike path_progress (delta), this gives a signal even when hovering.
        geodesic_shaping = 1.0 - torch.tanh(remaining / 2.0)

        # 3. Wall proximity penalty from ToF sensor
        proximity_penalty = torch.zeros(self.num_envs, device=self.device)
        if self._lidar is not None:
            ray_hits_w = self._lidar.data.ray_hits_w
            lidar_origin_w = self._lidar.data.pos_w.unsqueeze(1)
            self._lidar_ranges_raw = torch.linalg.norm(ray_hits_w - lidar_origin_w, dim=-1)
            min_range, _ = self._lidar_ranges_raw.min(dim=-1)
            proximity_penalty = torch.clamp(
                1.0 - min_range / self.cfg.wall_danger_distance, min=0.0,
            )

        # 4. Stability & smoothness penalties
        tilt_error = 1.0 + proj_grav[:, 2]
        lin_vel_sq = torch.sum(torch.square(lin_vel_b), dim=1)
        ang_vel_sq = torch.sum(torch.square(ang_vel_b), dim=1)
        action_diff_sq = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)

        # 5. Low altitude penalty — soft gradient before the hard z<0.1 termination
        local_z = pos_w[:, 2] - self._env_origins[:, 2]
        low_alt_penalty = torch.clamp(self.cfg.low_altitude_threshold - local_z, min=0.0)

        rewards = {
            "path_progress": path_progress * self.cfg.distance_to_goal_reward_scale,
            "goal_reached": goal_reached * self.cfg.goal_reached_bonus,
            "alive": torch.full((self.num_envs,), self.cfg.alive_bonus * dt, device=self.device),
            "proximity": proximity_penalty * self.cfg.wall_proximity_reward_scale * dt,
            "tilt": tilt_error * self.cfg.tilt_reward_scale * dt,
            "lin_vel": lin_vel_sq * self.cfg.lin_vel_reward_scale * dt,
            "ang_vel": ang_vel_sq * self.cfg.ang_vel_reward_scale * dt,
            "action_smoothness": action_diff_sq * self.cfg.action_smoothness_scale * dt,
            "low_altitude": low_alt_penalty * self.cfg.low_altitude_penalty_scale * dt,
            "geodesic_shaping": geodesic_shaping * self.cfg.geodesic_shaping_scale * dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        reward = torch.nan_to_num(reward, nan=0.0, posinf=10.0, neginf=-10.0)

        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    # ── termination ──────────────────────────────────────────────────────────

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # NaN / degenerate state guard — catch bad transforms before they
        # propagate to the camera and trigger "matrix may not be orthonormal".
        # See https://github.com/isaac-sim/IsaacLab/issues/2179
        pos_w = self._robot.data.root_pos_w
        bad_state = torch.isnan(pos_w).any(dim=1) | torch.isinf(pos_w).any(dim=1)

        # Height bounds
        too_low = pos_w[:, 2] < 0.1
        too_high = pos_w[:, 2] > self.cfg.labyrinth.wall_height - 0.05

        # XY out-of-bounds (safety net — perimeter walls should prevent this)
        local_pos_xy = pos_w[:, :2] - self._env_origins[:, :2]
        out_of_bounds = (local_pos_xy.abs() > self.cfg.labyrinth.size / 2).any(dim=1)

        # Contact termination with grace period
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        collision = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self._body_id], dim=-1), dim=1)[0]
            > self.cfg.collision_force_threshold,
            dim=1,
        )
        grace_active = self._reset_grace_steps_remaining > 0
        self._reset_grace_steps_remaining[grace_active] -= 1

        died = bad_state | too_low | too_high | out_of_bounds | (collision & ~grace_active)

        # # ── Debug: termination reasons (remove once training is stable) ──
        # if died.any():
        #     print(
        #         f"[DEATH] n={died.sum().item():.0f}/{self.num_envs} | "
        #         f"bad={bad_state.sum().item():.0f} low={too_low.sum().item():.0f} "
        #         f"high={too_high.sum().item():.0f} oob={out_of_bounds.sum().item():.0f} "
        #         f"coll={(collision & ~grace_active).sum().item():.0f} | "
        #         f"avg_z={pos_w[:, 2].mean().item():.3f} "
        #         f"min_z={pos_w[:, 2].min().item():.3f} "
        #         f"max_z={pos_w[:, 2].max().item():.3f} "
        #         f"ep_len={self.episode_length_buf[died].float().mean().item():.1f}"
        #     )

        return died, time_out

    # ── reset ────────────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # ── Logging ──
        n_resetting = len(env_ids)
        final_rem = self._prev_remaining[env_ids]

        # Batch all GPU→CPU scalar transfers into a single operation.
        scalars_gpu = torch.stack([
            torch.linalg.norm(self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1).mean(),
            self._waypoint_idx[env_ids].float().mean(),
            self._full_loop_completed[env_ids].sum().float(),
            torch.count_nonzero(self.reset_terminated[env_ids]).float(),
            torch.count_nonzero(self.reset_time_outs[env_ids]).float(),
            final_rem.nan_to_num(nan=0.0, posinf=0.0).mean(),
            self.episode_length_buf[env_ids].float().mean(),
        ])
        (
            final_distance_to_goal,
            rings_completed,
            success_count,
            n_collision,
            n_timeout,
            path_remaining,
            ep_len,
        ) = scalars_gpu.cpu().tolist()

        extras = {}
        for key in self._episode_sums:
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras[f"Episode_Reward/{key}"] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = {}
        self.extras["log"].update(extras)
        self.extras["log"]["Episode_Termination/collision"] = n_collision
        self.extras["log"]["Episode_Termination/time_out"] = n_timeout
        self.extras["log"]["Metrics/final_distance_to_goal"] = final_distance_to_goal
        self.extras["log"]["Metrics/final_path_remaining"] = path_remaining
        self.extras["log"]["Metrics/rings_completed"] = rings_completed
        self.extras["log"]["Metrics/success_rate"] = success_count / max(n_resetting, 1)
        self.extras["log"]["Metrics/episode_length"] = ep_len

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

        # Reset frame stack history
        if self._frame_buffer is not None:
            self._frame_buffer[env_ids] = 0.0

        # Reset path tracking
        self._prev_remaining[env_ids] = float("inf")
        self._full_loop_completed[env_ids] = False

        # Set ring waypoints in world frame using per-env layout assignment
        origins = self._env_origins[env_ids]  # (B, 3)
        layout_ids = self._env_layout_id[env_ids]                          # (B,)
        wp_local = self._ring_waypoints_per_layout[layout_ids]             # (B, n_wp, 3)
        self._waypoints_w[env_ids] = origins.unsqueeze(1) + wp_local

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

        # # ── Debug: detect reset issues (remove once training is stable) ──
        # if len(env_ids) < 64:  # only print for small batches to avoid spam
        #     pos = default_root_state[:, :3]
        #     print(
        #         f"[RESET] n={len(env_ids)} | "
        #         f"pos_z=[{pos[:, 2].min().item():.2f}, {pos[:, 2].max().item():.2f}] | "
        #         f"wp0={self._desired_pos_w[env_ids[0]].cpu().tolist()} | "
        #         f"terminated={self.reset_terminated[env_ids].sum().item():.0f} "
        #         f"timeout={self.reset_time_outs[env_ids].sum().item():.0f}"
        #     )

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
        # Each env's n_waypoints positions get indices 0..n-1 so the gradient
        # colors correctly reflect position in the sequence across all envs.
        marker_indices = torch.arange(
            self._n_waypoints, device=self.device
        ).repeat(self.num_envs)
        self._waypoint_markers.visualize(
            self._waypoints_w.view(-1, 3),
            marker_indices=marker_indices,
        )
