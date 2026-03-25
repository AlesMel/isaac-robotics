from __future__ import annotations

import math
import numpy as np
import torch
from gymnasium import spaces

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import Camera, ContactSensor
from isaaclab.utils.math import subtract_frame_transforms

from .isaac_racing_env_cfg import RacingEnvCfg
from isaac_robots.scripts.racing_track_builder import (
    RacingTrackBuilder, _tang_to_perp_quat, _set_translate_op, _set_orient_op,
)


class RacingDirectEnv(DirectRLEnv):
    cfg: RacingEnvCfg

    def __init__(self, cfg: RacingEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        # Recompute obs dim — cfg.observation_space may be stale if camera was
        # assigned after __post_init__ ran (hydra_task_config pattern).
        if self.cfg.camera is not None:
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

        self.cfg.observation_space = _obs_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(_obs_dim,), dtype=np.float32,
        )
        self.single_observation_space["policy"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(_obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Action buffers
        self._actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # ── Multi-layout setup ────────────────────────────────────────────────
        n_layouts = self.cfg.track.n_layouts
        self._env_layout_id = torch.arange(self.num_envs, device=self.device) % n_layouts

        # Checkpoints: (n_layouts, n_control_points, 3)
        n_cp = self.cfg.track.n_control_points
        self._n_waypoints = n_cp
        wp_list = self._track._centerline_checkpoints_per_layout  # list of (n_cp, 3) arrays
        self._checkpoints_per_layout = torch.tensor(
            np.stack(wp_list, axis=0), dtype=torch.float32, device=self.device,
        )  # (n_layouts, n_cp, 3)

        # Tangents at each checkpoint for spawn-behind-gate positioning: (n_layouts, n_cp, 2)
        self._checkpoint_tangents_per_layout = torch.tensor(
            np.stack(self._track._checkpoint_tangents_per_layout, axis=0),
            dtype=torch.float32, device=self.device,
        )  # (n_layouts, n_cp, 2)

        self._waypoints_w = torch.zeros(self.num_envs, self._n_waypoints, 3, device=self.device)
        self._waypoint_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Episode tracking
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "forward_speed", "goal_reached", "alive",
                "proximity", "tilt", "action_smoothness", "low_altitude",
            ]
        }
        self._laps_completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._checkpoints_this_ep = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Robot physics
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight_scalar = (self._robot_mass * self._gravity_magnitude).item()
        self._robot_weight = torch.full(
            (self.num_envs,), self._robot_weight_scalar, device=self.device,
        )

        # Lidar range cache
        self._lidar_ranges_raw: torch.Tensor | None = None

        # Grace period after reset (for contact termination)
        self._reset_grace_steps_remaining = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._reset_grace_duration = int(
            self.cfg.grace_seconds / (self.cfg.decimation * self.cfg.sim.dt)
        )

        # Visualization: one sphere per checkpoint, colour-coded green→red
        gradient_markers = {}
        for i in range(n_cp):
            t = i / max(n_cp - 1, 1)
            gradient_markers[f"cp_{i:03d}"] = sim_utils.SphereCfg(
                radius=0.06,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(t, 1.0 - t, 0.0),
                ),
            )
        self._checkpoint_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path=f"/Visuals/racing_checkpoints_{n_cp:03d}",
                markers=gradient_markers,
            )
        )
        self.set_debug_vis(self.cfg.debug_vis)

        # Oscillating hanging bars
        self._sim_steps = 0
        self._bar_physx_view = None
        if self.cfg.track.oscillate_hanging_bars and self.cfg.track.n_hanging_bars > 0:
            self._setup_oscillating_bars()


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

        # Camera (optional)
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

        # Build racing track for all envs
        self._track = RacingTrackBuilder(self.cfg.track)
        self._track.build(self.scene, self._env_origins)

        # Clone environments
        self.scene.clone_environments(copy_from_source=True)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Reposition prims for non-zero layouts BEFORE sim.reset()
        if self.cfg.track.n_layouts > 1:
            self._track.apply_layouts_to_envs(
                self.scene.cfg.num_envs, self._env_origins
            )

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _setup_oscillating_bars(self) -> None:
        """Create a PhysX rigid-body view for hanging bars and pre-compute oscillation tensors."""
        import re
        import omni.physics.tensors

        n_bars = self.cfg.track.n_hanging_bars
        # Keep sim_view alive on self — if it goes out of scope the RigidBodyView backend dies.
        self._bar_sim_view = omni.physics.tensors.create_simulation_view("torch")
        self._bar_sim_view.set_subspace_roots("/World/envs")
        raw_view = self._bar_sim_view._backend.create_rigid_body_view(
            "/World/envs/env_*/racing_track/hanging_*"
        )
        if raw_view is None:
            print("[WARN] Hanging bar prims not found in physics view. Disabling oscillation.")
            return

        from omni.physics.tensors.impl.api import RigidBodyView
        self._bar_physx_view = RigidBodyView(raw_view, self._bar_sim_view._frontend)

        total = self._bar_physx_view.count
        expected = self.num_envs * n_bars
        if total != expected:
            print(
                f"[WARN] Oscillating bar count mismatch: view={total}, expected={expected}."
                " Disabling bar oscillation."
            )
            self._bar_physx_view = None
            return

        # get_transforms() → (count, 7) as [x, y, z, qx, qy, qz, qw] in env-local frame
        # (env-local because set_subspace_roots was called)
        init_transforms = self._bar_physx_view.get_transforms()  # torch tensor on device
        transforms_np = init_transforms.cpu().numpy()

        bar_base_z = np.zeros(total, dtype=np.float32)
        bar_amplitude = np.zeros(total, dtype=np.float32)
        bar_frequency = np.ones(total, dtype=np.float32)
        bar_phase = np.zeros(total, dtype=np.float32)

        env_id_re = re.compile(r"/env_(\d+)/")
        bar_idx_re = re.compile(r"/hanging_(\d+)$")

        for vi, path in enumerate(self._bar_physx_view.prim_paths):
            env_id = int(env_id_re.search(path).group(1))
            bar_idx = int(bar_idx_re.search(path).group(1))
            layout_id = int(self._env_layout_id[env_id].item())
            bt = self._track._hanging_bars_per_layout[layout_id][bar_idx]
            bar_base_z[vi] = float(bt[2])
            bar_amplitude[vi] = float(bt[5])
            bar_frequency[vi] = float(bt[6])
            bar_phase[vi] = float(bt[7])

        self._bar_base_z = torch.tensor(bar_base_z, device=self.device)
        self._bar_amplitude = torch.tensor(bar_amplitude, device=self.device)
        self._bar_frequency = torch.tensor(bar_frequency, device=self.device)
        self._bar_phase = torch.tensor(bar_phase, device=self.device)
        # Mutable transform buffer: [x, y, z, qx, qy, qz, qw] env-local
        self._bar_transforms = torch.tensor(transforms_np, dtype=torch.float32, device=self.device)
        self._bar_all_indices = torch.arange(total, dtype=torch.int32, device=self.device)
        # Place bars at their initial Z positions
        self._bar_physx_view.set_transforms(self._bar_transforms, self._bar_all_indices)

    def _randomize_obstacles(self, env_ids: torch.Tensor) -> None:
        """Reassign layouts and apply per-obstacle noise for resetting envs via USD stage ops.

        Uses the same _set_translate_op/_set_orient_op helpers as apply_layouts_to_envs so
        no extra PhysX simulation views are needed — avoids conflicts with Isaac Lab's own
        internal physics_sim_view lifecycle.
        """
        import random as _random
        import omni.usd

        dr = self.cfg.domain_rand
        stage = omni.usd.get_context().get_stage()
        rng = _random.Random(int(torch.randint(0, 2 ** 30, (1,)).item()))

        # ── Layout assignment ──────────────────────────────────────────────
        if dr.reshuffle_layouts_on_reset:
            new_lids = torch.randint(
                0, self.cfg.track.n_layouts, (len(env_ids),),
                dtype=torch.long, device=self.device,
            )
            self._env_layout_id[env_ids] = new_lids
        else:
            new_lids = self._env_layout_id[env_ids]

        new_lids_cpu = new_lids.cpu().tolist()
        env_ids_cpu = env_ids.cpu().tolist()

        canonical_n_gates = len(self._track._gate_posts_per_layout[0])
        canonical_n_cb = len(self._track._crossbars_per_layout[0]) \
            if self._track._crossbars_per_layout else 0
        canonical_n_sc = len(self._track._scatter_per_layout[0]) \
            if self._track._scatter_per_layout else 0
        z_lo, z_hi = 0.10, self.cfg.track.wall_height - 0.05
        half = self.cfg.track.size / 2 - 0.2

        for batch_i, env_id in enumerate(env_ids_cpu):
            lid = int(new_lids_cpu[batch_i])
            gate_posts = self._track._gate_posts_per_layout[lid]
            n_cp_gates = len(gate_posts) // 2

            # Per-checkpoint gate angle noise (same rotation for both posts in a pair)
            angle_noise = [
                rng.uniform(-dr.gate_angle_noise_deg, dr.gate_angle_noise_deg)
                if dr.gate_angle_noise_deg > 0 else 0.0
                for _ in range(n_cp_gates)
            ]

            # ── Gate posts ────────────────────────────────────────────────
            for i in range(canonical_n_gates):
                prim = stage.GetPrimAtPath(
                    f"/World/envs/env_{env_id}/racing_track/gate_{i:03d}"
                )
                if not prim.IsValid():
                    continue
                if i < len(gate_posts):
                    px, py, pz, _r = gate_posts[i]
                    cp_idx = i // 2
                    if dr.gate_angle_noise_deg > 0 and cp_idx < n_cp_gates:
                        # Rotate around midpoint of this gate pair
                        pa = gate_posts[cp_idx * 2]
                        pb = gate_posts[cp_idx * 2 + 1]
                        mid_x = (pa[0] + pb[0]) / 2.0
                        mid_y = (pa[1] + pb[1]) / 2.0
                        angle = math.radians(angle_noise[cp_idx])
                        cos_a, sin_a = math.cos(angle), math.sin(angle)
                        dx, dy = px - mid_x, py - mid_y
                        px = mid_x + cos_a * dx - sin_a * dy
                        py = mid_y + sin_a * dx + cos_a * dy
                    _set_translate_op(prim, px, py, pz)
                else:
                    _set_translate_op(prim, 0.0, 0.0, -10.0)

            # ── Crossbars ─────────────────────────────────────────────────
            if canonical_n_cb > 0:
                crossbars = self._track._crossbars_per_layout[lid]
                for i in range(canonical_n_cb):
                    prim = stage.GetPrimAtPath(
                        f"/World/envs/env_{env_id}/racing_track/crossbar_{i:03d}"
                    )
                    if not prim.IsValid():
                        continue
                    if i < len(crossbars):
                        cx, cy, cz, _length, tang = crossbars[i]
                        if dr.crossbar_z_noise_m > 0:
                            cz += rng.uniform(-dr.crossbar_z_noise_m, dr.crossbar_z_noise_m)
                            cz = max(z_lo, min(z_hi, cz))
                        _set_translate_op(prim, cx, cy, cz)
                        _set_orient_op(prim, _tang_to_perp_quat(np.array(tang, dtype=np.float64)))
                    else:
                        _set_translate_op(prim, 0.0, 0.0, -10.0)

            # ── Scatter obstacles ─────────────────────────────────────────
            if canonical_n_sc > 0:
                scatter = self._track._scatter_per_layout[lid]
                # Checkpoint centers for this layout (used to prevent scatter overlap with goals)
                cp_clearance = self.cfg.track.track_width / 2 + self.cfg.track.scatter_radius_max
                cp_centers = [
                    (
                        (gate_posts[j * 2][0] + gate_posts[j * 2 + 1][0]) / 2.0,
                        (gate_posts[j * 2][1] + gate_posts[j * 2 + 1][1]) / 2.0,
                    )
                    for j in range(n_cp_gates)
                ]
                for i in range(canonical_n_sc):
                    prim = stage.GetPrimAtPath(
                        f"/World/envs/env_{env_id}/racing_track/scatter_{i:03d}"
                    )
                    if not prim.IsValid():
                        continue
                    if i < len(scatter):
                        ox, oy, oz, _size, _is_box = scatter[i]
                        if dr.scatter_xy_noise_m > 0:
                            jx = rng.uniform(-dr.scatter_xy_noise_m, dr.scatter_xy_noise_m)
                            jy = rng.uniform(-dr.scatter_xy_noise_m, dr.scatter_xy_noise_m)
                            nx = max(-half, min(half, ox + jx))
                            ny = max(-half, min(half, oy + jy))
                            # Only apply jitter if it doesn't push the obstacle onto a checkpoint
                            if not any(math.hypot(nx - cx, ny - cy) < cp_clearance for cx, cy in cp_centers):
                                ox, oy = nx, ny
                        _set_translate_op(prim, ox, oy, oz)
                    else:
                        _set_translate_op(prim, 0.0, 0.0, -10.0)

    # ── physics / actions ────────────────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._sim_steps += 1
        if self._bar_physx_view is not None:
            t = float(self._sim_steps) * self.step_dt
            # Drive kinematic bars directly via transform: z = base_z + A*sin(ω*t + φ)
            omega = 2.0 * math.pi * self._bar_frequency
            self._bar_transforms[:, 2] = self._bar_base_z + self._bar_amplitude * torch.sin(
                omega * t + self._bar_phase
            )
            self._bar_physx_view.set_transforms(self._bar_transforms, self._bar_all_indices)

        self._previous_actions = self._actions.clone()
        if torch.isnan(actions).any():
            print(
                f"[WARN] NaN actions detected — model weights likely exploded. "
                f"NaN count: {torch.isnan(actions).sum().item()}/{actions.numel()}"
            )
        self._actions = actions.clone().nan_to_num_(nan=0.0).clamp(-1.0, 1.0)
        thrust = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        if self.cfg.domain_rand.thrust_noise_std > 0:
            noise = torch.randn(self.num_envs, device=self.device) * self.cfg.domain_rand.thrust_noise_std
            thrust = thrust * (1.0 + noise)
        self._thrust[:, 0, 2] = thrust
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self) -> None:
        forces = self._thrust
        if self.cfg.domain_rand.turbulence_std > 0:
            turb = torch.randn(self.num_envs, 1, 3, device=self.device)
            turb *= self.cfg.domain_rand.turbulence_std * self._robot_weight.unsqueeze(-1).unsqueeze(-1)
            forces = self._thrust + turb
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id,
            forces=forces,
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
            ) / 255.0

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
            ], dim=-1)

            # ── Critic obs: privileged state (10D) ───────────────────────────
            if self._lidar_ranges_raw is not None:
                min_obstacle = self._lidar_ranges_raw.min(dim=-1).values.unsqueeze(-1)
            else:
                min_obstacle = torch.full(
                    (self.num_envs, 1),
                    self.cfg.sensor_selection.lidar_max_distance_m,
                    device=self.device,
                )
            critic_obs = torch.cat([
                desired_pos_b, lin_vel_b, ang_vel_b, min_obstacle,
            ], dim=-1)  # (N, 10)

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

        if self.cfg.state_space:
            if self._lidar_ranges_raw is not None:
                min_obstacle = self._lidar_ranges_raw.min(dim=-1).values.unsqueeze(-1)
            else:
                min_obstacle = torch.full(
                    (self.num_envs, 1),
                    self.cfg.sensor_selection.lidar_max_distance_m,
                    device=self.device,
                )
            critic_obs = torch.cat([
                desired_pos_b, lin_vel_b, ang_vel_b, min_obstacle,
            ], dim=-1)
            return {"policy": policy_obs.nan_to_num_(), "critic": critic_obs.nan_to_num_()}

        return {"policy": policy_obs.nan_to_num_()}

    # ── rewards ──────────────────────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        dt = self.step_dt
        pos_w = self._robot.data.root_pos_w
        lin_vel_w = self._robot.data.root_lin_vel_w
        ang_vel_b = self._robot.data.root_ang_vel_b
        proj_grav = self._robot.data.projected_gravity_b

        # 1. Checkpoint reached — simple Euclidean distance
        dist = (self._desired_pos_w - pos_w).norm(dim=1)
        goal_reached = (dist < self.cfg.goal_reached_threshold).float()
        reached_ids = goal_reached.bool().nonzero(as_tuple=False).squeeze(-1)
        if reached_ids.numel() > 0:
            new_idx = (self._waypoint_idx[reached_ids] + 1) % self._n_waypoints
            # Track lap completion (full loop)
            wrapped = new_idx < self._waypoint_idx[reached_ids]
            self._laps_completed[reached_ids[wrapped]] = True
            self._checkpoints_this_ep[reached_ids] += 1
            self._waypoint_idx[reached_ids] = new_idx
            self._desired_pos_w[reached_ids] = self._waypoints_w[
                reached_ids, self._waypoint_idx[reached_ids]
            ]

        # 2. Primary reward: forward speed toward current checkpoint
        goal_dir = self._desired_pos_w - pos_w  # (N, 3)
        goal_dist = goal_dir.norm(dim=1, keepdim=True).clamp(min=1e-6)
        goal_dir_unit = goal_dir / goal_dist
        forward_speed = (lin_vel_w * goal_dir_unit).sum(dim=1)

        # 3. Wall proximity penalty
        proximity_penalty = torch.zeros(self.num_envs, device=self.device)
        if self._lidar is not None:
            ray_hits_w = self._lidar.data.ray_hits_w
            lidar_origin_w = self._lidar.data.pos_w.unsqueeze(1)
            self._lidar_ranges_raw = torch.linalg.norm(ray_hits_w - lidar_origin_w, dim=-1)
            min_range, _ = self._lidar_ranges_raw.min(dim=-1)
            proximity_penalty = torch.clamp(
                1.0 - min_range / self.cfg.wall_danger_distance, min=0.0,
            )

        # 4. Stability & smoothness
        tilt_error = 1.0 + proj_grav[:, 2]
        ang_vel_sq = torch.sum(torch.square(ang_vel_b), dim=1)
        action_diff_sq = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)

        # 5. Low altitude penalty
        local_z = pos_w[:, 2] - self._env_origins[:, 2]
        low_alt_penalty = torch.clamp(self.cfg.low_altitude_threshold - local_z, min=0.0)

        rewards = {
            "forward_speed":      forward_speed * self.cfg.forward_speed_scale * dt,
            "goal_reached":       goal_reached  * self.cfg.goal_reached_bonus,
            "alive":              torch.full((self.num_envs,), self.cfg.alive_bonus * dt, device=self.device),
            "proximity":          proximity_penalty * self.cfg.wall_proximity_reward_scale * dt,
            "tilt":               tilt_error * self.cfg.tilt_reward_scale * dt,
            "action_smoothness":  action_diff_sq * self.cfg.action_smoothness_scale * dt,
            "low_altitude":       low_alt_penalty * self.cfg.low_altitude_penalty_scale * dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        reward = torch.nan_to_num(reward, nan=0.0, posinf=10.0, neginf=-10.0)

        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    # ── termination ──────────────────────────────────────────────────────────

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        pos_w = self._robot.data.root_pos_w
        bad_state = torch.isnan(pos_w).any(dim=1) | torch.isinf(pos_w).any(dim=1)

        too_low = pos_w[:, 2] < 0.1
        too_high = pos_w[:, 2] > self.cfg.track.wall_height - 0.05

        local_pos_xy = pos_w[:, :2] - self._env_origins[:, :2]
        out_of_bounds = (local_pos_xy.abs() > self.cfg.track.size / 2).any(dim=1)

        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        collision = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self._body_id], dim=-1), dim=1)[0]
            > self.cfg.collision_force_threshold,
            dim=1,
        )
        grace_active = self._reset_grace_steps_remaining > 0
        self._reset_grace_steps_remaining[grace_active] -= 1

        died = bad_state | too_low | too_high | out_of_bounds | (collision & ~grace_active)

        return died, time_out

    # ── reset ────────────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        n_resetting = len(env_ids)

        # ── Logging ──
        scalars_gpu = torch.stack([
            torch.linalg.norm(self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1).mean(),
            self._checkpoints_this_ep[env_ids].float().mean(),
            self._laps_completed[env_ids].sum().float(),
            torch.count_nonzero(self.reset_terminated[env_ids]).float(),
            torch.count_nonzero(self.reset_time_outs[env_ids]).float(),
            self.episode_length_buf[env_ids].float().mean(),
        ])
        (
            final_dist,
            checkpoints_completed,
            laps_completed,
            n_collision,
            n_timeout,
            ep_len,
        ) = scalars_gpu.cpu().tolist()

        extras = {}
        for key in self._episode_sums:
            extras[f"Episode_Reward/{key}"] = (
                torch.mean(self._episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = {}
        self.extras["log"].update(extras)
        self.extras["log"]["Episode_Termination/collision"] = n_collision
        self.extras["log"]["Episode_Termination/time_out"] = n_timeout
        self.extras["log"]["Metrics/final_distance_to_checkpoint"] = final_dist
        self.extras["log"]["Metrics/checkpoints_completed"] = checkpoints_completed
        self.extras["log"]["Metrics/laps_completed"] = laps_completed
        self.extras["log"]["Metrics/success_rate"] = laps_completed / max(n_resetting, 1)
        self.extras["log"]["Metrics/episode_length"] = ep_len

        # ── Reset ──
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )
            # Recreate the hanging bar physics view — Isaac Lab's sim.reset() invalidates it.
            if self.cfg.track.oscillate_hanging_bars and self.cfg.track.n_hanging_bars > 0:
                self._setup_oscillating_bars()

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._thrust[env_ids] = 0.0
        self._moment[env_ids] = 0.0

        if self._frame_buffer is not None:
            self._frame_buffer[env_ids] = 0.0

        self._laps_completed[env_ids] = False
        self._checkpoints_this_ep[env_ids] = 0

        # Randomize obstacles + (optionally) reshuffle layouts for this reset batch
        dr = self.cfg.domain_rand
        if (dr.reshuffle_layouts_on_reset or dr.gate_angle_noise_deg > 0
                or dr.crossbar_z_noise_m > 0 or dr.scatter_xy_noise_m > 0):
            self._randomize_obstacles(env_ids)

        # Set checkpoints in world frame using (potentially updated) per-env layout assignment
        origins = self._env_origins[env_ids]      # (B, 3)
        layout_ids = self._env_layout_id[env_ids]  # (B,) — may have been reshuffled above
        cp_local = self._checkpoints_per_layout[layout_ids]  # (B, n_cp, 3)
        self._waypoints_w[env_ids] = origins.unsqueeze(1) + cp_local

        # Checkpoint height noise
        if self.cfg.domain_rand.checkpoint_height_noise_m > 0:
            z_lo = 0.10
            z_hi = self.cfg.track.wall_height - 0.10
            z_noise = (torch.rand(len(env_ids), self._n_waypoints, device=self.device) * 2 - 1) \
                * self.cfg.domain_rand.checkpoint_height_noise_m
            self._waypoints_w[env_ids, :, 2] = (
                self._waypoints_w[env_ids, :, 2] + z_noise
            ).clamp(z_lo, z_hi)

        self._waypoint_idx[env_ids] = 0
        self._desired_pos_w[env_ids] = self._waypoints_w[env_ids, 0]

        self._reset_grace_steps_remaining[env_ids] = self._reset_grace_duration

        # Domain randomization: mass
        if self.cfg.domain_rand.mass_randomization_pct > 0:
            pct = self.cfg.domain_rand.mass_randomization_pct
            scale = 1.0 + (torch.rand(len(env_ids), device=self.device) * 2 - 1) * pct
            self._robot_weight[env_ids] = self._robot_weight_scalar * scale

        # Reset robot pose — spawn at checkpoint 0, slightly behind the gate along the track
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids].clone()

        # Spawn position: checkpoint 0 XY offset 0.5 m upstream (against tangent direction)
        cp0_local = self._checkpoints_per_layout[layout_ids, 0]   # (B, 3)
        tang0 = self._checkpoint_tangents_per_layout[layout_ids, 0]  # (B, 2)
        offset_xy = -tang0 * 0.5  # step back along approach direction

        spawn_noise = torch.randn(len(env_ids), 2, device=self.device) * self.cfg.spawn_noise_xy_m
        spawn_xy = origins[:, :2] + cp0_local[:, :2] + offset_xy + spawn_noise
        default_root_state[:, 0] = spawn_xy[:, 0]
        default_root_state[:, 1] = spawn_xy[:, 1]
        default_root_state[:, 2] = origins[:, 2] + cp0_local[:, 2]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # ── debug vis ────────────────────────────────────────────────────────────

    # Max envs for which checkpoint spheres are drawn — keeps marker count manageable.
    _VIS_MAX_ENVS: int = 8

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                # Large bright-yellow sphere marks the active target for every env.
                self.goal_pos_visualizer = VisualizationMarkers(
                    VisualizationMarkersCfg(
                        prim_path="/Visuals/Racing/active_target",
                        markers={
                            "sphere": sim_utils.SphereCfg(
                                radius=0.12,
                                visual_material=sim_utils.PreviewSurfaceCfg(
                                    diffuse_color=(1.0, 0.9, 0.0), emissive_color=(0.4, 0.35, 0.0)
                                ),
                            )
                        },
                    )
                )
            self.goal_pos_visualizer.set_visibility(True)
            self._checkpoint_markers.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
            self._checkpoint_markers.set_visibility(False)

    def _debug_vis_callback(self, event):
        # Active target: bright sphere at the current waypoint for every env.
        self.goal_pos_visualizer.visualize(self._desired_pos_w)

        # All checkpoint spheres — only shown for the first _VIS_MAX_ENVS envs
        # to avoid thousands of markers during large-scale training.
        n_vis = min(self.num_envs, self._VIS_MAX_ENVS)
        marker_indices = torch.arange(self._n_waypoints, device=self.device).repeat(n_vis)
        self._checkpoint_markers.visualize(
            self._waypoints_w[:n_vis].reshape(-1, 3),
            marker_indices=marker_indices,
        )
