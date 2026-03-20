from __future__ import annotations

import heapq
import os
import numpy as np
import torch
from gymnasium import spaces

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_yaw, subtract_frame_transforms

from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

from .obstacle_nav_env_cfg import ObstacleNavEnvCfg
import logging

logger = logging.getLogger(__name__)


# ── A* Pathfinding ───────────────────────────────────────────────────────────

def _astar_path_3d(occupancy: np.ndarray, start_idx: tuple, goal_idx: tuple) -> list[tuple]:
    """A* shortest path on a 3D occupancy grid with 26-connected neighbors.

    Args:
        occupancy: 3D boolean array (True = blocked).
        start_idx: (x, y, z) start voxel indices.
        goal_idx:  (x, y, z) goal voxel indices.

    Returns:
        List of (x, y, z) voxel indices from start to goal, or [] if unreachable.
    """
    nx, ny, nz = occupancy.shape

    def _nearest_free(idx):
        if not occupancy[idx]:
            return idx
        free = np.argwhere(~occupancy)
        dists = np.linalg.norm(free - np.array(idx), axis=1)
        nearest = tuple(free[dists.argmin()])
        logger.warning(f"Voxel {idx} is blocked; snapped to {nearest}")
        return nearest

    start_idx = _nearest_free(start_idx)
    goal_idx = _nearest_free(goal_idx)
    gx, gy, gz = goal_idx

    # 3-D octile heuristic (admissible for 26-connected grids)
    def h(pos):
        vals = sorted([abs(pos[0] - gx), abs(pos[1] - gy), abs(pos[2] - gz)])
        return (1.7320508 - 1.4142136) * vals[0] + (1.4142136 - 1.0) * vals[1] + vals[2]

    open_set = [(h(start_idx), 0.0, start_idx)]
    g_score = {start_idx: 0.0}
    came_from = {}
    closed = set()

    while open_set:
        _f, g, cur = heapq.heappop(open_set)
        if cur == goal_idx:
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            return path[::-1]
        if cur in closed:
            continue
        closed.add(cur)
        cx, cy, cz = cur
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    x2, y2, z2 = cx + dx, cy + dy, cz + dz
                    if 0 <= x2 < nx and 0 <= y2 < ny and 0 <= z2 < nz:
                        nb = (x2, y2, z2)
                        if nb in closed or occupancy[x2, y2, z2]:
                            continue
                        cost = (dx * dx + dy * dy + dz * dz) ** 0.5
                        ng = g + cost
                        if nb not in g_score or ng < g_score[nb]:
                            g_score[nb] = ng
                            came_from[nb] = cur
                            heapq.heappush(open_set, (ng + h(nb), ng, nb))
    return []


def _subsample_path(path: np.ndarray, spacing: float) -> np.ndarray:
    """Subsample a world-coordinate path so consecutive points are ~*spacing* apart."""
    if len(path) <= 2:
        return path
    result = [path[0]]
    accum = 0.0
    for i in range(1, len(path)):
        accum += np.linalg.norm(path[i] - path[i - 1])
        if accum >= spacing:
            result.append(path[i])
            accum = 0.0
    if not np.allclose(result[-1], path[-1]):
        result.append(path[-1])
    return np.array(result)


# ── Chase Camera ─────────────────────────────────────────────────────────────

from isaaclab.envs.ui import ViewportCameraController


class ChaseCameraController(ViewportCameraController):
    """Viewport camera that rotates eye/lookat offsets with the tracked asset's orientation."""

    def _update_tracking_callback(self, event):
        if self.cfg.origin_type != "asset_root" or self.cfg.asset_name is None:
            super()._update_tracking_callback(event)
            return

        asset = self._env.scene[self.cfg.asset_name]
        idx = self.cfg.env_index
        # update origin to current robot position
        self.viewer_origin = asset.data.root_pos_w[idx]
        origin = self.viewer_origin.detach().cpu().numpy()

        # rotate the static offsets by the robot's orientation
        quat = asset.data.root_quat_w[idx].unsqueeze(0)  # (1, 4) wxyz
        eye_t = torch.as_tensor(self.default_cam_eye, dtype=torch.float32, device=quat.device).unsqueeze(0)
        lookat_t = torch.as_tensor(self.default_cam_lookat, dtype=torch.float32, device=quat.device).unsqueeze(0)

        cam_eye = origin + quat_apply_yaw(quat, eye_t).squeeze(0).detach().cpu().numpy()
        cam_target = origin + quat_apply_yaw(quat, lookat_t).squeeze(0).detach().cpu().numpy()
        self._env.sim.set_camera_view(eye=cam_eye, target=cam_target)


# ── Environment ──────────────────────────────────────────────────────────────

class ObstacleNavDirectEnv(DirectRLEnv):
    cfg: ObstacleNavEnvCfg

    def __init__(self, cfg: ObstacleNavEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        # Replace the default camera controller with a chase camera that
        # rotates the eye/lookat offsets with the drone's orientation.
        if self.viewport_camera_controller is not None:
            del self.viewport_camera_controller
            self.viewport_camera_controller = ChaseCameraController(self, self.cfg.viewer)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.cfg.observation_space,), dtype=np.float32,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self._actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel", "ang_vel", "path_progress", "goal_reached",
                "tilt", "action_smoothness", "alive", "proximity",
            ]
        }
        self._goal_offsets = torch.tensor(
            [
                [0.0, -2.0, 3.0],   # GOAL0
                [-4.5, 3.5, 7.0],   # GOAL1
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

        # Grace period after reset
        _grace_seconds = 0.3
        self._reset_grace_steps_remaining = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._reset_grace_duration = int(_grace_seconds / (self.cfg.decimation * self.cfg.sim.dt))

        # ── A* path planning ────────────────────────────────────────────────
        _voxel_dir = os.path.join(os.path.dirname(__file__), "../../../../../../voxel_output")
        _data = np.load(os.path.join(_voxel_dir, "distance_field.npz"))
        occupancy = _data["occupancy"]                              # (X, Y, Z) bool
        grid_origin = _data["origin"].astype(np.float64)
        grid_res = float(_data["resolution"])

        spawn_local = np.array(self.cfg.robot.init_state.pos, dtype=np.float64)
        goals_local = self._goal_offsets.cpu().numpy().astype(np.float64)

        def _w2v(pos):
            idx = np.round((pos - grid_origin) / grid_res).astype(int)
            return tuple(np.clip(idx, 0, np.array(occupancy.shape) - 1))

        # Precompute path segments: spawn->g0, g0->g1, g1->g0
        path_pairs = [
            (spawn_local, goals_local[0]),
            (goals_local[0], goals_local[1]),
            (goals_local[1], goals_local[0]),
        ]

        paths_local: list[np.ndarray] = []
        for i, (start, end) in enumerate(path_pairs):
            logger.info("Computing A* path segment %d: %s -> %s", i, start, end)
            voxel_path = _astar_path_3d(occupancy, _w2v(start), _w2v(end))
            if not voxel_path:
                raise RuntimeError(f"A* found no path from {start} to {end}!")
            world_path = np.array(voxel_path, dtype=np.float64) * grid_res + grid_origin
            subsampled = _subsample_path(world_path, spacing=self.cfg.path_subsample_spacing)
            logger.info("  Segment %d: %d voxels -> %d waypoints", i, len(voxel_path), len(subsampled))
            paths_local.append(subsampled)

        max_path_len = max(len(p) for p in paths_local)
        num_segments = len(path_pairs)
        self._astar_paths = torch.zeros(num_segments, max_path_len, 3, device=self.device)
        self._astar_path_lens = torch.zeros(num_segments, dtype=torch.long, device=self.device)
        for i, p in enumerate(paths_local):
            self._astar_paths[i, :len(p)] = torch.tensor(p, dtype=torch.float32, device=self.device)
            self._astar_path_lens[i] = len(p)

        # Remaining-distance-to-end lookup: dist_to_end[seg, k] = path length from wp k to final wp
        self._astar_dist_to_end = torch.zeros(num_segments, max_path_len, device=self.device)
        for i in range(num_segments):
            L = int(self._astar_path_lens[i].item())
            for j in range(L - 2, -1, -1):
                seg_d = torch.linalg.norm(self._astar_paths[i, j + 1] - self._astar_paths[i, j]).item()
                self._astar_dist_to_end[i, j] = self._astar_dist_to_end[i, j + 1].item() + seg_d

        # Per-env path tracking
        self._path_segment = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._path_wp_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._prev_remaining = torch.full((self.num_envs,), float("inf"), device=self.device)

        self.set_debug_vis(self.cfg.debug_vis)

    # ── path helpers ─────────────────────────────────────────────────────────

    def _path_target_local(self) -> torch.Tensor:
        """Current path-waypoint target in env-local coords for every env."""
        return self._astar_paths[self._path_segment, self._path_wp_idx]  # (N, 3)

    def _remaining_distance(self, pos_local: torch.Tensor) -> torch.Tensor:
        """Distance from *pos_local* to end of the current path segment."""
        target = self._path_target_local()
        d_to_target = torch.linalg.norm(pos_local - target, dim=-1)
        d_to_end = self._astar_dist_to_end[self._path_segment, self._path_wp_idx]
        return d_to_target + d_to_end

    # ── scene ────────────────────────────────────────────────────────────────

    def _setup_scene(self) -> None:
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        warehouse_prim_path = self.cfg.warehouse.prim_path.replace("env_.*", "env_0")
        self.cfg.warehouse.spawn.func(
            warehouse_prim_path,
            self.cfg.warehouse.spawn,
            translation=self.cfg.warehouse.init_state.pos,
            orientation=self.cfg.warehouse.init_state.rot,
        )

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        if self.cfg.lidar is not None:
            self._lidar = self.cfg.lidar.class_type(self.cfg.lidar)
            self.scene.sensors["lidar"] = self._lidar
        else:
            self._lidar = None

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self.scene.clone_environments(copy_from_source=False)
        self._env_origins = self._terrain.env_origins

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path, self.cfg.warehouse.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ── physics / actions ────────────────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor):
        self._previous_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self) -> None:
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id,
            forces=self._thrust,
            torques=self._moment,
        )

    # ── observations ─────────────────────────────────────────────────────────

    def _get_observations(self) -> dict[str, torch.Tensor]:
        # Direction to next path waypoint in body frame
        path_target_w = self._path_target_local() + self._env_origins
        target_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            path_target_w,
        )

        obs_list = [
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            self._robot.data.projected_gravity_b,
            target_pos_b,
        ]

        if self._lidar is not None:
            ray_hits_w = self._lidar.data.ray_hits_w
            lidar_origin_w = self._lidar.data.pos_w.unsqueeze(1)

            lidar_ranges = torch.linalg.norm(ray_hits_w - lidar_origin_w, dim=-1)
            max_dist = self.cfg.sensor_selection.lidar_max_distance_m

            lidar_ranges.div_(max_dist)
            lidar_ranges.nan_to_num_(nan=1.0, posinf=1.0, neginf=0.0)
            lidar_ranges.clamp_(0.0, 1.0)

            obs_list.append(lidar_ranges.view(self.num_envs, -1))

        obs = torch.cat(obs_list, dim=-1)
        return {"policy": obs}

    # ── rewards ──────────────────────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        root_pos_w = self._robot.data.root_pos_w
        pos_local = root_pos_w - self._env_origins
        lin_vel_b = self._robot.data.root_lin_vel_b
        ang_vel_b = self._robot.data.root_ang_vel_b
        proj_grav = self._robot.data.projected_gravity_b
        dt = self.step_dt

        # 1. Advance path waypoints when the drone is close enough
        target = self._path_target_local()
        dist_to_target = torch.linalg.norm(pos_local - target, dim=-1)
        close_mask = dist_to_target < self.cfg.path_advance_threshold

        goal_reached = torch.zeros(self.num_envs, device=self.device)

        if close_mask.any():
            close_ids = close_mask.nonzero(as_tuple=False).squeeze(-1)
            new_wp = self._path_wp_idx[close_ids] + 1
            path_lens = self._astar_path_lens[self._path_segment[close_ids]]

            reached_end = new_wp >= path_lens
            still_on_path = ~reached_end

            # Advance intermediate waypoints
            if still_on_path.any():
                ids = close_ids[still_on_path]
                self._path_wp_idx[ids] = new_wp[still_on_path]

            # Final waypoint reached = goal reached -> cycle to next goal segment
            if reached_end.any():
                ids = close_ids[reached_end]
                goal_reached[ids] = 1.0

                old_wp = self._waypoint_idx[ids]
                new_wp_idx = (old_wp + 1) % self._num_waypoints
                self._waypoint_idx[ids] = new_wp_idx
                self._desired_pos_w[ids] = self._waypoints_w[ids, new_wp_idx]

                # Assign next path segment: after reaching goal0 use seg 1, after goal1 use seg 2
                self._path_segment[ids] = old_wp + 1   # 0->1, 1->2
                self._path_wp_idx[ids] = 0
                self._prev_remaining[ids] = float("inf")

        # 2. Path progress (primary navigation signal)
        remaining = self._remaining_distance(pos_local)
        path_progress = torch.where(
            torch.isinf(self._prev_remaining),
            torch.zeros_like(remaining),
            self._prev_remaining - remaining,
        ).clamp(-1.0, 1.0)
        self._prev_remaining = remaining.detach()

        # 3. Proximity penalty from ToF sensor ranges
        proximity_penalty = torch.zeros(self.num_envs, device=self.device)
        if self._lidar is not None:
            ray_hits_w = self._lidar.data.ray_hits_w
            lidar_origin_w = self._lidar.data.pos_w.unsqueeze(1)
            lidar_ranges = torch.linalg.norm(ray_hits_w - lidar_origin_w, dim=-1)
            min_range, _ = lidar_ranges.min(dim=-1)
            # Smooth penalty: 1.0 when touching wall, 0.0 at safety distance
            proximity_penalty = torch.clamp(
                1.0 - min_range / self.cfg.obstacle_safety_distance, min=0.0
            )

        # 4. Penalties
        tilt_error = 1.0 + proj_grav[:, 2]
        lin_vel_sq = torch.sum(torch.square(lin_vel_b), dim=1)
        ang_vel_sq = torch.sum(torch.square(ang_vel_b), dim=1)
        action_diff_sq = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)

        rewards = {
            "path_progress": path_progress * self.cfg.distance_to_goal_reward_scale,
            "goal_reached": goal_reached * self.cfg.goal_reached_bonus,
            "alive": torch.full((self.num_envs,), self.cfg.alive_bonus * dt, device=self.device),
            "proximity": proximity_penalty * self.cfg.obstacle_proximity_reward_scale * dt,
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
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        collision = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self._body_id], dim=-1), dim=1)[0]
            > self.cfg.collision_force_threshold,
            dim=1,
        )
        grace_active = self._reset_grace_steps_remaining > 0
        self._reset_grace_steps_remaining[grace_active] -= 1
        died = collision & ~grace_active
        return died, time_out

    # ── reset ────────────────────────────────────────────────────────────────

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
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()

        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        final_rem = self._prev_remaining[env_ids]
        self.extras["log"]["Metrics/final_path_remaining"] = final_rem.nan_to_num(nan=0.0, posinf=0.0).mean().item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._thrust[env_ids] = 0.0
        self._moment[env_ids] = 0.0

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset path tracking: start at segment 0 (spawn -> goal0)
        self._path_segment[env_ids] = 0
        self._path_wp_idx[env_ids] = 0
        self._prev_remaining[env_ids] = float("inf")
        self._reset_grace_steps_remaining[env_ids] = self._reset_grace_duration

        self._waypoints_w[env_ids] = self._env_origins[env_ids].unsqueeze(1) + self._goal_offsets.unsqueeze(0)
        self._waypoint_idx[env_ids] = 0
        self._desired_pos_w[env_ids] = self._waypoints_w[env_ids, 0]

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
