"""UR3e privileged-state cube lifting task."""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import CUBOID_MARKER_CFG, VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import matrix_from_quat, quat_inv, sample_uniform, subtract_frame_transforms

from .._shared.grippers import GripperBase
from .ur3e_lift_cube_env_cfg import UR3eLiftCubeEnvCfg


class UR3eLiftCubeDirectEnv(DirectRLEnv):
    """Lift a cube from a tabletop with privileged cube-pose observations."""

    cfg: UR3eLiftCubeEnvCfg

    def __init__(
        self,
        cfg: UR3eLiftCubeEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(cfg.observation_space,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(cfg.action_space,), dtype=np.float32
        )

        self._actions = torch.zeros(self.num_envs, cfg.action_space, device=self.device)
        self._prev_actions = torch.zeros_like(self._actions)
        self._target_q = torch.zeros(self.num_envs, 6, device=self.device)

        self._ik_controller = DifferentialIKController(
            DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method=cfg.ik_method,
            ),
            num_envs=self.num_envs,
            device=self.device,
        )
        self._ik_action_scale = torch.tensor(
            (
                cfg.tcp_pos_action_scale,
                cfg.tcp_pos_action_scale,
                cfg.tcp_pos_action_scale,
                cfg.tcp_rot_action_scale,
                cfg.tcp_rot_action_scale,
                cfg.tcp_rot_action_scale,
            ),
            device=self.device,
        )

        ee_match = self._robot.find_bodies(cfg.ee_body_name)
        if not ee_match[0]:
            raise RuntimeError(
                f"End-effector body '{cfg.ee_body_name}' not found on UR3e USD. "
                f"Available: {self._robot.body_names}"
            )
        self._ee_body_id = ee_match[0][0]
        self._jacobi_body_id = self._ee_body_id - 1 if self._robot.is_fixed_base else self._ee_body_id
        self._joint_ids = list(range(6))

        soft_limits = self._robot.data.soft_joint_pos_limits[:, :6, :]
        self._joint_lower = soft_limits[..., 0]
        self._joint_upper = soft_limits[..., 1]

        self._ur_base_flip = torch.tensor([-1.0, -1.0, 1.0], device=self.device)

        self._gripper.register_graspable_object(
            self._cube,
            self._ee_body_id,
            grasp_offset_w=(0.0, 0.0, cfg.cube_half_extent),
            surface_normal_w=(0.0, 0.0, 1.0),
        )

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in (
                "approach",
                "align",
                "grasp_contact",
                "lift_height",
                "success",
                "action_rate",
                "joint_vel",
            )
        }

        self.set_debug_vis(self.cfg.debug_vis)

    # ------------------------------------------------------------------ #
    # Scene
    # ------------------------------------------------------------------ #
    def _setup_scene(self) -> None:
        self._gripper: GripperBase = self.cfg.gripper.class_type(self.cfg.gripper, self)

        spawn_ground_plane(
            prim_path=self.cfg.ground_prim_path,
            cfg=GroundPlaneCfg(),
            translation=(0.0, 0.0, self.cfg.ground_z),
        )

        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # Match Isaac Lab's manager-based lift examples: the SeattleLabTable
        # is static USD scene geometry, while only the cube needs a batched
        # RigidObject handle.
        if self.cfg.table.spawn is not None:
            self.cfg.table.spawn.func(
                self.cfg.table.prim_path,
                self.cfg.table.spawn,
                translation=self.cfg.table.init_state.pos,
                orientation=self.cfg.table.init_state.rot,
            )

        self._cube = RigidObject(self.cfg.cube)
        self.scene.rigid_objects["cube"] = self._cube

        self._gripper.setup_scene()

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.ground_prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------ #
    # Actions
    # ------------------------------------------------------------------ #
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions.dim() == 1:
            actions = actions.unsqueeze(0).expand(self.num_envs, -1)
        self._prev_actions.copy_(self._actions)
        self._actions = actions.clone().clamp(-1.0, 1.0)

        ee_pos_b, ee_quat_b = self._compute_ee_pose_b()
        delta_tcp_b = self._actions[:, :6] * self._ik_action_scale

        # Policy deltas use the same UR Base position convention as Reach.
        # Convert vector components back to the URDF/root frame before IK.
        delta_tcp_b[:, :3] *= self._ur_base_flip
        delta_tcp_b[:, 3:6] *= self._ur_base_flip
        self._ik_controller.set_command(delta_tcp_b, ee_pos_b, ee_quat_b)

        if self._gripper.action_dim:
            self._gripper.apply_action(self._actions[:, 6 : 6 + self._gripper.action_dim])

    def _apply_action(self) -> None:
        ee_pos_b, ee_quat_b = self._compute_ee_pose_b()
        joint_pos = self._robot.data.joint_pos[:, self._joint_ids]
        jacobian = self._compute_jacobian_b()
        joint_pos_des = self._ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        self._target_q = torch.clamp(joint_pos_des, self._joint_lower, self._joint_upper)
        self._robot.set_joint_position_target(self._target_q, joint_ids=self._joint_ids)
        if hasattr(self._gripper, "update_collision_body"):
            self._gripper.update_collision_body()
        self._gripper.update_attachment()

    # ------------------------------------------------------------------ #
    # Observations
    # ------------------------------------------------------------------ #
    def _get_observations(self) -> dict[str, torch.Tensor]:
        joint_pos = self._robot.data.joint_pos[:, :6]
        joint_vel = self._robot.data.joint_vel[:, :6]

        ee_pos_b, ee_quat_b = self._compute_ee_pose_b()
        cube_pos_b, cube_quat_b = self._compute_cube_pose_b()

        ee_pos_ur = ee_pos_b * self._ur_base_flip
        cube_pos_ur = cube_pos_b * self._ur_base_flip
        ee_quat_b = self._canonicalize_quat_wxyz(ee_quat_b)
        cube_quat_b = self._canonicalize_quat_wxyz(cube_quat_b)

        obs = torch.cat(
            (
                joint_pos,
                joint_vel,
                ee_pos_ur,
                ee_quat_b,
                cube_pos_ur,
                cube_quat_b,
                cube_pos_ur - ee_pos_ur,
                self._gripper.get_observation(),
            ),
            dim=-1,
        )
        return {"policy": obs}

    # ------------------------------------------------------------------ #
    # Rewards / dones
    # ------------------------------------------------------------------ #
    def _scheduled_reward_weight(self, initial: float, final: float) -> float:
        if self.cfg.reward_curriculum_steps <= 0:
            return final
        if self.common_step_counter > self.cfg.reward_curriculum_steps:
            return final
        return initial

    def _get_rewards(self) -> torch.Tensor:
        distance, alignment, ready = self._gripper.compute_grasp_metrics()
        align_reward = torch.clamp(alignment, 0.0, 1.0)
        holding = self._gripper.is_holding.float()
        cube_height = self._cube_height_env()
        height_above_rest = cube_height - self.cfg.cube_rest_center_z
        lift_progress = torch.clamp(height_above_rest / self.cfg.success_lift_height, 0.0, 1.0)
        success = self._compute_success().float()

        action_rate_l2 = ((self._actions - self._prev_actions) ** 2).sum(dim=-1)
        joint_vel_l2 = (self._robot.data.joint_vel[:, :6] ** 2).sum(dim=-1)
        action_rate_weight = self._scheduled_reward_weight(
            self.cfg.action_rate_l2_weight_initial,
            self.cfg.action_rate_l2_weight_final,
        )
        joint_vel_weight = self._scheduled_reward_weight(
            self.cfg.joint_vel_l2_weight_initial,
            self.cfg.joint_vel_l2_weight_final,
        )

        rewards = {
            "approach": (
                (1.0 - torch.tanh(distance / self.cfg.approach_std))
                * self.cfg.approach_weight
                * self.step_dt
            ),
            "align": align_reward * self.cfg.align_weight * self.step_dt,
            "grasp_contact": (
                (ready.float() * self.cfg.grasp_ready_weight + holding * self.cfg.holding_weight)
                * self.step_dt
            ),
            "lift_height": lift_progress * self.cfg.lift_height_weight * self.step_dt,
            "success": success * self.cfg.success_weight * self.step_dt,
            "action_rate": action_rate_l2 * action_rate_weight * self.step_dt,
            "joint_vel": joint_vel_l2 * joint_vel_weight * self.step_dt,
        }
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return torch.stack(list(rewards.values()), dim=0).sum(dim=0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        success = self._compute_success()
        cube_dropped = self._compute_cube_dropped()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(success, cube_dropped)
        return died, time_out

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #
    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        log: dict[str, torch.Tensor] = {}
        for key, sums in self._episode_sums.items():
            log[f"Episode_Reward/{key}"] = torch.mean(sums[env_ids]) / self.max_episode_length_s
            sums[env_ids] = 0.0

        distance, _, _ = self._gripper.compute_grasp_metrics()
        log["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).float()
        log["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).float()
        log["Episode_Termination/success"] = self._compute_success()[env_ids].float().mean()
        log["Episode_Termination/cube_dropped"] = self._compute_cube_dropped()[env_ids].float().mean()
        log["Metrics/final_cube_height"] = self._cube_height_env()[env_ids].mean()
        log["Metrics/success_rate"] = self._compute_success()[env_ids].float().mean()
        log["Metrics/holding_rate"] = self._gripper.is_holding[env_ids].float().mean()
        log["Metrics/final_tcp_cube_distance"] = distance[env_ids].mean()
        self.extras["log"] = log

        self._robot.reset(env_ids)
        self._cube.reset(env_ids)
        super()._reset_idx(env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        n = len(env_ids)
        cube_pos_ur = torch.zeros(n, 3, device=self.device)
        cube_pos_ur[:, 0] = sample_uniform(*self.cfg.cube_pos_x_range, (n,), self.device)
        cube_pos_ur[:, 1] = sample_uniform(*self.cfg.cube_pos_y_range, (n,), self.device)
        cube_pos_ur[:, 2] = self.cfg.cube_rest_center_z
        cube_pos_w = cube_pos_ur * self._ur_base_flip + self.scene.env_origins[env_ids]

        cube_root_state = self._cube.data.default_root_state[env_ids].clone()
        cube_root_state[:, :3] = cube_pos_w
        cube_root_state[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=torch.float, device=self.device)
        cube_root_state[:, 7:] = 0.0
        self._cube.write_root_pose_to_sim(cube_root_state[:, :7], env_ids=env_ids)
        self._cube.write_root_velocity_to_sim(cube_root_state[:, 7:], env_ids=env_ids)

        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._target_q[env_ids] = joint_pos[:, :6]
        self._ik_controller.reset(env_ids)
        self._gripper.reset(env_ids)
        self.scene.update(dt=self.physics_dt)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _compute_ee_pose_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        return subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._robot.data.body_pos_w[:, self._ee_body_id],
            self._robot.data.body_quat_w[:, self._ee_body_id],
        )

    def _compute_cube_pose_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        return subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._cube.data.root_pos_w,
            self._cube.data.root_quat_w,
        )

    def _compute_jacobian_b(self) -> torch.Tensor:
        jacobian = self._robot.root_physx_view.get_jacobians()[
            :, self._jacobi_body_id, :, self._joint_ids
        ].clone()
        base_rot_matrix = matrix_from_quat(quat_inv(self._robot.data.root_quat_w))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        return jacobian

    def _canonicalize_quat_wxyz(self, quat: torch.Tensor) -> torch.Tensor:
        mode = self.cfg.quat_sign_mode
        if mode == "raw":
            return quat
        if mode == "positive-w":
            return torch.where(quat[:, 0:1] < 0.0, -quat, quat)
        if mode == "positive-max":
            max_ids = torch.argmax(torch.abs(quat), dim=-1, keepdim=True)
            max_values = torch.gather(quat, 1, max_ids)
            return torch.where(max_values < 0.0, -quat, quat)
        raise ValueError(f"Unsupported quat_sign_mode: {mode}")

    def _cube_height_env(self) -> torch.Tensor:
        return self._cube.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]

    def _compute_success(self) -> torch.Tensor:
        return self._cube_height_env() > self.cfg.success_height_threshold

    def _compute_cube_dropped(self) -> torch.Tensor:
        return self._cube_height_env() < self.cfg.cube_drop_height

    def _grasp_point_w(self) -> torch.Tensor:
        return self._cube.data.root_pos_w + torch.tensor(
            (0.0, 0.0, self.cfg.cube_half_extent),
            dtype=torch.float,
            device=self.device,
        )

    # ------------------------------------------------------------------ #
    # Debug visualization
    # ------------------------------------------------------------------ #
    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis:
            if not hasattr(self, "_grasp_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.025, 0.025, 0.025)
                marker_cfg.prim_path = "/Visuals/Command/ur3e_lift_cube_grasp"
                self._grasp_visualizer = VisualizationMarkers(marker_cfg)
            self._grasp_visualizer.set_visibility(True)
            if hasattr(self._gripper, "set_debug_vis"):
                self._gripper.set_debug_vis(True)
        elif hasattr(self, "_grasp_visualizer"):
            self._grasp_visualizer.set_visibility(False)
            if hasattr(self._gripper, "set_debug_vis"):
                self._gripper.set_debug_vis(False)

    def _debug_vis_callback(self, event) -> None:
        del event
        self._grasp_visualizer.visualize(self._grasp_point_w())
        if hasattr(self._gripper, "visualize"):
            self._gripper.visualize()
