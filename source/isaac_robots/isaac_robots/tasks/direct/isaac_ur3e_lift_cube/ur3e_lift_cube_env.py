"""UR3e cube-lifting task (Hand-E gripper).

Reward / observation design follows Isaac Lab's manager-based Franka lift
example. The EE frame is a single offset from the ``tool0`` body, analogous to
Franka's ``panda_hand + (0, 0, 0.1034)`` FrameTransformer. Reward is
``tanh(distance(EE, cube)/std)`` plus a lift bonus and dense goal tracking --
no curriculum on the regularisers, no grasp-detection heuristic on the gripper.

The action layer is delta TCP via DLS-IK + a 1-D gripper close command. The
"UR Base" frame convention (180-deg-about-Z flip between URDF root and the
upstream UR convention) is applied to delta commands and to positions exposed
in the observation.
"""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_inv,
    sample_uniform,
    subtract_frame_transforms,
)

from .._shared.grippers import GripperBase
from .ur3e_lift_cube_env_cfg import UR3eLiftCubeEnvCfg


class UR3eLiftCubeDirectEnv(DirectRLEnv):
    """Reach-grasp-lift-to-goal cube task, NVIDIA Franka-lift style."""

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

        # 180-deg-about-Z mapping between URDF root and "UR Base" convention.
        # Applies to position vectors (and equivalently to axis-angle rotation
        # vectors): (x, y, z) -> (-x, -y, z).
        self._ur_base_flip = torch.tensor([-1.0, -1.0, 1.0], device=self.device)

        # EE-frame offset (tool0-local). Franka analogue: panda_hand + (0,0,0.1034).
        self._ee_grasp_offset_local = torch.tensor(
            cfg.ee_grasp_offset_local, dtype=torch.float, device=self.device
        ).repeat(self.num_envs, 1)

        # Per-episode goal position, sampled at reset in "UR Base" frame.
        self._goal_pos_ur = torch.zeros(self.num_envs, 3, device=self.device)

        reward_keys = [
            "reaching",
            "lifting",
            "object_goal",
            "object_goal_fine",
            "action_rate",
            "joint_vel",
        ]
        if self._contact_sensor is not None:
            reward_keys.append("ee_contact")
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in reward_keys
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

        if self.cfg.table.spawn is not None:
            self.cfg.table.spawn.func(
                self.cfg.table.prim_path,
                self.cfg.table.spawn,
                translation=self.cfg.table.init_state.pos,
                orientation=self.cfg.table.init_state.rot,
            )

        self._cube = RigidObject(self.cfg.cube)
        self.scene.rigid_objects["cube"] = self._cube

        if self.cfg.contact_sensor is not None:
            self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
            self.scene.sensors["contact_sensor"] = self._contact_sensor
        else:
            self._contact_sensor = None

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

        # Policy deltas use the "UR Base" convention; convert back to URDF root.
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

    # ------------------------------------------------------------------ #
    # Observations
    # ------------------------------------------------------------------ #
    def _get_observations(self) -> dict[str, torch.Tensor]:
        # Franka lift PolicyCfg analogue: joint_pos_rel (6), joint_vel (6),
        # object_position_in_robot_root_frame (3), target_object_position (3),
        # last_action (7) plus the gripper observation slice (1 for Hand-E).
        joint_pos_rel = (
            self._robot.data.joint_pos[:, :6] - self._robot.data.default_joint_pos[:, :6]
        )
        joint_vel = self._robot.data.joint_vel[:, :6]

        cube_pos_b, _ = self._compute_cube_pose_b()
        cube_pos_ur = cube_pos_b * self._ur_base_flip

        obs = torch.cat(
            (
                joint_pos_rel,
                joint_vel,
                cube_pos_ur,
                self._goal_pos_ur,
                self._actions,
                self._gripper.get_observation(),
            ),
            dim=-1,
        )
        return {"policy": obs}

    # ------------------------------------------------------------------ #
    # Rewards / dones
    # ------------------------------------------------------------------ #
    def _get_rewards(self) -> torch.Tensor:
        # ---- reaching: tanh-kernel distance from EE frame to cube center ----
        distance = torch.linalg.norm(
            self._cube.data.root_pos_w - self._ee_grasp_pos_w(), dim=-1
        )
        reaching = 1.0 - torch.tanh(distance / self.cfg.reaching_std)

        # ---- lifting: binary, fires once cube clears the lift threshold ----
        cube_height = self._cube_height_env()
        lifted = (cube_height > self.cfg.lifting_min_height).float()

        # ---- goal tracking: gated by `lifted`, dense + fine-grained ----
        cube_pos_env = self._cube.data.root_pos_w - self.scene.env_origins
        # goal lives in "UR Base" frame; convert to env (= URDF root) frame.
        goal_pos_env = self._goal_pos_ur * self._ur_base_flip
        goal_distance = torch.linalg.norm(cube_pos_env - goal_pos_env, dim=-1)
        object_goal = lifted * (1.0 - torch.tanh(goal_distance / self.cfg.goal_tracking_std))
        object_goal_fine = lifted * (
            1.0 - torch.tanh(goal_distance / self.cfg.goal_tracking_fine_std)
        )

        # ---- regularisers (constant weights) ----
        action_rate_l2 = ((self._actions - self._prev_actions) ** 2).sum(dim=-1)
        joint_vel_l2 = (self._robot.data.joint_vel[:, :6] ** 2).sum(dim=-1)

        rewards = {
            "reaching": reaching * self.cfg.reaching_weight,
            "lifting": lifted * self.cfg.lifting_weight,
            "object_goal": object_goal * self.cfg.goal_tracking_weight,
            "object_goal_fine": object_goal_fine * self.cfg.goal_tracking_fine_weight,
            "action_rate": action_rate_l2 * self.cfg.action_rate_l2_weight,
            "joint_vel": joint_vel_l2 * self.cfg.joint_vel_l2_weight,
        }

        # EE / table contact penalty. Reads total normal contact force on the
        # finger bodies (no filter -- table is a static collider; PhysX won't
        # route filtered contacts through it). Threshold > 2 * effort_limit_sim
        # so steady cube-grasp forces don't trigger; hard contacts do.
        if self._contact_sensor is not None:
            nfw = self._contact_sensor.data.net_forces_w_history  # (N, T, B, 3) or None
            if nfw is not None:
                ee_contact_force = torch.max(torch.linalg.norm(nfw, dim=-1), dim=1)[0].sum(dim=-1)
                excess = torch.clamp(ee_contact_force - self.cfg.ee_contact_penalty_threshold, min=0.0)
                rewards["ee_contact"] = excess * self.cfg.ee_contact_penalty_weight

        for key, value in rewards.items():
            self._episode_sums[key] += value
        return torch.stack(list(rewards.values()), dim=0).sum(dim=0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Franka lift terminations: time_out + cube_dropped only. Success
        # doesn't kill the episode -- goal tracking keeps paying out for as
        # long as the cube stays near the target.
        cube_dropped = self._compute_cube_dropped()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return cube_dropped, time_out

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

        cube_height = self._cube_height_env()
        cube_pos_env = self._cube.data.root_pos_w - self.scene.env_origins
        goal_pos_env = self._goal_pos_ur * self._ur_base_flip
        goal_distance = torch.linalg.norm(cube_pos_env - goal_pos_env, dim=-1)
        grasp_distance = torch.linalg.norm(
            self._cube.data.root_pos_w - self._ee_grasp_pos_w(), dim=-1
        )

        log["Episode_Termination/cube_dropped"] = torch.count_nonzero(
            self.reset_terminated[env_ids]
        ).float()
        log["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).float()
        log["Metrics/final_cube_height"] = cube_height[env_ids].mean()
        log["Metrics/lifted_rate"] = (
            (cube_height[env_ids] > self.cfg.lifting_min_height).float().mean()
        )
        log["Metrics/final_cube_to_goal_distance"] = goal_distance[env_ids].mean()
        log["Metrics/final_grasp_distance"] = grasp_distance[env_ids].mean()
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

        # Sample a per-episode lift target in "UR Base" frame.
        goal_ur = torch.zeros(n, 3, device=self.device)
        goal_ur[:, 0] = sample_uniform(*self.cfg.goal_pos_x_range, (n,), self.device)
        goal_ur[:, 1] = sample_uniform(*self.cfg.goal_pos_y_range, (n,), self.device)
        goal_ur[:, 2] = sample_uniform(*self.cfg.goal_pos_z_range, (n,), self.device)
        self._goal_pos_ur[env_ids] = goal_ur

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

    def _cube_height_env(self) -> torch.Tensor:
        return self._cube.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]

    def _compute_cube_dropped(self) -> torch.Tensor:
        return self._cube_height_env() < self.cfg.cube_drop_height

    def _ee_grasp_pos_w(self) -> torch.Tensor:
        """World-frame EE grasp point = ``tool0`` + cfg offset, rotated by tool0's quat.

        Franka analogue: ``FrameTransformer(panda_hand, offset=(0,0,0.1034))``.
        """
        tool0_pos_w = self._robot.data.body_pos_w[:, self._ee_body_id]
        tool0_quat_w = self._robot.data.body_quat_w[:, self._ee_body_id]
        return tool0_pos_w + quat_apply(tool0_quat_w, self._ee_grasp_offset_local)

    def _goal_pos_w(self) -> torch.Tensor:
        return self._goal_pos_ur * self._ur_base_flip + self.scene.env_origins

    # ------------------------------------------------------------------ #
    # Debug visualization
    # ------------------------------------------------------------------ #
    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis:
            if not hasattr(self, "_ee_visualizer"):
                ee_cfg = FRAME_MARKER_CFG.copy()
                ee_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
                ee_cfg.prim_path = "/Visuals/Command/ur3e_lift_cube_ee"
                self._ee_visualizer = VisualizationMarkers(ee_cfg)
            if not hasattr(self, "_goal_visualizer"):
                goal_cfg = FRAME_MARKER_CFG.copy()
                goal_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
                goal_cfg.prim_path = "/Visuals/Command/ur3e_lift_cube_goal"
                self._goal_visualizer = VisualizationMarkers(goal_cfg)
            self._ee_visualizer.set_visibility(True)
            self._goal_visualizer.set_visibility(True)
        else:
            if hasattr(self, "_ee_visualizer"):
                self._ee_visualizer.set_visibility(False)
            if hasattr(self, "_goal_visualizer"):
                self._goal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        del event
        # Weakref-proxied event tap; can fire once after scene tear-down at end-of-run.
        if getattr(self, "scene", None) is None:
            return
        self._ee_visualizer.visualize(self._ee_grasp_pos_w())
        self._goal_visualizer.visualize(self._goal_pos_w())
