"""UR3e end-effector reaching task.

The policy outputs a 6D vector of absolute joint-position target offsets
(plus optional gripper commands), which are scaled around the default joint
pose, clamped to the soft joint limits, and sent to the implicit PD
controllers as joint position targets. Reward shapes the end-effector toward
a randomly sampled target inside the arm's workspace.

Read from top to bottom -- the lifecycle methods follow Isaac Lab's
``DirectRLEnv`` order:

    __init__  -> _setup_scene  -> _pre_physics_step  -> _apply_action
              -> _get_observations -> _get_rewards -> _get_dones -> _reset_idx
"""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import CUBOID_MARKER_CFG, VisualizationMarkers
from isaaclab.utils.math import quat_error_magnitude, sample_uniform, subtract_frame_transforms

from .._shared.grippers import GripperBase
from .ur3e_reach_env_cfg import UR3eReachEnvCfg


class UR3eReachDirectEnv(DirectRLEnv):
    """Reach a randomized 3D target with the UR3e end-effector."""

    cfg: UR3eReachEnvCfg

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        cfg: UR3eReachEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        # Gym spaces. action_space / observation_space are sized by the cfg
        # (arm + gripper).
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(cfg.observation_space,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(cfg.action_space,), dtype=np.float32
        )

        # The pluggable gripper is created inside _setup_scene (which is
        # called from super().__init__ above), so self._gripper already
        # exists at this point.

        # Cached buffers.
        self._actions = torch.zeros(self.num_envs, cfg.action_space, device=self.device)
        self._prev_actions = torch.zeros_like(self._actions)
        self._target_q = torch.zeros(self.num_envs, 6, device=self.device)
        self._target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._target_quat_w[:, 0] = 1.0

        # Per-term reward sums for episode logging.
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in (
                "position_tracking",
                "position_tracking_fine_grained",
                "orientation_tracking",
                "action_rate",
                "joint_vel",
            )
        }

        # Resolve the end-effector body index once (same across envs).
        ee_match = self._robot.find_bodies(cfg.ee_body_name)
        if not ee_match[0]:
            raise RuntimeError(
                f"End-effector body '{cfg.ee_body_name}' not found on UR3e USD. "
                f"Available: {self._robot.body_names}"
            )
        self._ee_body_id = ee_match[0][0]

        # Cache soft joint limits; shape (num_envs, num_joints, 2).
        soft_limits = self._robot.data.soft_joint_pos_limits[:, :6, :]
        self._joint_lower = soft_limits[..., 0]
        self._joint_upper = soft_limits[..., 1]

        # URDF base_link -> UR "Base" frame conversion. The URDF rotates
        # base_link_inertia 180 deg around z relative to base_link (REP-103),
        # so what Isaac Sim reports as ee_pos in URDF base_link frame is
        # mirrored on x and y relative to what ur_rtde.getActualTCPPose()
        # returns on the real arm. Applying this flip in the observation
        # makes the policy speak the same frame as the hardware -- the
        # deployment script can feed getActualTCPPose() straight through.
        self._ur_base_flip = torch.tensor(
            [-1.0, -1.0, 1.0], device=self.device
        )

        self.set_debug_vis(self.cfg.debug_vis)

    def _canonicalize_quat_wxyz(self, quat: torch.Tensor) -> torch.Tensor:
        """Pick a deterministic representative of the q / -q quaternion pair."""
        mode = self.cfg.ee_quat_sign_mode
        if mode == "raw":
            return quat
        if mode == "positive-w":
            return torch.where(quat[:, 0:1] < 0.0, -quat, quat)
        if mode == "positive-max":
            max_ids = torch.argmax(torch.abs(quat), dim=-1, keepdim=True)
            max_values = torch.gather(quat, 1, max_ids)
            return torch.where(max_values < 0.0, -quat, quat)
        raise ValueError(f"Unsupported ee_quat_sign_mode: {mode}")

    # ------------------------------------------------------------------ #
    # Scene
    # ------------------------------------------------------------------ #
    def _setup_scene(self) -> None:
        # Instantiate the pluggable gripper before the rest of the scene so
        # any extra prims (graspable objects, suction-cup markers) it spawns
        # are cloned into every parallel env below.
        self._gripper: GripperBase = self.cfg.gripper.class_type(
            self.cfg.gripper, self
        )

        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # Plane terrain.
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Let the gripper add its own scene state (objects, markers, ...).
        self._gripper.setup_scene()

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------ #
    # Action pipeline
    # ------------------------------------------------------------------ #
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Convert normalized policy actions into joint position targets."""
        # Defensive reshape: zero_agent.py / random_agent.py pass actions of
        # shape (action_dim,), while the policy at training time provides
        # (num_envs, action_dim). Make sure we always work in 2D.
        if actions.dim() == 1:
            actions = actions.unsqueeze(0).expand(self.num_envs, -1)
        self._prev_actions.copy_(self._actions)
        self._actions = actions.clone().clamp(-1.0, 1.0)

        # Arm: UR10-style absolute joint-position control around the default
        # pose, equivalent to JointPositionActionCfg(use_default_offset=True).
        arm_offset = self._actions[:, :6] * self.cfg.action_scale
        target = self._robot.data.default_joint_pos[:, :6] + arm_offset
        self._target_q = torch.clamp(target, self._joint_lower, self._joint_upper)

        # Gripper: forward whatever the gripper claimed via action_dim.
        if self._gripper.action_dim:
            self._gripper.apply_action(self._actions[:, 6 : 6 + self._gripper.action_dim])

    def _apply_action(self) -> None:
        # Setting the PD targets on the articulation is the single moment
        # of physical control per policy step.
        self._robot.set_joint_position_target(self._target_q, joint_ids=list(range(6)))

    # ------------------------------------------------------------------ #
    # Observations
    # ------------------------------------------------------------------ #
    def _get_observations(self) -> dict[str, torch.Tensor]:
        # Joint state of the 6 arm joints.
        joint_pos = self._robot.data.joint_pos[:, :6]
        joint_vel = self._robot.data.joint_vel[:, :6]

        # End-effector pose, expressed in the robot's base frame so the
        # observation is invariant to where the robot was placed.
        ee_pos_w = self._robot.data.body_pos_w[:, self._ee_body_id]
        ee_quat_w = self._robot.data.body_quat_w[:, self._ee_body_id]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            ee_pos_w,
            ee_quat_w,
        )
        if self.cfg.include_ee_quat:
            ee_quat_b = self._canonicalize_quat_wxyz(ee_quat_b)
        target_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._target_pos_w,
        )

        # Convert to UR "Base" frame so the policy sees the same numbers
        # ur_rtde.getActualTCPPose() reports on the real arm. See the
        # _ur_base_flip comment in __init__.
        ee_pos_b = ee_pos_b * self._ur_base_flip
        target_pos_b = target_pos_b * self._ur_base_flip

        obs_parts = [
            joint_pos,                 # 6
            joint_vel,                 # 6
            ee_pos_b,                  # 3
        ]
        if self.cfg.include_ee_quat:
            obs_parts.append(ee_quat_b)  # 4
        obs_parts.extend(
            [
                target_pos_b,          # 3
                target_pos_b - ee_pos_b,  # 3
            ]
        )
        obs = torch.cat(obs_parts, dim=-1)  # total: 21 or 25

        if self._gripper.obs_dim:
            obs = torch.cat((obs, self._gripper.get_observation()), dim=-1)

        return {"policy": obs}

    # ------------------------------------------------------------------ #
    # Rewards
    # ------------------------------------------------------------------ #
    def _scheduled_reward_weight(self, initial: float, final: float) -> float:
        """Match Isaac Lab's modify_reward_weight curriculum behavior."""
        if self.cfg.reward_curriculum_steps <= 0:
            return final
        if self.common_step_counter > self.cfg.reward_curriculum_steps:
            return final
        return initial

    def _get_rewards(self) -> torch.Tensor:
        ee_pos_w = self._robot.data.body_pos_w[:, self._ee_body_id]
        ee_quat_w = self._robot.data.body_quat_w[:, self._ee_body_id]
        position_error = torch.linalg.norm(self._target_pos_w - ee_pos_w, dim=-1)
        orientation_error = quat_error_magnitude(ee_quat_w, self._target_quat_w)
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
            "position_tracking": (
                position_error * self.cfg.position_tracking_weight * self.step_dt
            ),
            "position_tracking_fine_grained": (
                (1.0 - torch.tanh(position_error / self.cfg.position_tracking_fine_std))
                * self.cfg.position_tracking_fine_weight
                * self.step_dt
            ),
            "orientation_tracking": (
                orientation_error * self.cfg.orientation_tracking_weight * self.step_dt
            ),
            "action_rate": action_rate_l2 * action_rate_weight * self.step_dt,
            "joint_vel": joint_vel_l2 * joint_vel_weight * self.step_dt,
        }
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return torch.stack(list(rewards.values()), dim=0).sum(dim=0)

    # ------------------------------------------------------------------ #
    # Termination
    # ------------------------------------------------------------------ #
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.zeros_like(time_out)
        return died, time_out

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #
    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Log per-episode metrics before zeroing the sums.
        ee_pos_w = self._robot.data.body_pos_w[env_ids, self._ee_body_id]
        final_distance = torch.linalg.norm(
            self._target_pos_w[env_ids] - ee_pos_w, dim=-1
        ).mean()

        # NOTE: keep these as 0-dim torch tensors -- skrl's trainer only
        # logs ``info["log"]`` entries that pass
        # ``isinstance(v, torch.Tensor) and v.numel() == 1``. Plain Python
        # floats are silently dropped, which makes the keys never show up
        # in TensorBoard.
        log: dict[str, torch.Tensor] = {}
        for key, sums in self._episode_sums.items():
            log[f"Episode_Reward/{key}"] = (
                torch.mean(sums[env_ids]) / self.max_episode_length_s
            )
            sums[env_ids] = 0.0

        log["Episode_Termination/died"] = torch.count_nonzero(
            self.reset_terminated[env_ids]
        ).float()
        log["Episode_Termination/time_out"] = torch.count_nonzero(
            self.reset_time_outs[env_ids]
        ).float()
        log["Metrics/final_distance_to_target"] = final_distance
        self.extras["log"] = log

        # Reset robot to its default joint configuration.
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self.scene.update(dt=self.physics_dt)

        # Sample a fresh target inside the workspace box. The cfg ranges
        # are in UR "Base" frame (matching ur_rtde.getActualTCPPose() on
        # hardware), so we apply the same x/y sign flip used in the obs
        # to convert into URDF base_link frame before adding env origin.
        # The stored _target_pos_w is in URDF world coords, which is what
        # the marker and reward use.
        n = len(env_ids)
        target_b = torch.zeros(n, 3, device=self.device)
        target_b[:, 0] = sample_uniform(*self.cfg.target_pos_x_range, (n,), self.device)
        target_b[:, 1] = sample_uniform(*self.cfg.target_pos_y_range, (n,), self.device)
        target_b[:, 2] = sample_uniform(*self.cfg.target_pos_z_range, (n,), self.device)
        target_b = target_b * self._ur_base_flip
        self._target_pos_w[env_ids] = target_b + self._terrain.env_origins[env_ids]

        # The direct task has no pose command manager. Use the reset tool
        # orientation as the desired orientation for the UR10-style
        # orientation tracking term.
        self._target_quat_w[env_ids] = self._robot.data.body_quat_w[
            env_ids, self._ee_body_id
        ].clone()

        # Optional sim-to-real sampling: some episodes start with
        # target=current EE so zero target error appears in the data.
        if self.cfg.target_current_ee_probability > 0.0:
            current_target_mask = (
                torch.rand(n, device=self.device) < self.cfg.target_current_ee_probability
            )
            if torch.any(current_target_mask):
                current_target_env_ids = env_ids[current_target_mask]
                current_target = self._robot.data.body_pos_w[
                    current_target_env_ids, self._ee_body_id
                ].clone()
                if self.cfg.target_current_ee_noise_std > 0.0:
                    current_target += (
                        torch.randn_like(current_target)
                        * self.cfg.target_current_ee_noise_std
                    )
                self._target_pos_w[current_target_env_ids] = current_target

        # Reset task buffers and gripper state.
        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._target_q[env_ids] = joint_pos[:, :6]
        self._gripper.reset(env_ids)

    # ------------------------------------------------------------------ #
    # Debug visualization
    # ------------------------------------------------------------------ #
    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis:
            if not hasattr(self, "_target_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.03, 0.03, 0.03)
                marker_cfg.prim_path = "/Visuals/Command/ur3e_target"
                self._target_visualizer = VisualizationMarkers(marker_cfg)
            self._target_visualizer.set_visibility(True)
        elif hasattr(self, "_target_visualizer"):
            self._target_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        self._target_visualizer.visualize(self._target_pos_w)
