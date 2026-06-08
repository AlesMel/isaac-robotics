"""Robotiq Hand-E parallel-jaw gripper (real friction-physics grasp).

Drives the two prismatic slider joints in the assembled UR3e + Hand-E USD
toward an open or closed target. The cube is held purely by simulated PhysX
contact + friction at the blade pads -- there is no kinematic attach. Grip
force is set by the slider actuator's ``effort_limit_sim`` in
``UR3E_ROBOTIQ_HANDE_CFG`` (assets.py); the sliders saturate at that limit
when pressed against the cube, so that effort cap is the steady clamp force.

Lifecycle:
  * ``apply_action(action)`` -- writes the close/open slider target.
  * ``get_observation()`` -- returns ``[finger_openness]`` (0=closed, 1=open).
  * ``reset(env_ids)`` -- commands the fingers open after a reset.

The env owns reward + grasp-detection logic now (see
``UR3eLiftCubeDirectEnv``); the gripper no longer ships
``compute_grasp_metrics`` / ``is_holding`` / ``register_graspable_object`` /
``update_attachment`` / ``_jaw_center_pos_w``. The env computes its EE frame
as ``tool0 + ee_grasp_offset_local`` (Franka-style) and rewards
``tanh(dist(EE, cube)/std)`` directly.
"""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from .base import GripperBase, GripperCfg


class RobotiqHandEGripper(GripperBase):
    """Parallel-jaw gripper with a real friction grasp.

    Action layout (per env): ``[close_cmd]`` in ``[-1, 1]``.
        > 0  -> drive fingers closed.
        <= 0 -> drive fingers open.

    Observation layout (per env): ``[finger_openness]`` where
    ``finger_openness`` is 0.0 fully closed, 1.0 fully open.
    """

    def __init__(self, cfg: "RobotiqHandEGripperCfg", env) -> None:
        super().__init__(cfg, env)
        self._finger_joint_ids: list[int] | None = None

    def setup_scene(self) -> None:
        # No-op: finger slider indices are resolved lazily on first use,
        # after the articulation's PhysX view is initialised. Calling
        # find_joints here raises before the view exists.
        return

    def _ensure_finger_joints(self) -> None:
        """Resolve the finger slider joint indices on first use."""
        if self._finger_joint_ids is not None:
            return
        ids, names = self._env._robot.find_joints(list(self.cfg.gripper_joint_names_expr))
        if not ids:
            raise RuntimeError(
                f"Hand-E finger joints matching {self.cfg.gripper_joint_names_expr} were not "
                f"found on the robot articulation. Available joints: {self._env._robot.joint_names}. "
                "Build the combined UR3e+Hand-E USD with "
                "scripts/ur3e/build_ur3e_robotiq_hande_usd.py and point UR3E_ROBOTIQ_HANDE_USD_PATH "
                "at it (and confirm the slider joint names there)."
            )
        self._finger_joint_ids = ids
        self._finger_joint_names = names
        n = len(ids)
        self._open_targets = torch.full(
            (self._num_envs, n), float(self.cfg.finger_open_pos), device=self._device
        )
        self._closed_targets = torch.full(
            (self._num_envs, n), float(self.cfg.finger_closed_pos), device=self._device
        )

    def apply_action(self, action: torch.Tensor) -> None:
        self._ensure_finger_joints()
        # action shape: (num_envs, 1) -- threshold at zero.
        close_cmd = action[:, 0] > 0.0
        targets = torch.where(close_cmd.unsqueeze(-1), self._closed_targets, self._open_targets)
        self._env._robot.set_joint_position_target(targets, joint_ids=self._finger_joint_ids)

    def get_observation(self) -> torch.Tensor:
        self._ensure_finger_joints()
        return (1.0 - self.finger_fraction_closed()).clamp(0.0, 1.0).unsqueeze(-1)

    def reset(self, env_ids: torch.Tensor) -> None:
        self._ensure_finger_joints()
        # Command the fingers open on reset so a freshly reset env does not
        # inherit the previous episode's closed target for one step.
        self._env._robot.set_joint_position_target(
            self._open_targets[env_ids], joint_ids=self._finger_joint_ids, env_ids=env_ids
        )

    def finger_fraction_closed(self) -> torch.Tensor:
        """0.0 fully open, 1.0 fully closed (mean over the finger sliders)."""
        finger_pos = self._env._robot.data.joint_pos[:, self._finger_joint_ids]
        span = self.cfg.finger_open_pos - self.cfg.finger_closed_pos
        if abs(span) < 1.0e-6:
            return torch.zeros(self._num_envs, device=self._device)
        frac = (self.cfg.finger_open_pos - finger_pos.mean(dim=-1)) / span
        return frac.clamp(0.0, 1.0)


@configclass
class RobotiqHandEGripperCfg(GripperCfg):
    class_type: type = RobotiqHandEGripper
    action_dim: int = 1
    obs_dim: int = 1

    gripper_joint_names_expr: tuple[str, ...] = ("Slider_.*",)
    """Regex(es) matching the Hand-E finger slider joints. Confirm against the assembled USD."""

    finger_open_pos: float = 0.0
    """Slider target (m) for the fully open command (upper joint limit, matches init pose)."""

    finger_closed_pos: float = -0.02
    """Slider target (m) for the fully closed command.

    The local USD overrides the Hand-E slider limits to [-0.02, 0.0] so the
    sliders can only travel in the closing direction from the open rest pose.
    Both sliders use the same target. If you re-edit the USD limits, update
    this value to match the new lower limit.
    """
