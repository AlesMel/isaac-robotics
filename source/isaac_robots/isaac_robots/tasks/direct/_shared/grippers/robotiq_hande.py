"""Robotiq Hand-E parallel-jaw gripper (real friction-physics grasp).

The fingers are driven to a commanded width by the articulation's prismatic
slider actuators, and the cube is held *purely by simulated PhysX contact and
friction* -- there is no kinematic attach. Grip force is set by the slider
actuator ``effort_limit`` in ``UR3E_ROBOTIQ_HANDE_CFG`` (assets.py): the fingers
drive toward a closed target and saturate at the effort limit when they press on
the cube, so that effort cap is the steady clamping force.

This deliberately does **not** subclass the suction gripper: a parallel jaw and
a vacuum cup share no grasp physics. It still implements the lightweight gripper
interface the lift env expects (``register_graspable_object``,
``compute_grasp_metrics``, ``is_holding``, ``update_attachment``) so the env
stays gripper-agnostic, but ``update_attachment`` is a no-op -- the object is
held by contact, not by overwriting its pose.

Grasp detection mirrors the repo's real-physics 2F-85 stack task: no contact
sensors, just a geometric heuristic (object inside the jaw region + fingers
clamped). The honest lift signal is the cube's physical height, which the lift
env already rewards.
"""

from __future__ import annotations

import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

from .base import GripperBase, GripperCfg


class RobotiqHandEGripper(GripperBase):
    """Parallel-jaw gripper with a real friction grasp.

    Action layout (per env): ``[close_cmd]`` in ``[-1, 1]``.
        > 0  -> drive fingers closed.
        <= 0 -> drive fingers open.

    Observation layout (per env): ``[is_holding, finger_openness]`` where
    ``finger_openness`` is 0.0 fully closed, 1.0 fully open.
    """

    def __init__(self, cfg: "RobotiqHandEGripperCfg", env) -> None:
        super().__init__(cfg, env)
        self._is_holding = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        self._object = None
        self._tcp_body_id: int | None = None
        self._finger_joint_ids: list[int] | None = None

        self._grasp_offset_w = torch.zeros(self._num_envs, 3, device=self._device)
        self._surface_normal_w = torch.zeros(self._num_envs, 3, device=self._device)
        self._surface_normal_w[:, 2] = 1.0
        self._approach_axis_local = torch.tensor(
            cfg.approach_axis_local, dtype=torch.float, device=self._device
        ).repeat(self._num_envs, 1)

    def setup_scene(self) -> None:
        # Intentionally a no-op. This runs during the env's _setup_scene, which
        # is BEFORE the simulation is played and the articulation's PhysX view
        # exists -- calling find_joints here raises
        # "'Articulation' object has no attribute '_root_physx_view'".
        # The finger slider indices are resolved lazily on first use instead
        # (see _ensure_finger_joints), once the articulation is initialized.
        return

    def _ensure_finger_joints(self) -> None:
        """Resolve the finger slider indices on first use (post-initialization)."""
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

    def register_graspable_object(
        self,
        rigid_object,
        tcp_body_id: int,
        grasp_offset_w: tuple[float, float, float] = (0.0, 0.0, 0.0),
        surface_normal_w: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> None:
        """Register the object the gripper grasps (used for reward metrics only)."""
        self._object = rigid_object
        self._tcp_body_id = tcp_body_id
        self._grasp_offset_w[:] = torch.tensor(grasp_offset_w, dtype=torch.float, device=self._device)

        normal = torch.tensor(surface_normal_w, dtype=torch.float, device=self._device)
        normal = normal / torch.clamp(torch.linalg.norm(normal), min=1.0e-6)
        self._surface_normal_w[:] = normal

    def apply_action(self, action: torch.Tensor) -> None:
        self._ensure_finger_joints()
        # action shape: (num_envs, 1) -- threshold at zero.
        close_cmd = action[:, 0] > 0.0
        targets = torch.where(close_cmd.unsqueeze(-1), self._closed_targets, self._open_targets)
        self._env._robot.set_joint_position_target(targets, joint_ids=self._finger_joint_ids)
        self._update_holding(close_cmd)

    def get_observation(self) -> torch.Tensor:
        self._ensure_finger_joints()
        holding = self._is_holding.float().unsqueeze(-1)
        openness = (1.0 - self._finger_fraction_closed()).clamp(0.0, 1.0).unsqueeze(-1)
        return torch.cat((holding, openness), dim=-1)

    @property
    def is_holding(self) -> torch.Tensor:
        return self._is_holding

    def compute_grasp_metrics(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return distance, alignment cosine, and readiness for the registered object."""
        if self._object is None or self._tcp_body_id is None:
            zeros = torch.zeros(self._num_envs, device=self._device)
            ready = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
            return zeros, zeros, ready

        tcp_quat_w = self._env._robot.data.body_quat_w[:, self._tcp_body_id]
        jaw_center_w = self._jaw_center_pos_w()
        grasp_point_w = self._object.data.root_pos_w + self._grasp_offset_w

        distance = torch.linalg.norm(grasp_point_w - jaw_center_w, dim=-1)
        approach_axis_w = quat_apply(tcp_quat_w, self._approach_axis_local)
        alignment = torch.sum(-approach_axis_w * self._surface_normal_w, dim=-1)
        ready = torch.logical_and(distance <= self.cfg.grasp_distance, alignment >= self.cfg.normal_align_cos)
        return distance, alignment, ready

    def update_attachment(self) -> None:
        """No-op: the cube is held by simulated contact/friction, not by pose overwrite.

        The lift env calls this unconditionally to drive kinematic grippers
        (e.g. suction). A real friction grasp has nothing to snap.
        """
        return

    def reset(self, env_ids: torch.Tensor) -> None:
        self._ensure_finger_joints()
        self._is_holding[env_ids] = False
        # Command the fingers open on reset so a freshly reset env does not
        # inherit the previous episode's closed target for one step.
        self._env._robot.set_joint_position_target(
            self._open_targets[env_ids], joint_ids=self._finger_joint_ids, env_ids=env_ids
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _update_holding(self, close_cmd: torch.Tensor) -> None:
        """Physics-based grasp heuristic: object in the jaw region + fingers clamped.

        No contact sensors (matching the repo's 2F-85 stack task). This only
        shapes the reward / logging -- whether the cube actually rises is
        decided by the simulated grip, not by this flag.
        """
        if self._object is None or self._tcp_body_id is None:
            self._is_holding = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
            return
        _, _, ready = self.compute_grasp_metrics()
        clamped = self._finger_fraction_closed() >= self.cfg.holding_finger_frac
        self._is_holding = torch.logical_and(torch.logical_and(close_cmd, ready), clamped)

    def _jaw_center_pos_w(self) -> torch.Tensor:
        tcp_pos_w = self._env._robot.data.body_pos_w[:, self._tcp_body_id]
        tcp_quat_w = self._env._robot.data.body_quat_w[:, self._tcp_body_id]
        offset = torch.tensor(
            self.cfg.jaw_center_offset_local, dtype=torch.float, device=self._device
        ).repeat(self._num_envs, 1)
        return tcp_pos_w + quat_apply(tcp_quat_w, offset)

    def _finger_fraction_closed(self) -> torch.Tensor:
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
    obs_dim: int = 2

    gripper_joint_names_expr: tuple[str, ...] = ("Slider_.*",)
    """Regex(es) matching the Hand-E finger slider joints. Confirm against the assembled USD."""

    finger_open_pos: float = 0.025
    """Slider target (m) for the fully open command."""

    finger_closed_pos: float = -0.025
    """Slider target (m) for the fully closed command. FLAG: confirm the sign against the USD."""

    jaw_center_offset_local: tuple[float, float, float] = (0.0, 0.0, 0.096)
    """TCP-local offset from tool0 to the jaw center (used for grasp metrics).

    Measured on the assembled USD (build defaults rpy 90 0 0, offset 0 0 0.076):
    the finger pads sit at tool0 +Z ~0.092..0.105 (mean 0.096), with the coupling
    seated at the flange (z~0). Fine-tune for the exact grasp contact point.
    """

    approach_axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0)
    """TCP-local direction that points from the jaws toward the grasped surface."""

    grasp_distance: float = 0.03
    """Max distance (m) from the jaw center to the grasp point to count as 'in the jaws'."""

    normal_align_cos: float = 0.5
    """Min alignment cosine between the approach axis and the surface normal."""

    holding_finger_frac: float = 0.5
    """Fingers must be at least this fraction closed to count as holding (0=open, 1=closed)."""
