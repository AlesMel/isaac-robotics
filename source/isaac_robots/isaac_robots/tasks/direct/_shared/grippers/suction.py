"""Single-cup suction gripper (Robotiq EPick-style).

Grasping model -- read this before extending:

    The cup does NOT use a real fixed joint at runtime. While suction is
    "on" and a registered graspable object is within ``grasp_distance`` of
    the TCP, the object's root pose is overwritten each physics step so it
    tracks the TCP transform. This is a kinematic approximation -- there is
    no slip, no leak, no finite hold force. Any policy that exploits magic
    adhesion in sim will fail on the real robot.

For this v0 release the implementation is a stub: the action and observation
slots are reserved (so the modular abstraction is exercised end-to-end) but
no attachment logic runs. The Pick task will fill this in.
"""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from .base import GripperBase, GripperCfg


class SuctionGripper(GripperBase):
    """Kinematic single-cup suction gripper.

    Action layout (per env): ``[suction_cmd]`` in ``[-1, 1]``.
        > 0 -> suction on, attempt to grasp.
        <= 0 -> suction off, release.

    Observation layout (per env): ``[is_holding]`` (0.0 or 1.0).
    """

    def __init__(self, cfg: "SuctionGripperCfg", env) -> None:
        super().__init__(cfg, env)
        self._is_on = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        self._is_holding = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)

    def reset(self, env_ids: torch.Tensor) -> None:
        self._is_on[env_ids] = False
        self._is_holding[env_ids] = False

    def apply_action(self, action: torch.Tensor) -> None:
        # action shape: (num_envs, 1) -- threshold at zero.
        self._is_on = action[:, 0] > 0.0
        # Attachment / release logic lands with the Pick task. For now the
        # cup just tracks its own on/off state so the observation channel
        # carries useful information during Reach.
        self._is_holding = self._is_on

    def get_observation(self) -> torch.Tensor:
        return self._is_holding.float().unsqueeze(-1)


@configclass
class SuctionGripperCfg(GripperCfg):
    class_type: type = SuctionGripper
    action_dim: int = 1
    obs_dim: int = 1

    tcp_frame_name: str = "tool0"
    """Body frame of the suction cup tip on the arm."""

    grasp_distance: float = 0.02
    """Max distance (m) from TCP to object center for grasp to succeed."""

    normal_align_cos: float = 0.7
    """Min dot-product between cup z-axis and surface normal for grasp."""
