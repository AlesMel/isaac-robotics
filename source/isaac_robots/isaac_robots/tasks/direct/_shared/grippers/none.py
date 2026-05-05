"""No-gripper option -- the arm flies bare. Default for reaching tasks."""

from __future__ import annotations

from isaaclab.utils import configclass

from .base import GripperBase, GripperCfg


class NoGripper(GripperBase):
    """Empty gripper. Adds zero actions and zero observations."""


@configclass
class NoGripperCfg(GripperCfg):
    class_type: type = NoGripper
    action_dim: int = 0
    obs_dim: int = 0
