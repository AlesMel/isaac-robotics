"""Robotiq 2F-85 parallel-jaw gripper.

Placeholder. Isaac Lab ships ``UR10e_ROBOTIQ_2F_85_CFG`` in
``isaaclab_assets.robots.universal_robots`` -- the implementation here would
mount that articulation as a child of the UR3e flange and bind the finger
joints to a single ``[open, close]`` action. Left as a follow-up so the
GripperBase extension point is documented.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .base import GripperBase, GripperCfg


class Robotiq2F85Gripper(GripperBase):
    def __init__(self, cfg, env) -> None:
        raise NotImplementedError(
            "Robotiq 2F-85 gripper is not implemented. See SuctionGripper for the "
            "pattern, and isaaclab_assets.robots.universal_robots.UR10e_ROBOTIQ_2F_85_CFG "
            "for the articulation config to mount on the UR3e flange."
        )


@configclass
class Robotiq2F85GripperCfg(GripperCfg):
    class_type: type = Robotiq2F85Gripper
    action_dim: int = 1
    obs_dim: int = 1
