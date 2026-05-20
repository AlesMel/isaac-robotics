"""Robotiq 2F-85 parallel-jaw gripper.

Placeholder. Isaac Lab ships ``UR10e_ROBOTIQ_2F_85_CFG`` in
``isaaclab_assets.robots.universal_robots``. That official setup loads the
Robotiq through a robot USD variant, so the arm and gripper are one
articulation. For UR3e, use a combined UR3e+Robotiq USD instead of mounting
the standalone Robotiq USD as a child of the flange.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .base import GripperBase, GripperCfg


class Robotiq2F85Gripper(GripperBase):
    def __init__(self, cfg, env) -> None:
        raise NotImplementedError(
            "Robotiq 2F-85 gripper is not implemented. See SuctionGripper for the "
            "pattern, and isaaclab_assets.robots.universal_robots.UR10e_ROBOTIQ_2F_85_CFG "
            "for the single-articulation robot variant pattern."
        )


@configclass
class Robotiq2F85GripperCfg(GripperCfg):
    class_type: type = Robotiq2F85Gripper
    action_dim: int = 1
    obs_dim: int = 1
