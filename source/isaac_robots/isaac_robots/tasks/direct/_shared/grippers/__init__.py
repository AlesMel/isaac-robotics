"""Pluggable gripper subpackage.

Swap a gripper by changing one line on the env config, e.g.::

    gripper: GripperCfg = NoGripperCfg()         # bare arm
    gripper: GripperCfg = SuctionGripperCfg()    # Robotiq EPick suction
"""

from .base import GripperBase, GripperCfg
from .none import NoGripper, NoGripperCfg
from .robotiq_2f85 import Robotiq2F85Gripper, Robotiq2F85GripperCfg
from .suction import SuctionGripper, SuctionGripperCfg

__all__ = [
    "GripperBase",
    "GripperCfg",
    "NoGripper",
    "NoGripperCfg",
    "SuctionGripper",
    "SuctionGripperCfg",
    "Robotiq2F85Gripper",
    "Robotiq2F85GripperCfg",
]
