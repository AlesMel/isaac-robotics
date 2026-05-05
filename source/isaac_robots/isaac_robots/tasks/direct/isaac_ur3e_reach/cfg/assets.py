"""Re-export the UR3e articulation config from _shared so this task can
import it as ``from .cfg import UR3E_CFG`` -- mirrors the layout used by the
Crazyflie tasks."""

from ..._shared.assets import UR3E_CFG  # noqa: F401
