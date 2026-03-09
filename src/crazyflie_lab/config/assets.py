from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass


DEFAULT_NUCLEUS_ROOT = os.getenv("ISAAC_NUCLEUS_DIR", "omniverse://localhost/NVIDIA/Assets/Isaac/4.5")
DEFAULT_CRAZYFLIE_USD = os.getenv(
    "CRAZYFLIE_USD_PATH",
    f"{DEFAULT_NUCLEUS_ROOT}/Robots/Bitcraze/Crazyflie/crazyflie.usd",
)


@configclass
class CrazyflieAssetCfg(ArticulationCfg):
    prim_path: str = "{ENV_REGEX_NS}/Robot"
    spawn: sim_utils.UsdFileCfg = sim_utils.UsdFileCfg(
        usd_path=DEFAULT_CRAZYFLIE_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_linear_velocity=20.0,
            max_angular_velocity=80.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    )


CRAZYFLIE_CFG = CrazyflieAssetCfg()
