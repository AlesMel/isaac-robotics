from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

DEFAULT_CRAZYFLIE_USD = os.getenv(
    "CRAZYFLIE_USD_PATH",
    f"{ISAAC_NUCLEUS_DIR}/Robots/Bitcraze/Crazyflie/cf2x.usd",
)

# UR3e USD lives in source/isaac_robots/data/ur3e/ur3e.usd by default.
# Generate it once with Isaac Lab's URDF converter (see data/ur3e/README.md),
# or override the path with the UR3E_USD_PATH environment variable.
#
# This file is at:   <repo>/source/isaac_robots/isaac_robots/tasks/direct/_shared/assets.py
# parents[0]=_shared, [1]=direct, [2]=tasks, [3]=inner isaac_robots,
# [4]=outer isaac_robots (under source/), [5]=source, [6]=repo root.
_REPO_ROOT = Path(__file__).resolve().parents[6]
DEFAULT_UR3E_USD = os.getenv(
    "UR3E_USD_PATH",
    str(_REPO_ROOT / "source" / "isaac_robots" / "data" / "ur3e" / "ur3e.usd"),
)

CRAZYFLIE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=DEFAULT_CRAZYFLIE_USD,
        activate_contact_sensors=True,
        copy_from_source=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={".*": 0.0},
        joint_vel={
            "m1_joint": 200.0,
            "m2_joint": -200.0,
            "m3_joint": 200.0,
            "m4_joint": -200.0,
        },
    ),
    actuators={
        "dummy": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)


# Universal Robots UR3e (6-DOF arm).
#
# PD gains below are copied from Isaac Lab's UR10e config as a safe starting
# point. The UR3e is roughly 3x lighter than the UR10e, so if you observe
# oscillation or overshoot during training, halve the stiffness and damping
# values and retune from there.
#
# Joint names follow the upstream `ur_description` URDF convention:
#   shoulder_pan_joint, shoulder_lift_joint, elbow_joint,
#   wrist_1_joint, wrist_2_joint, wrist_3_joint
#
# The init_state below puts the arm in a neutral "tucked" pose with the
# end-effector pointing forward and slightly down -- a reasonable starting
# configuration for reaching tasks within the workspace.
UR3E_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=DEFAULT_UR3E_USD,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            # Pin the base to the world. Defense-in-depth: the USD already
            # carries a base->world fixed joint (we converted with
            # ``--fix-base``), and this flag just makes sure the joint stays
            # enabled even if some other code path disables it.
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.5708,
            "elbow_joint": 1.5708,
            "wrist_1_joint": -1.5708,
            "wrist_2_joint": -1.5708,
            "wrist_3_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*"],
            stiffness=1320.0,
            damping=72.66,
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["elbow_joint"],
            stiffness=600.0,
            damping=34.64,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_.*"],
            stiffness=216.0,
            damping=29.39,
        ),
    },
)
