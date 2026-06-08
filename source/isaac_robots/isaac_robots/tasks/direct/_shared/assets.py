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
DEFAULT_UR3E_ROBOTIQ_2F85_USD = os.getenv(
    "UR3E_ROBOTIQ_2F85_USD_PATH",
    str(_REPO_ROOT / "source" / "isaac_robots" / "data" / "ur3e" / "ur3e_robotiq_2f85.usd"),
)
DEFAULT_UR3E_ROBOTIQ_HANDE_USD = os.getenv(
    "UR3E_ROBOTIQ_HANDE_USD_PATH",
    str(_REPO_ROOT / "source" / "isaac_robots" / "data" / "ur3e" / "ur3e_robotiq_hande.usd"),
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
# PD gains below are a safe starting point for the UR3e. If you observe
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
            enabled_self_collisions=True,
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


# Universal Robots UR3e with a real Robotiq 2F-85 gripper articulation.
#
# This config expects the Robotiq meshes, rigid bodies, joints, and
# mimic/passive joint setup to already be authored into a combined
# UR3e+Robotiq USD where the arm and gripper are one articulation.
# Isaac Sim also ships the standalone gripper at:
#   {ISAAC_NUCLEUS_DIR}/Robots/Robotiq/2F-85/Robotiq_2F_85_edit.usd
# Do not reference that standalone USD under /Robot at runtime: it carries
# its own articulation root.
UR3E_ROBOTIQ_2F85_CFG = UR3E_CFG.copy()
UR3E_ROBOTIQ_2F85_CFG.spawn.usd_path = DEFAULT_UR3E_ROBOTIQ_2F85_USD
UR3E_ROBOTIQ_2F85_CFG.spawn.rigid_props.disable_gravity = True
UR3E_ROBOTIQ_2F85_CFG.spawn.articulation_props.enabled_self_collisions = False
UR3E_ROBOTIQ_2F85_CFG.init_state.joint_pos = UR3E_CFG.init_state.joint_pos.copy()
UR3E_ROBOTIQ_2F85_CFG.init_state.joint_pos.update(
    {
        "finger_joint": 0.0,
        "right_outer_knuckle_joint": 0.0,
        "left_inner_finger_joint": 0.0,
        "right_inner_finger_joint": 0.0,
        "left_inner_finger_knuckle_joint": 0.0,
        "right_inner_finger_knuckle_joint": 0.0,
    }
)
UR3E_ROBOTIQ_2F85_CFG.actuators["gripper_drive"] = ImplicitActuatorCfg(
    joint_names_expr=["finger_joint"],
    effort_limit_sim=10.0,
    velocity_limit_sim=1.0,
    stiffness=11.25,
    damping=0.1,
    friction=0.0,
    armature=0.0,
)
UR3E_ROBOTIQ_2F85_CFG.actuators["gripper_finger"] = ImplicitActuatorCfg(
    joint_names_expr=[".*_inner_finger_joint"],
    effort_limit_sim=1.0,
    velocity_limit_sim=1.0,
    stiffness=0.2,
    damping=0.001,
    friction=0.0,
    armature=0.0,
)
UR3E_ROBOTIQ_2F85_CFG.actuators["gripper_passive"] = ImplicitActuatorCfg(
    joint_names_expr=[".*_inner_finger_knuckle_joint", "right_outer_knuckle_joint"],
    effort_limit_sim=1.0,
    velocity_limit_sim=1.0,
    stiffness=0.0,
    damping=0.0,
    friction=0.0,
    armature=0.0,
)


# Universal Robots UR3e with a real Robotiq Hand-E parallel-jaw gripper articulation.
#
# Like the 2F-85 config above, this expects a combined UR3e+Hand-E USD where the
# arm and gripper are one articulation (build it with
# ``scripts/ur3e/build_ur3e_robotiq_hande_usd.py``). The Hand-E drives two
# prismatic finger sliders; unlike the 2F-85 it has no revolute knuckle/mimic
# chain.
#
# NOTE: The slider joint names below ("Slider_1", "Slider_2") follow the Isaac
# Sim ``Robotiq_Hand_E_edit.usd`` convention. If your assembled USD names them
# differently, update the names here *and* ``RobotiqHandEGripperCfg`` -- the
# build script logs ``robot.joint_names`` so you can confirm them.
UR3E_ROBOTIQ_HANDE_CFG = UR3E_CFG.copy()
UR3E_ROBOTIQ_HANDE_CFG.spawn.usd_path = DEFAULT_UR3E_ROBOTIQ_HANDE_USD
UR3E_ROBOTIQ_HANDE_CFG.spawn.rigid_props.disable_gravity = True
UR3E_ROBOTIQ_HANDE_CFG.spawn.articulation_props.enabled_self_collisions = False
# Required by ContactSensorCfg on the finger bodies (table-collision penalty).
UR3E_ROBOTIQ_HANDE_CFG.spawn.activate_contact_sensors = True
UR3E_ROBOTIQ_HANDE_CFG.init_state.joint_pos = UR3E_CFG.init_state.joint_pos.copy()
UR3E_ROBOTIQ_HANDE_CFG.init_state.joint_pos.update(
    {
        "Slider_1": 0.0,
        "Slider_2": 0.0,
    }
)
# Real friction grasp: the sliders drive toward a closed target with stiff
# position tracking and saturate at ``effort_limit_sim`` when they press on the
# cube, so that effort cap is the steady clamping force. Tune ``effort_limit_sim``
# (grip force, N) if the cube slips (raise) or gets flung (lower); the cube is
# held by simulated contact + friction, not a kinematic attach.
UR3E_ROBOTIQ_HANDE_CFG.actuators["gripper_slide"] = ImplicitActuatorCfg(
    joint_names_expr=["Slider_.*"],
    effort_limit_sim=20.0,
    velocity_limit_sim=0.2,
    stiffness=2000.0,
    damping=100.0,
    friction=0.0,
    armature=0.0,
)
