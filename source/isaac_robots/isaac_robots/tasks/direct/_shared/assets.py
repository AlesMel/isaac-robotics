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

UR3E_2F85_USD = os.getenv(
    "UR3E_ROBOTIQ_2F85_USD_PATH",
    str(_REPO_ROOT / "source" / "isaac_robots" / "data" / "ur3e" / "ur3e_robotiq_2f85.usd"),
)

UR3E_2F140_USD = os.getenv(
    "UR3E_ROBOTIQ_2F140_USD_PATH",
    str(_REPO_ROOT / "source" / "isaac_robots" / "data" / "ur3e" / "ur3e_robotiq_2f140.usd"),
)

UR3e_ROBOTIQ_2F85_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=UR3E_2F85_USD,
        # NOTE: manually-assembled USD => NO `variants={...}` here.
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # Robot links weightless (implicit gravity comp) — cube still has gravity.
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            # Grippers need high iteration counts to stay stable on contact.
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=16,
        ),
        activate_contact_sensors=True,  # useful for grasp detection / contact rewards
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # "Ready over table" pose: TCP hovers at ~(0.40, 0.00, 0.20) m in the base
        # frame, gripper pointing straight down, centered over the cube reset region.
        # This is critical for RL: with the old tucked pose [0,-1.57,1.57,1.57,1.57,0]
        # the TCP started ~0.32 m from the cube, in the dead zone of the
        # `1 - tanh(d/0.1)` reaching reward (reward ~0.003, no gradient) -> the policy
        # never bootstrapped and training was flat. Verified via UR3e FK from the URDF.
        joint_pos={
            "shoulder_pan_joint": -0.33,
            "shoulder_lift_joint": -1.21,
            "elbow_joint": 0.97,
            "wrist_1_joint": -1.33,
            "wrist_2_joint": -1.57,
            "wrist_3_joint": -0.34,
            # 0.0 == OPEN by the standard Robotiq convention — VERIFY (see header).
            "finger_joint": 0.0,
        },
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={
        # ---------------- arm (values from the stock UR3e cfg) ----------------
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*"],
            stiffness=1320.0,
            damping=72.6636085,
            friction=0.0,
            armature=0.0,
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["elbow_joint"],
            stiffness=600.0,
            damping=34.64101615,
            friction=0.0,
            armature=0.0,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_.*"],
            stiffness=216.0,
            damping=29.39387691,
            friction=0.0,
            armature=0.0,
        ),

        # ---------------- gripper: the single driven joint ----------------
        # Gains copied from Isaac Lab's known-good FRANKA_ROBOTIQ_GRIPPER_CFG
        # "gripper_drive" (same Robotiq 2F-85). The old effort_limit_sim=10 capped
        # grip force ~165x too low -> the drive saturated under load, the gripper
        # never closed/held firmly, and the policy resorted to scooping. The 5
        # passive linkage joints stay UN-actuated: they're driven by the USD PhysX
        # mimic (verified working), and adding drives would fight that constraint.
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=1650.0,
            velocity_limit_sim=10.0,
            stiffness=17.0,
            damping=0.02,
            friction=0.0,
            armature=0.0,
        ),
        # # ---------------- gripper: light auxiliary joints ----------------
        # # Help the underactuated linkage track the driver.
        # "gripper_finger": ImplicitActuatorCfg(
        #     joint_names_expr=[".*_inner_finger_joint"],
        #     effort_limit_sim=1.0,
        #     velocity_limit_sim=1.0,
        #     stiffness=0.2,
        #     damping=0.001,
        #     friction=0.0,
        #     armature=0.0,
        # ),
        
        # # ---------------- gripper: passive joints ----------------
        # # Driven by the USD mimic constraints; zero stiffness/damping here.
        # "gripper_passive": ImplicitActuatorCfg(
        #     joint_names_expr=[
        #         ".*_inner_finger_knuckle_joint",   # left_ + right_inner_finger_knuckle_joint
        #         "right_outer_knuckle_joint",
        #     ],
        #     effort_limit_sim=1.0,
        #     velocity_limit_sim=1.0,
        #     stiffness=0.0,
        #     damping=0.0,
        #     friction=0.0,
        #     armature=0.0,
        # ),
    },
)


# ---------------------------------------------------------------------------
# UR3e + Robotiq 2F-140 (wider-stroke sibling of the 2F-85)
# ---------------------------------------------------------------------------
# Duplicated from UR3e_ROBOTIQ_2F85_CFG; the GRIPPER SCHEME IS THE OPPOSITE.
#   * USD asset (assembled ur3e_robotiq_2f140.usd, payloads NVIDIA's 2F-140).
#   * finger_joint range is 0..45 deg (0..0.785 rad) vs the 2F-85's 0..47 deg
#     -> the env's close_command must use 0.785, not 0.82.
#   * NO MIMIC, NO LOOP CLOSURE. Unlike the 2F-85 (which composes
#     Robotiq_2F_85_phyisics_mimic.usda + _phyisics_loop.usda), NVIDIA's
#     Robotiq_2F_140_physics_edit ships zero mimic joints (verified across every
#     Nucleus 2F-140 variant). So you CANNOT drive finger_joint alone: the other
#     9 gripper joints would be left at the asset's weak/zero USD gains, uncoupled
#     and floppy -> the fingers barely move. This is exactly the "doesn't move" bug.
#   * FIX (IsaacLab Discussions #4124 / #1908, same UR3e+2F-140 asset): ACTUATE ALL
#     gripper joints in 3 groups and INITIALISE all of them (kinematic loop needs it).
#     Use the asset's WEAK finger_joint gains (k=11.25/maxF=10) -- the 2F-85's strong
#     k=17/maxF=1650 over-powers the loop and kicks the arm back when grasping.
#     NB: the gripper_finger/gripper_passive groups that are a TRAP for the 2F-85
#     (they fight its mimic) are exactly what the 2F-140 NEEDS (no mimic to fight).
UR3e_ROBOTIQ_2F140_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=UR3E_2F140_USD,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=16,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Same FK-derived "ready over table" arm pose as the 2F-85 (identical UR3e arm).
        # The 2F-140 is a little longer, so the TCP sits slightly lower with this pose;
        # still well inside the reaching-reward zone.
        joint_pos={
            "shoulder_pan_joint": -0.33,
            "shoulder_lift_joint": -1.21,
            "elbow_joint": 0.97,
            "wrist_1_joint": -1.33,
            "wrist_2_joint": -1.57,
            "wrist_3_joint": -0.34,
            # ALL 8 articulated gripper joints must be initialised (kinematic loop).
            # NB: the *_inner_knuckle_joints are loop-closure joints and are NOT in
            # the PhysX articulation tree, so they must NOT appear here.
            "finger_joint": 0.0,  # 0.0 == OPEN; the single driven joint
            ".*_inner_finger_joint": 0.0,
            ".*_inner_finger_pad_joint": 0.0,
            ".*_outer_finger_joint": 0.0,
            ".*_outer_knuckle_joint": 0.0,
        },
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*"], stiffness=1320.0, damping=72.6636085, friction=0.0, armature=0.0,
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["elbow_joint"], stiffness=600.0, damping=34.64101615, friction=0.0, armature=0.0,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_.*"], stiffness=216.0, damping=29.39387691, friction=0.0, armature=0.0,
        ),
        # ---- 2F-140 gripper: actuate ALL joints (no mimic to couple them) ----
        # The single driven joint. WEAK gains on purpose (asset's authored values);
        # strong gains here destabilise the loop and kick the arm back on contact.
        "gripper_drive": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            # Grip FORCE. The linkage closes fine (loop closure verified), but at
            # effort=10 the drive saturated on contact and couldn't hold -> scoop.
            # 1650 (the 2F-85 value) holds, BUT the 2F-85 absorbs the closing
            # reaction in its MIMIC; the 2F-140 has none, so that force goes through
            # the loop into the wrist and KICKS the arm (-> rapid jerky arm motion).
            # 400 is the compromise: 40x the original grip (holds the cube) but a
            # much gentler reaction. Raise toward 1650 only if it slips; drop toward
            # 200 if the arm still gets kicked.
            effort_limit_sim=400.0,
            velocity_limit_sim=10.0,
            stiffness=17.0,
            damping=0.5,
            friction=0.0,
            armature=0.01,
        ),
        # Inner fingers: soft springs that keep the pads parallel as it closes.
        "gripper_finger": ImplicitActuatorCfg(
            joint_names_expr=[".*_inner_finger_joint"],
            stiffness=0.2,
            damping=0.1,
            friction=0.0,
            armature=0.01,
        ),
        # Everything else in the articulation linkage. UNLIKE the 2F-85, the 2F-140
        # has NO mimic constraint holding this linkage -- only a PhysX loop closure.
        # With zero damping/armature those passive joints ring up on contact (the
        # loop pumps in energy with nothing to dissipate it) -> velocities blow up
        # -> the whole articulation NaNs and vanishes from the scene. The 2F-85 gets
        # this damping "for free" from its mimic; the 2F-140 must set it explicitly.
        "gripper_passive": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_outer_knuckle_joint",
                ".*_outer_finger_joint",
                ".*_inner_finger_pad_joint",
            ],
            stiffness=0.0,
            damping=0.1,
            friction=0.0,
            armature=0.01,
        ),
    },
)