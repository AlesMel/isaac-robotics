# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaac_robots.tasks.direct._shared.assets import UR3E_ROBOTIQ_HANDE_CFG
from isaac_robots.tasks.manager_based.lift import mdp
from isaac_robots.tasks.manager_based.lift.lift_env_cfg import LiftEnvCfg

##
# Curriculum tuning.
#
# Upstream's Franka-calibrated lift curriculum ramps ``action_rate`` and
# ``joint_vel`` penalties from -1e-4 to -1e-1 (1000x) at env step 10_000.
# Earlier we tried a 100x ramp at step 100_000; the post-ramp shock still
# collapsed the policy in TB run 2026-06-03_15-54-25 (lifting_object 13.9 -> 1.7
# right after the curriculum fired at step 100k), and KL-adaptive LR decayed
# to ~0 trying to recover. We disable the ramp entirely - the baseline -1e-4
# weights are already enough regularization for UR3e + Hand-E.
##
UR3E_HANDE_CURRICULUM_NUM_STEPS = 100_000
# Bumped from upstream baseline -1e-4 to -1e-3 (constant, NOT ramped via
# curriculum). Rationale: the -1e-4 baseline produced action_rate episode
# values ~-0.0001 (TB run 22-24) - negligible against +250 reward, so the
# policy had no real incentive to be smooth. -1e-3 constant gives ~-0.001
# episode value which is still small relative to total reward but visible
# enough to discourage limit-cycle oscillation. We keep the value constant
# (curriculum is a no-op) because every previous attempt to *ramp* the
# weight mid-training collapsed the policy.
UR3E_HANDE_CURRICULUM_ACTION_RATE_WEIGHT = -1e-3
UR3E_HANDE_CURRICULUM_JOINT_VEL_WEIGHT = -1e-3

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


# Arm joint regex pattern (matches the UR ROS-Industrial joint names).
UR3E_ARM_JOINT_NAMES = ["shoulder_.*", "elbow_joint", "wrist_.*"]

# Robotiq Hand-E slider joints. USD limits are [-0.02, 0.0]: 0.0 = open
# (spawn pose), -0.02 = fully closed.
UR3E_HANDE_GRIPPER_JOINT_NAMES = ["Slider_.*"]
UR3E_HANDE_OPEN_COMMAND = {"Slider_.*": 0.0}
UR3E_HANDE_CLOSE_COMMAND = {"Slider_.*": -0.02}

# End-effector frame. ``tool0`` is the UR3e wrist-3 flange; the Hand-E grasp
# point sits ~119 mm out along tool0's local +Z (between the closed blade pads).
UR3E_HANDE_EE_BODY_NAME = "tool0"
UR3E_HANDE_EE_OFFSET = (0.0, 0.0, 0.119)


@configclass
class UR3eHandECubeLiftEnvCfg(LiftEnvCfg):
    """UR3e + Robotiq Hand-E lift-cube task with joint-position arm control."""

    def __post_init__(self):
        super().__post_init__()

        # Longer episode: relative joint control needs more steps to traverse
        # the workspace because per-step delta is hard-capped (~0.05 rad).
        # Upstream sets 5.0 s; 8.0 s gives ~400 control steps at 50 Hz,
        # comfortably enough for reach + grasp + lift + hold.
        self.episode_length_s = 8.0

        # Bump PhysX GPU buffers to handle large num_envs.
        # IsaacLab/upstream defaults are tuned for ~4096 envs; at 8192+ they
        # overflow with one of two errors:
        #   1) "Patch buffer overflow" -> bump gpu_max_rigid_patch_count
        #   2) "totalAggregatePairsCapacity ... otherwise simulation will miss
        #      interactions" -> bump gpu_total_aggregate_pairs_capacity
        # 4x defaults gives comfortable headroom up to ~16-32k envs.
        self.sim.physx.gpu_max_rigid_patch_count = 5 * 2**17  # default 5 * 2**15 = 163_840 -> 655_360
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024  # default 16 * 1024 = 16_384 -> 65_536

        # Set UR3e + Hand-E as a single articulation.
        # Disable contact-sensor reporting on the USD - we don't use a
        # ContactSensor in this task, so PhysX can skip the contact accounting.
        self.scene.robot = UR3E_ROBOTIQ_HANDE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.activate_contact_sensors = False

        # Arm action: 6-DoF *relative* joint position control.
        #
        # Why relative instead of absolute (upstream Franka uses absolute):
        #   - Absolute JointPositionAction with use_default_offset=True maps
        #     action -> target = default + scale * action. That couples per-
        #     step velocity to workspace coverage: low scale gives stable
        #     hold-pose noise but forces huge action magnitudes for reach,
        #     which then snap violently during pose transitions (e.g. tearing
        #     the cube out of the gripper when lifting).
        #   - RelativeJointPositionAction maps action -> target = current +
        #     scale * action. scale is a true per-step delta cap. Workspace
        #     is unbounded (policy accumulates deltas). Arm can't snap.
        #
        # scale=0.05 rad = ~2.9 deg max per joint per 20 ms step => ~143 deg/s
        # cap, which is plenty for manipulation but slow enough to keep grasp
        # stable.
        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=UR3E_ARM_JOINT_NAMES,
            scale=0.05,
            use_zero_offset=True,
        )

        # Gripper action: binary open/close on the Hand-E sliders.
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=UR3E_HANDE_GRIPPER_JOINT_NAMES,
            open_command_expr=UR3E_HANDE_OPEN_COMMAND,
            close_command_expr=UR3E_HANDE_CLOSE_COMMAND,
        )

        # Object pose command targets the EE body.
        self.commands.object_pose.body_name = UR3E_HANDE_EE_BODY_NAME
        # Tighten the goal workspace for UR3e's shorter reach (~0.5 m).
        self.commands.object_pose.ranges.pos_x = (0.3, 0.45)
        self.commands.object_pose.ranges.pos_y = (-0.2, 0.2)
        self.commands.object_pose.ranges.pos_z = (0.15, 0.35)

        # Target object: instanceable DexCube from the Isaac Nucleus.
        # IMPORTANT: cube scale must match Hand-E stroke.
        #   - DexCube native size:    ~80 mm
        #   - Hand-E gripper stroke:  ~40 mm total (Sliders are [-0.02, 0.0] each side)
        #   - Franka uses 0.8 scale (64 mm) because Franka has 80 mm stroke;
        #     for Hand-E that's too big - the cube can't fit between the pads
        #     and ends up jammed against the gripper body (the direct env has
        #     the same fix at scale=0.5). Spawn z drops too because the cube
        #     half-extent shrinks: 0.4 * 0.05 = 0.02 m -> rest center 0.02.
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, 0.036], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.5, 0.5, 0.5),  # 0.8 -> 0.5 (~40 mm cube, fits Hand-E pads)
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Bump base penalty weights from upstream -1e-4 to -1e-3 (constant,
        # no curriculum). Curriculum target = base weight so the curriculum
        # is effectively a no-op (we set both to the same value to stay
        # consistent with the existing curriculum infrastructure).
        self.rewards.action_rate.weight = UR3E_HANDE_CURRICULUM_ACTION_RATE_WEIGHT
        self.rewards.joint_vel.weight = UR3E_HANDE_CURRICULUM_JOINT_VEL_WEIGHT
        self.curriculum.action_rate.params["num_steps"] = UR3E_HANDE_CURRICULUM_NUM_STEPS
        self.curriculum.action_rate.params["weight"] = UR3E_HANDE_CURRICULUM_ACTION_RATE_WEIGHT
        self.curriculum.joint_vel.params["num_steps"] = UR3E_HANDE_CURRICULUM_NUM_STEPS
        self.curriculum.joint_vel.params["weight"] = UR3E_HANDE_CURRICULUM_JOINT_VEL_WEIGHT

        # Velocity-gated tanh fine-grained tracking (replaces upstream tanh
        # and the failed Gaussian experiment in run 2026-06-07_23-03).
        #
        # Problem: upstream tanh has nonzero gradient at d=0 (-1/std = -20
        # here), causing limit-cycle oscillation when holding cube at goal.
        # Gaussian kernel removed the gradient but caused late-stage drift
        # (no centering pull).
        #
        # Solution: keep tanh (centering pull preserved) but multiply by a
        # joint-velocity gate inside the goal neighborhood. Outside the
        # neighborhood, behavior is identical to upstream tanh - approach
        # gradient unchanged. Inside, the reward is only earned when the
        # arm is still. The policy thus has a direct economic incentive to
        # hold steady at the goal, not just be near it.
        #
        # Filtering robot_cfg to arm joints only excludes the Hand-E sliders
        # (whose closing motion would otherwise spuriously trigger the gate
        # at the wrong moments).
        self.rewards.object_goal_tracking_fine_grained.func = mdp.object_goal_distance_velocity_gated
        self.rewards.object_goal_tracking_fine_grained.params = {
            "std": 0.05,
            "minimal_height": 0.04,
            "command_name": "object_pose",
            "velocity_thresh": 0.5,   # rad/s magnitude across arm joints
            "neighborhood": 0.10,     # m - gate fires only when cube within 10 cm of goal
            "robot_cfg": SceneEntityCfg("robot", joint_names=UR3E_ARM_JOINT_NAMES),
        }

        # EE frame transformer: base_link -> tool0 + grasp offset.
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/{UR3E_HANDE_EE_BODY_NAME}",
                    name="end_effector",
                    offset=OffsetCfg(pos=UR3E_HANDE_EE_OFFSET),
                ),
            ],
        )


@configclass
class UR3eHandECubeLiftEnvCfg_PLAY(UR3eHandECubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Smaller scene + no observation corruption for interactive play.
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        # Show ONLY the EE FrameTransformer marker so we can visually verify
        # the grasp-point offset (tool0 + (0, 0, 0.119)) actually sits between
        # the Hand-E pads. Goal-pose command marker is disabled to avoid
        # visual clutter (upstream defaults it to True).
        self.scene.ee_frame.debug_vis = False
        self.commands.object_pose.debug_vis = False
