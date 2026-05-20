# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from pathlib import Path

import numpy as np
import torch

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObjectCfg, SurfaceGripper, SurfaceGripperCfg
from isaaclab.envs.mdp.actions.actions_cfg import SurfaceGripperBinaryActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaacsim.core.utils.extensions import enable_extension

from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
from isaaclab_tasks.manager_based.manipulation.stack.stack_env_cfg import StackEnvCfg

from isaac_robots.tasks.manager_based.stack import mdp as ur3e_stack_mdp
from isaac_robots.tasks.direct._shared.assets import UR3E_CFG, UR3E_ROBOTIQ_2F85_CFG  # isort: skip

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


UR3E_STACK_DEFAULT_JOINT_POSE = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.5707,
    "elbow_joint": 1.5707,
    "wrist_1_joint": -1.5707,
    "wrist_2_joint": 1.5707,
    "wrist_3_joint": 0.0,
}

UR3E_LONG_SUCTION_CFG = UR3E_CFG.copy()
UR3E_LONG_SUCTION_CFG.spawn.rigid_props.disable_gravity = True
UR3E_LONG_SUCTION_CFG.init_state.joint_pos = UR3E_STACK_DEFAULT_JOINT_POSE.copy()

UR3E_SHORT_SUCTION_CFG = UR3E_LONG_SUCTION_CFG.copy()

UR3E_SUCTION_VISUALIZE = os.getenv("UR3E_SUCTION_VISUALIZE", "0").lower() in ("1", "true", "yes", "on")
UR3E_ARM_JOINT_NAMES = ["shoulder_.*", "elbow_joint", "wrist_.*"]
UR3E_ROBOTIQ_CLOSE_POS = 0.69
UR3E_ROBOTIQ_GRIPPER_JOINT_NAMES = [
    "finger_joint",
    "right_outer_knuckle_joint",
    "left_inner_finger_joint",
    "right_inner_finger_joint",
    "left_inner_finger_knuckle_joint",
    "right_inner_finger_knuckle_joint",
]
UR3E_ROBOTIQ_OPEN_COMMAND = {
    "finger_joint": 0.0,
    "right_outer_knuckle_joint": 0.0,
    "left_inner_finger_joint": 0.0,
    "right_inner_finger_joint": 0.0,
    "left_inner_finger_knuckle_joint": 0.0,
    "right_inner_finger_knuckle_joint": 0.0,
}
UR3E_ROBOTIQ_CLOSE_COMMAND = {
    "finger_joint": UR3E_ROBOTIQ_CLOSE_POS,
    "right_outer_knuckle_joint": UR3E_ROBOTIQ_CLOSE_POS,
    "left_inner_finger_joint": -UR3E_ROBOTIQ_CLOSE_POS,
    "right_inner_finger_joint": UR3E_ROBOTIQ_CLOSE_POS,
    "left_inner_finger_knuckle_joint": -UR3E_ROBOTIQ_CLOSE_POS,
    "right_inner_finger_knuckle_joint": -UR3E_ROBOTIQ_CLOSE_POS,
}
UR3E_ROBOTIQ_EE_BODY_NAME = os.getenv("UR3E_ROBOTIQ_EE_BODY_NAME", "tool0")
# The assembled Robotiq 2F-85 points along tool0 local +Z. Keep the TCP between the inner fingers.
UR3E_ROBOTIQ_EE_OFFSET = tuple(
    float(value.strip()) for value in os.getenv("UR3E_ROBOTIQ_EE_OFFSET", "0.0,0.0,0.12").split(",")
)

UR3E_ROBOTIQ_2F85_STACK_CFG = UR3E_ROBOTIQ_2F85_CFG.copy()
UR3E_ROBOTIQ_2F85_STACK_CFG.init_state.joint_pos.update(UR3E_STACK_DEFAULT_JOINT_POSE)


def require_ur3e_robotiq_asset() -> None:
    """Fail early with a useful message if the combined Robotiq USD is missing."""
    usd_path = UR3E_ROBOTIQ_2F85_STACK_CFG.spawn.usd_path
    usd_name = usd_path.replace("\\", "/").rsplit("/", 1)[-1].lower()
    if usd_name in {"robotiq_2f_85.usd", "robotiq_2f_85_edit.usd"}:
        raise ValueError(
            "UR3E_ROBOTIQ_2F85_USD_PATH points at the standalone Robotiq gripper USD. "
            "That asset has its own articulation root and cannot be nested under /Robot. "
            "Use a single combined UR3e+Robotiq articulation USD instead."
        )
    if "://" in usd_path:
        return
    if not Path(usd_path).is_file():
        raise FileNotFoundError(
            "UR3e Robotiq 2F-85 tasks need a combined UR3e+Robotiq USD articulation. "
            f"Expected one at '{usd_path}'. Build it with "
            "'python scripts/ur3e/build_ur3e_robotiq_2f85_usd.py' or set "
            "UR3E_ROBOTIQ_2F85_USD_PATH to your combined USD."
        )


def fixed_xr_anchor_rotation(headpose, primpose):
    """Return the default fixed XR anchor rotation as an importable callable."""
    return np.array([1, 0, 0, 0], dtype=np.float64)


def randomize_arm_joint_by_gaussian_offset(
    env,
    env_ids: torch.Tensor,
    mean: float,
    std: float,
    arm_joint_names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize only arm joints, leaving Robotiq finger defaults untouched."""
    asset = env.scene[asset_cfg.name]
    arm_joint_ids, _ = asset.find_joints(arm_joint_names)

    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    joint_pos[:, arm_joint_ids] += math_utils.sample_gaussian(
        mean, std, (len(env_ids), len(arm_joint_ids)), joint_pos.device
    )

    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids][:, arm_joint_ids]
    joint_pos[:, arm_joint_ids] = joint_pos[:, arm_joint_ids].clamp(
        joint_pos_limits[..., 0], joint_pos_limits[..., 1]
    )

    asset.set_joint_position_target(joint_pos, env_ids=env_ids)
    asset.set_joint_velocity_target(joint_vel, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


class AutoSpawnSurfaceGripper(SurfaceGripper):
    """Surface gripper that authors missing gripper prims on converted UR3e USDs."""

    def __init__(self, cfg: SurfaceGripperCfg):
        self._spawn_missing_surface_grippers(cfg)
        super().__init__(cfg)

    @staticmethod
    def _spawn_missing_surface_grippers(cfg: SurfaceGripperCfg) -> None:
        enable_extension("isaacsim.robot.surface_gripper")
        from usd.schema.isaac import robot_schema

        parent_expr, gripper_name = cfg.prim_path.rsplit("/", 1)
        parent_prims = sim_utils.find_matching_prims(parent_expr)
        if not parent_prims:
            return

        stage = parent_prims[0].GetStage()
        for parent_prim in parent_prims:
            parent_path = parent_prim.GetPath().pathString
            gripper_path = f"{parent_path}/{gripper_name}"
            visual_path = f"{parent_path}/SuctionCupVisual"
            gripper_prim = stage.GetPrimAtPath(gripper_path)
            gripper_exists = gripper_prim.IsValid() and AutoSpawnSurfaceGripper._has_typed_gripper_attrs(
                gripper_prim, robot_schema
            )
            visual_exists = stage.GetPrimAtPath(visual_path).IsValid()
            if gripper_exists and visual_exists:
                continue

            robot_path = parent_path.rsplit("/", 1)[0]
            sim_utils.make_uninstanceable(robot_path, stage=stage)
            sim_utils.make_uninstanceable(parent_path, stage=stage)
            if not gripper_exists:
                if gripper_prim.IsValid():
                    stage.RemovePrim(gripper_path)
                gripper_prim = robot_schema.CreateSurfaceGripper(stage, gripper_path)
                # IsaacSurfaceGripper is a typed schema prim, not an Xformable.
                # The ee_frame offset carries the local TCP offset.
                AutoSpawnSurfaceGripper._set_gripper_attr(
                    gripper_prim, "isaac:maxGripDistance", cfg.max_grip_distance
                )
                AutoSpawnSurfaceGripper._set_gripper_attr(
                    gripper_prim, "isaac:coaxialForceLimit", cfg.coaxial_force_limit
                )
                AutoSpawnSurfaceGripper._set_gripper_attr(
                    gripper_prim, "isaac:shearForceLimit", cfg.shear_force_limit
                )
                AutoSpawnSurfaceGripper._set_gripper_attr(gripper_prim, "isaac:retryInterval", cfg.retry_interval)
            AutoSpawnSurfaceGripper._spawn_visual_suction_cup(stage, visual_path, cfg)

    @staticmethod
    def _spawn_visual_suction_cup(stage, visual_path: str, cfg: SurfaceGripperCfg) -> None:
        if not getattr(cfg, "visualize", True):
            return

        from pxr import Gf, UsdGeom

        local_pos = tuple(float(value) for value in getattr(cfg, "local_pos", (0.0, 0.0, 0.0)))
        tip_distance = abs(local_pos[0])
        if tip_distance <= 1.0e-6:
            return

        direction = 1.0 if local_pos[0] >= 0.0 else -1.0
        cup_length = min(float(getattr(cfg, "visual_cup_length", 0.035)), max(tip_distance * 0.45, 0.01))
        seal_length = min(float(getattr(cfg, "visual_seal_length", 0.006)), cup_length * 0.5)
        stem_length = max(tip_distance - cup_length, 0.01)

        if stage.GetPrimAtPath(visual_path).IsValid():
            stage.RemovePrim(visual_path)
        visual_root = stage.DefinePrim(visual_path, "Xform")
        sim_utils.standardize_xform_ops(visual_root)

        def create_colored_cylinder(
            name: str, radius: float, height: float, x_center: float, color: tuple[float, ...]
        ):
            prim = stage.DefinePrim(f"{visual_path}/{name}", "Cylinder")
            cylinder = UsdGeom.Cylinder(prim)
            cylinder.CreateRadiusAttr(float(radius))
            cylinder.CreateHeightAttr(float(height))
            cylinder.CreateAxisAttr("X")
            cylinder.CreateDisplayColorAttr()
            cylinder.GetDisplayColorAttr().Set(
                [Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))]
            )
            sim_utils.standardize_xform_ops(prim, translation=(x_center, local_pos[1], local_pos[2]))

        create_colored_cylinder(
            "Stem",
            getattr(cfg, "visual_stem_radius", 0.012),
            stem_length,
            direction * stem_length * 0.5,
            getattr(cfg, "visual_stem_color", (0.35, 0.35, 0.38)),
        )
        create_colored_cylinder(
            "Cup",
            getattr(cfg, "visual_cup_radius", 0.035),
            cup_length,
            direction * (stem_length + cup_length * 0.5),
            getattr(cfg, "visual_cup_color", (0.08, 0.08, 0.09)),
        )
        create_colored_cylinder(
            "Seal",
            getattr(cfg, "visual_cup_radius", 0.035) * 1.08,
            seal_length,
            direction * (tip_distance - seal_length * 0.5),
            getattr(cfg, "visual_seal_color", (0.02, 0.02, 0.02)),
        )

    @staticmethod
    def _set_gripper_attr(prim, attr_name: str, value: float | None) -> None:
        if value is None:
            return
        attr = prim.GetAttribute(attr_name)
        if attr.IsValid():
            attr.Set(value)

    @staticmethod
    def _has_typed_gripper_attrs(prim, robot_schema) -> bool:
        required_attrs = (
            robot_schema.Attributes.MAX_GRIP_DISTANCE.name,
            robot_schema.Attributes.COAXIAL_FORCE_LIMIT.name,
            robot_schema.Attributes.SHEAR_FORCE_LIMIT.name,
            robot_schema.Attributes.RETRY_INTERVAL.name,
        )
        return all(prim.GetAttribute(attr_name).GetTypeName() for attr_name in required_attrs)


@configclass
class AutoSpawnSurfaceGripperCfg(SurfaceGripperCfg):
    """Surface gripper config that can create the missing USD prim."""

    local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    local_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    # The fallback visual is a debug-only approximation, not official UR/Robotiq geometry.
    visualize: bool = UR3E_SUCTION_VISUALIZE
    visual_stem_radius: float = 0.012
    visual_cup_radius: float = 0.035
    visual_cup_length: float = 0.035
    visual_seal_length: float = 0.006
    visual_stem_color: tuple[float, float, float] = (0.35, 0.35, 0.38)
    visual_cup_color: tuple[float, float, float] = (0.08, 0.08, 0.09)
    visual_seal_color: tuple[float, float, float] = (0.02, 0.02, 0.02)
    class_type: type = AutoSpawnSurfaceGripper


@configclass
class EventCfgUR3eSuction:
    """Configuration for events."""

    init_ur3e_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [0.0, -1.5707, 1.5707, -1.5707, -1.5707, 0.0],
        },
    )

    randomize_ur3e_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    randomize_cube_positions = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.4, 0.6), "y": (-0.10, 0.10), "z": (0.0203, 0.0203), "yaw": (-1.0, 1.0, 0)},
            "min_separation": 0.1,
            "asset_cfgs": [SceneEntityCfg("cube_1"), SceneEntityCfg("cube_2"), SceneEntityCfg("cube_3")],
        },
    )


@configclass
class EventCfgUR3eRobotiq:
    """Configuration for UR3e stack events with a real Robotiq gripper."""

    randomize_ur3e_arm_joint_state = EventTerm(
        func=randomize_arm_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "arm_joint_names": UR3E_ARM_JOINT_NAMES,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    randomize_cube_positions = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.4, 0.6), "y": (-0.10, 0.10), "z": (0.0203, 0.0203), "yaw": (-1.0, 1.0, 0)},
            "min_separation": 0.1,
            "asset_cfgs": [SceneEntityCfg("cube_1"), SceneEntityCfg("cube_2"), SceneEntityCfg("cube_3")],
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for PPO training on the UR3e stack task."""

    reach_cube_2 = RewTerm(
        func=ur3e_stack_mdp.reaching_object,
        weight=2.0,
        params={"object_cfg": SceneEntityCfg("cube_2"), "std": 0.12},
    )
    lift_cube_2 = RewTerm(
        func=ur3e_stack_mdp.lifting_object,
        weight=4.0,
        params={"object_cfg": SceneEntityCfg("cube_2"), "table_height": 0.0203, "lift_height": 0.08},
    )
    align_cube_2_on_1 = RewTerm(
        func=ur3e_stack_mdp.aligning_object_over_object,
        weight=4.0,
        params={"upper_object_cfg": SceneEntityCfg("cube_2"), "lower_object_cfg": SceneEntityCfg("cube_1")},
    )
    stack_cube_2_on_1 = RewTerm(
        func=ur3e_stack_mdp.object_stacked_without_gripper_check,
        weight=10.0,
        params={"upper_object_cfg": SceneEntityCfg("cube_2"), "lower_object_cfg": SceneEntityCfg("cube_1")},
    )

    reach_cube_3 = RewTerm(
        func=ur3e_stack_mdp.reaching_object,
        weight=1.0,
        params={"object_cfg": SceneEntityCfg("cube_3"), "std": 0.14},
    )
    lift_cube_3 = RewTerm(
        func=ur3e_stack_mdp.lifting_object,
        weight=3.0,
        params={"object_cfg": SceneEntityCfg("cube_3"), "table_height": 0.0203, "lift_height": 0.10},
    )
    align_cube_3_on_2 = RewTerm(
        func=ur3e_stack_mdp.aligning_object_over_object,
        weight=5.0,
        params={"upper_object_cfg": SceneEntityCfg("cube_3"), "lower_object_cfg": SceneEntityCfg("cube_2")},
    )
    stack_cube_3_on_2 = RewTerm(
        func=ur3e_stack_mdp.object_stacked_without_gripper_check,
        weight=15.0,
        params={"upper_object_cfg": SceneEntityCfg("cube_3"), "lower_object_cfg": SceneEntityCfg("cube_2")},
    )

    full_stack = RewTerm(func=ur3e_stack_mdp.three_cube_stack_success, weight=30.0)
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.005)
    joint_vel = RewTerm(
        func=base_mdp.joint_vel_l2,
        weight=-0.0005,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class UR3eCubeStackEnvCfg(StackEnvCfg):
    cube_position_obs_noise_std: float = float(os.getenv("UR3E_CUBE_POSITION_OBS_NOISE_STD", "0.0"))
    cube_position_obs_dropout_prob: float = float(os.getenv("UR3E_CUBE_POSITION_OBS_DROPOUT_PROB", "0.0"))

    # Rigid body properties of each cube
    cube_properties = RigidBodyPropertiesCfg(
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=1,
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=5.0,
        disable_gravity=False,
    )
    cube_scale = (1.0, 1.0, 1.0)
    # Listens to the required transforms
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set events
        self.events = EventCfgUR3eSuction()
        self.rewards = RewardsCfg()
        self.terminations.success = DoneTerm(func=ur3e_stack_mdp.three_cube_stack_success)
        self.observations.policy.concatenate_terms = True
        self.observations.policy.object = ObsTerm(
            func=ur3e_stack_mdp.cube_positions_from_pose_provider,
            params={
                "noise_std": self.cube_position_obs_noise_std,
                "dropout_prob": self.cube_position_obs_dropout_prob,
            },
        )
        self.observations.policy.cube_positions = None
        self.observations.policy.cube_orientations = None
        if self.xr is not None:
            self.xr.anchor_rotation_custom_func = fixed_xr_anchor_rotation

        # Set actions for the specific robot type (UR3e)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=UR3E_ARM_JOINT_NAMES, scale=0.5, use_default_offset=True
        )
        # Set surface gripper action
        self.actions.gripper_action = SurfaceGripperBinaryActionCfg(
            asset_name="surface_gripper",
            open_command=-1.0,
            close_command=1.0,
        )

        # Set each stacking cube deterministically
        self.scene.cube_1 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_1",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, 0.0203], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
                scale=self.cube_scale,
                rigid_props=self.cube_properties,
            ),
        )
        self.scene.cube_2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_2",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.55, 0.05, 0.0203], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
                scale=self.cube_scale,
                rigid_props=self.cube_properties,
            ),
        )
        self.scene.cube_3 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_3",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.60, -0.1, 0.0203], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd",
                scale=self.cube_scale,
                rigid_props=self.cube_properties,
            ),
        )

        self.decimation = 5
        self.episode_length_s = 30.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = 5


@configclass
class UR3eRobotiq2F85CubeStackEnvCfg(UR3eCubeStackEnvCfg):
    """Configuration for the UR3e stack task with a real Robotiq 2F-85 gripper."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        require_ur3e_robotiq_asset()

        # Set events
        self.events = EventCfgUR3eRobotiq()

        # Set UR3e + Robotiq as a single robot articulation
        self.scene.robot = UR3E_ROBOTIQ_2F85_STACK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (UR3e + Robotiq 2F-85)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=UR3E_ARM_JOINT_NAMES, scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=UR3E_ROBOTIQ_GRIPPER_JOINT_NAMES,
            open_command_expr=UR3E_ROBOTIQ_OPEN_COMMAND,
            close_command_expr=UR3E_ROBOTIQ_CLOSE_COMMAND,
        )

        # Utilities for the upstream stack observations/terminations, which expect a two-finger gripper.
        self.gripper_joint_names = ["finger_joint", "right_outer_knuckle_joint"]
        self.gripper_open_val = 0.0
        self.gripper_threshold = 0.02

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=self.marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/{UR3E_ROBOTIQ_EE_BODY_NAME}",
                    name="end_effector",
                    offset=OffsetCfg(pos=UR3E_ROBOTIQ_EE_OFFSET),
                ),
            ],
        )


@configclass
class UR3eLongSuctionCubeStackEnvCfg(UR3eCubeStackEnvCfg):
    """Configuration for the UR3e Long Suction Cube Stack Environment."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Suction grippers currently require CPU simulation
        self.device = "cpu"
        self.sim.device = "cpu"

        # Set events
        self.events = EventCfgUR3eSuction()

        # Set UR3e as robot
        self.scene.robot = UR3E_LONG_SUCTION_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set surface gripper: Ensure the SurfaceGripper prim has the required attributes
        self.scene.surface_gripper = AutoSpawnSurfaceGripperCfg(
            prim_path="{ENV_REGEX_NS}/Robot/tool0/SurfaceGripper",
            local_pos=(0.22, 0.0, 0.0),
            max_grip_distance=0.0075,
            shear_force_limit=5000.0,
            coaxial_force_limit=5000.0,
            retry_interval=0.05,
        )

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=self.marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/tool0",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.22, 0.0, 0.0],
                    ),
                ),
            ],
        )


@configclass
class UR3eShortSuctionCubeStackEnvCfg(UR3eCubeStackEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Suction grippers currently require CPU simulation
        self.device = "cpu"
        self.sim.device = "cpu"

        # Set UR3e as robot
        self.scene.robot = UR3E_SHORT_SUCTION_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set surface gripper: Ensure the SurfaceGripper prim has the required attributes
        self.scene.surface_gripper = AutoSpawnSurfaceGripperCfg(
            prim_path="{ENV_REGEX_NS}/Robot/tool0/SurfaceGripper",
            local_pos=(0.1585, 0.0, 0.0),
            max_grip_distance=0.0075,
            shear_force_limit=5000.0,
            coaxial_force_limit=5000.0,
            retry_interval=0.05,
        )

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=self.marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/tool0",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.1585, 0.0, 0.0],
                    ),
                ),
            ],
        )
