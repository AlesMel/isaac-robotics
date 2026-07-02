"""Lift-cube environment (joint-position control) for UR3e + Robotiq 2F-140.

Duplicated from the working ur3e_2f_85 config. The arm and the task (rewards,
observations, goal box, cube) are identical -- only the gripper differs:
  * robot asset      = UR3e_ROBOTIQ_2F140_CFG
  * gripper close    = 0.785 rad (the 2F-140 finger_joint range is 0..45 deg,
                       vs the 2F-85's 0..47 deg = 0.82 rad)
  * end-effector TCP = larger offset (the 2F-140 is longer than the 2F-85)
"""

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaac_robots.tasks.direct._shared.assets import UR3e_ROBOTIQ_2F140_CFG
import math

from .lift_ur3e_2f140_env_cfg import LiftEnvCfg


@configclass
class UR3e2F140CubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ---------------- robot ----------------
        self.scene.robot = UR3e_ROBOTIQ_2F140_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ---------------- actions ----------------
        # Arm: list the 6 UR3e joints EXPLICITLY (never ".*_joint" -- it would also
        # grab the gripper joints and collide with the gripper action term).
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            scale=0.5,
            use_default_offset=True,
        )

        # Gripper: binary open/close on the single driven joint. The 2F-140
        # finger_joint range is 0..0.785 rad (0..45 deg): 0.0 == open,
        # 0.785 == full close (contact on the cube stops it earlier).
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["finger_joint"],
            open_command_expr={"finger_joint": 0.0},
            close_command_expr={"finger_joint": 0.785},
        )

        # Pose-command target body (same wrist as the 2F-85).
        self.commands.object_pose.body_name = "wrist_3_link"
        self.commands.object_pose.ranges.pitch = (math.pi / 2, math.pi / 2)

        # ---------------- object to lift (same cube as 2F-85 / Franka) ----------------
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
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

        # ---------------- end-effector frame ----------------
        # TCP offset from wrist_3_link to the grasp point (between the jaws).
        # The 2F-140 is LONGER than the 2F-85 (which used 0.14 m). 0.19 is an
        # estimate -- VERIFY visually in play mode (the marker should sit between
        # the 2F-140 fingertips at the default pose) and adjust if it's off.
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ur3e/base_link",
            debug_vis=False,  # show the end_effector marker; set offset so it sits between the jaws
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ur3e/wrist_3_link",
                    name="end_effector",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.19)),
                ),
            ],
        )


@configclass
class UR3e2F140CubeLiftEnvCfg_PLAY(UR3e2F140CubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # smaller, deterministic setup for visualization / eval
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
