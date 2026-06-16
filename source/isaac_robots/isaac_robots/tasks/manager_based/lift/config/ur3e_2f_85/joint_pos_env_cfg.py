"""Lift-cube environment (joint-position control) for UR3e + Robotiq 2F-85.

This is the primary variant for RL training. The relative-IK variant
(ik_rel_env_cfg.py) is better for teleop / data collection.
"""

from isaaclab.assets import RigidObjectCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.config.ur3e_2f_85.lift_ur3e_2f85_env_cfg import LiftEnvCfg
from isaac_robots.tasks.direct._shared.assets import UR3e_ROBOTIQ_2F85_CFG
from isaaclab.sensors import ContactSensorCfg
from . import mdp_grasp_rewards as gmdp
from isaaclab.managers import RewardTermCfg as RewTerm

@configclass
class UR3e2F85CubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ---------------- robot ----------------
        self.scene.robot = UR3e_ROBOTIQ_2F85_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ---------------- actions ----------------
        # Arm: list joints EXPLICITLY. Never ".*_joint" — it also grabs finger_joint
        # and collides with the gripper action term.
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

        # Gripper: binary open/close on the single driven joint.
        # VERIFY the sign on your asset (see ur3e_robotiq_2f85.py header).
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["finger_joint"],
            open_command_expr={ "finger_joint": 0.0},   # 0.0 == open (standard convention)
            close_command_expr={"finger_joint": 0.82},  # ~full close for 2F-85
        )

        # ---------------- end-effector frame ----------------
        # The lift reward uses object<->ee distance, so the TCP offset matters.
        # MEASURE the offset from wrist_3_link to the fingertip in your assembled
        # USD (GUI: read both body world positions). ~0.15 m is a 2F-85 starting guess.
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ur3e/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ur3e/wrist_3_link",
                    name="end_effector",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.14)),  # verified visually at default pose: marker between the 2F-85 jaws
                ),
            ],
        )

        # Pose-command target body (there is no panda_hand here).
        self.commands.object_pose.body_name = "wrist_3_link"

        # ---------------- object to lift ----------------
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

        # ---------------- workspace tuning ----------------
        # The UR3e reach (~0.5 m) is much smaller than the Franka's. If the lift
        # targets fall outside reach the policy can't learn — tighten the ranges.
        self.commands.object_pose.ranges.pos_x = (0.35, 0.50)
        self.commands.object_pose.ranges.pos_y = (-0.20, 0.20)
        self.commands.object_pose.ranges.pos_z = (0.15, 0.35)

        self.scene.contact_left_finger = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Robotiq_2F_85_edit/Robotiq_2F_85/left_inner_finger",
            update_period=0.0,
            debug_vis=False,
        )
        self.scene.contact_right_finger = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Robotiq_2F_85_edit/Robotiq_2F_85/right_inner_finger",
            update_period=0.0,
            debug_vis=False,
        )
        
        self.rewards.grasping = RewTerm(
            func=gmdp.lift_with_grasp_gate,
            weight=1.0,
            params={
                "object_cfg": "object",
                "ee_frame_cfg": "ee_frame", # Crucial for the dense reaching term!
                "left_sensor_name": "contact_left_finger",
                "right_sensor_name": "contact_right_finger",
                "force_threshold": 0.5,
            },
        )

@configclass
class UR3e2F85CubeLiftEnvCfg_PLAY(UR3e2F85CubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # smaller, deterministic setup for visualization / eval
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False