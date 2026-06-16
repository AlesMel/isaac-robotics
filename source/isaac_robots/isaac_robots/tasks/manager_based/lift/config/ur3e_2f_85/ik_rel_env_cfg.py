from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass
 
from . import joint_pos_env_cfg
 
@configclass
class UR3e2F85CubeLiftEnvCfg_IK_Rel(joint_pos_env_cfg.UR3e2F85CubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
 
        # Replace the joint-space arm action with relative-pose IK.
        # Arm joints ONLY — never include finger_joint in the IK chain.
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            body_name="wrist_3_link",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
            ),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                # Match the ee_frame TCP offset from joint_pos_env_cfg.py.
                pos=(0.0, 0.0, 0.14),
                # Aligns the TCP approach axis to the UR flange convention.
                rot=(0.0, -0.7071, 0.0, -0.7071),
            ),
        )
 