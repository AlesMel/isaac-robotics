"""Configuration for the UR3e lift-cube direct task.

Hand-E-only. The parent ``UR3eLiftCubeEnvCfg`` defaults are tuned for the
Hand-E gripper; the suction variant (kept registered for backward compat) is
no longer guaranteed to train without the gripper-agnostic plumbing that
previously lived in the env.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .._shared.grippers import GripperCfg, RobotiqHandEGripperCfg, SuctionGripperCfg
from .cfg import UR3E_CFG, UR3E_ROBOTIQ_HANDE_CFG


@configclass
class UR3eLiftCubeEnvCfg(DirectRLEnvCfg):
    # --- episode / control rate ---
    episode_length_s: float = 5.0
    decimation: int = 2
    debug_vis: bool = True

    action_space: int = 7
    observation_space: int = 26
    state_space: int = 0

    # --- scene / sim ---
    viewer: ViewerCfg = ViewerCfg(
        eye=(1.4, 1.2, 0.9),
        lookat=(0.35, 0.0, 0.15),
        origin_type="env",
        env_index=0,
        asset_name=None,
    )
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=2.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    table: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.0),
            rot=(0.707, 0.0, 0.0, 0.707),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
    )
    ground_prim_path: str = "/World/ground"
    ground_z: float = -1.05

    robot: ArticulationCfg = UR3E_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, 0.0, 0.055), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.8, 0.8, 0.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    gripper: GripperCfg = SuctionGripperCfg()
    ee_body_name: str = "tool0"
    quat_sign_mode: str = "positive-max"

    # Offset from the ``ee_body_name`` body origin to the *grasp point* (in the
    # body's local frame). This is the Franka analogue: their ``FrameTransformer``
    # parents an EE frame at ``panda_hand + (0, 0, 0.1034)``. For the UR3e +
    # Hand-E stack, tool0 -> the point where the closed blade pads meet is
    # +0.119 m along tool0's local +Z (which points away from the wrist toward
    # the gripper). Derived empirically from the bump-test: first table contact
    # at tool0 z = 0.155 with table top z = 0.021 implies a 134 mm tool0 -> blade
    # tip offset; the 31 mm-thick blade pads put their *center* (where grasping
    # happens) ~119 mm below tool0.
    ee_grasp_offset_local: tuple[float, float, float] = (0.0, 0.0, 0.119)

    # Optional contact sensor on the end-effector. When set, the env reads the
    # total net normal contact force on the sensor bodies and applies
    # ``ee_contact_penalty_*`` as a reward penalty for force *above* a threshold.
    # Left None on the suction variant.
    #
    # NOTE: We deliberately don't filter via ``filter_prim_paths_expr`` to
    # isolate "table contacts" vs "cube contacts". The table USD ships as a
    # static collider with no ``RigidBodyAPI``, so PhysX never routes contacts
    # through the filter (verified: net_force=285 N on slam, force_matrix=0 N
    # for the same slam). Instead we use the threshold to discriminate:
    # cube grasps peak around 40 N (2 fingers x 20 N effort_limit_sim each),
    # so any threshold > 40 N admits only "harder than a grasp" contacts.
    contact_sensor: ContactSensorCfg | None = None
    ee_contact_penalty_weight: float = -0.05
    # Threshold > 2 * effort_limit_sim (= 40 N steady cube grasp force) PLUS
    # headroom for the impulse spike at the moment of first finger-cube contact.
    # At 50 N the policy was being lightly penalised for honest grasps; at 80 N
    # only the table-slam regime (200-285 N from the bump test) gets caught.
    ee_contact_penalty_threshold: float = 80.0  # Newtons

    # --- delta TCP action shaping ---
    tcp_pos_action_scale: float = 0.02
    tcp_rot_action_scale: float = 0.12
    ik_method: str = "dls"

    # --- cube sampling, in UR "Base" frame for x/y/z positions ---
    cube_pos_x_range: tuple[float, float] = (-0.42, -0.25)
    cube_pos_y_range: tuple[float, float] = (-0.16, 0.16)
    cube_rest_center_z: float = 0.055
    cube_half_extent: float = 0.04
    cube_drop_height: float = -0.05

    # --- goal sampling (lift target), in UR "Base" frame ---
    # Sized to the UR3e's *actual* reachable workspace. The UR3e has ~500 mm
    # reach from the shoulder; goals beyond that get clipped by IK soft limits
    # and starve the policy of a usable gradient. Conservative envelope: world
    # x in (0.20, 0.40) -> UR Base x in (-0.40, -0.20); world y in (-0.15, 0.15);
    # world z in (0.10, 0.25) -- comfortably above the table and below the
    # vertical reach limit.
    goal_pos_x_range: tuple[float, float] = (-0.40, -0.20)
    goal_pos_y_range: tuple[float, float] = (-0.15, 0.15)
    goal_pos_z_range: tuple[float, float] = (0.10, 0.25)

    # --- reward shaping (matches Franka manager-based lift weights / std) ---
    reaching_weight: float = 1.0
    reaching_std: float = 0.1
    lifting_min_height: float = 0.04
    lifting_weight: float = 15.0
    goal_tracking_std: float = 0.3
    goal_tracking_weight: float = 16.0
    goal_tracking_fine_std: float = 0.05
    goal_tracking_fine_weight: float = 5.0

    # --- regularisers (constant weights; no curriculum) ---
    action_rate_l2_weight: float = -1.0e-4
    joint_vel_l2_weight: float = -1.0e-4

    def __post_init__(self) -> None:
        self.sim.render_interval = self.decimation
        self.action_space = 6 + self.gripper.action_dim
        self.observation_space = 25 + self.gripper.obs_dim

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625


@configclass
class UR3eLiftCubeHandEEnvCfg(UR3eLiftCubeEnvCfg):
    """Lift-cube variant using a combined UR3e + Robotiq Hand-E articulation.

    Geometry note: cube is set to scale 0.5 (DexCube intrinsic ~60 mm -> 30 mm
    here, NOT 50 mm as a naive 100 mm-cube assumption would give). At 30 mm
    the cube fits inside the ~40 mm open jaw gap with ~5 mm clearance per side,
    AND the blade tip clears the table top by ~3 mm at the grasp pose
    (body z = cube_center + ee_grasp_offset = 0.036 + 0.119 = 0.155 -> blade
    tip at world z = 0.024, vs table top at 0.021). Smaller cubes leave no
    blade-tip clearance and force the gripper to slam through the table to
    reach the cube center.
    """

    robot: ArticulationCfg = UR3E_ROBOTIQ_HANDE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    gripper: GripperCfg = RobotiqHandEGripperCfg()

    contact_sensor: ContactSensorCfg | None = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/Robotiq_Hand_E/.*_gripper",
        history_length=1,
        track_air_time=False,
    )

    def __post_init__(self) -> None:
        self.cube.spawn.scale = (0.5, 0.5, 0.5)
        self.cube.init_state.pos = (0.35, 0.0, 0.036)
        self.cube_half_extent = 0.015
        self.cube_rest_center_z = 0.036
        # Franka-style threshold: BELOW the cube's resting height (here, at the
        # table top) so ``lifted = True`` even while the cube is on the table.
        # This is what keeps ``object_goal_tracking`` producing a live gradient
        # to move the cube toward the goal -- and that gradient is what makes
        # the policy *try* the close-and-lift sequence in the first place.
        # See Franka manager-based lift: minimal_height=0.04 with rest=0.055.
        self.lifting_min_height = self.cube_half_extent
        super().__post_init__()
