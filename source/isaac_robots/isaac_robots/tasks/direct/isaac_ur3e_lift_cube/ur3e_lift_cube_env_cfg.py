"""Configuration for the UR3e lift-cube direct task."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
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
    # Mirrors UniformPoseCommand ranges in the Franka lift example
    # (world x in (0.4, 0.6) maps to UR base x in (-0.6, -0.4); y/z unchanged
    # by the 180-deg-about-Z flip between URDF root and "UR Base").
    goal_pos_x_range: tuple[float, float] = (-0.6, -0.4)
    goal_pos_y_range: tuple[float, float] = (-0.25, 0.25)
    goal_pos_z_range: tuple[float, float] = (0.25, 0.5)

    # --- reward shaping (matches Franka manager-based lift weights / std) ---
    reaching_weight: float = 1.0
    reaching_std: float = 0.1
    lifting_min_height: float = 0.04
    lifting_weight: float = 15.0
    goal_tracking_std: float = 0.3
    goal_tracking_weight: float = 16.0
    goal_tracking_fine_std: float = 0.05
    goal_tracking_fine_weight: float = 5.0

    # --- regularisation curriculum (Franka lift CurriculumCfg) ---
    action_rate_l2_weight_initial: float = -1.0e-4
    action_rate_l2_weight_final: float = -1.0e-1
    joint_vel_l2_weight_initial: float = -1.0e-4
    joint_vel_l2_weight_final: float = -1.0e-1
    reward_curriculum_steps: int = 10000

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

    The Hand-E performs a *real friction grasp*: the fingers physically clamp
    the cube and PhysX contact/friction holds it (no kinematic attach). The env
    is unchanged -- it delegates grasp logic to the gripper, and the Hand-E's
    ``update_attachment`` is a no-op.

    PHYSICAL CONSTRAINT -- cube vs jaw opening: the Robotiq Hand-E has only a
    ~50 mm stroke, so it can only grasp an object that fits within its open
    fingers. The base task's DexCube at scale 0.8 (~80 mm) is too wide for the
    jaws, so this subclass overrides the cube to scale 0.5 (~50 mm) and shifts
    ``cube_half_extent``, ``cube_rest_center_z``, and ``lifting_min_height`` to
    stay consistent with the smaller footprint.
    """

    robot: ArticulationCfg = UR3E_ROBOTIQ_HANDE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    gripper: GripperCfg = RobotiqHandEGripperCfg()

    def __post_init__(self) -> None:
        self.cube.spawn.scale = (0.5, 0.5, 0.5)
        self.cube.init_state.pos = (0.35, 0.0, 0.040)
        self.cube_half_extent = 0.025
        self.cube_rest_center_z = 0.040
        # Cube rests at z ~= half_extent on the table. Match Franka's pattern:
        # threshold == rest height so any actual lift starts paying out.
        self.lifting_min_height = self.cube_half_extent
        super().__post_init__()
