"""Configuration for the UR3e Reach task.

The task: move the end-effector to a randomly sampled 3D target inside the
arm's workspace. Joint-position-delta control, no obstacles, no gripper.
This is the simplest RL task in the extension and a good first stop on the
sim-to-real path.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from .._shared.grippers import GripperCfg, NoGripperCfg
from .cfg import UR3E_CFG, SensorSelectionCfg


@configclass
class UR3eReachEnvCfg(DirectRLEnvCfg):
    # --- episode / control rate ---
    episode_length_s: float = 5.0
    decimation: int = 2  # 120 Hz sim -> 60 Hz policy
    debug_vis: bool = True

    # action_space and observation_space are computed in __post_init__
    # from the arm dims + the selected gripper's dims.
    action_space: int = 6
    observation_space: int = 25
    state_space: int = 0

    # --- scene / sim ---
    viewer: ViewerCfg = ViewerCfg(
        eye=(1.5, 1.5, 1.0),
        lookat=(0.0, 0.0, 0.3),
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
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        debug_vis=False,
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=2.0,
        replicate_physics=True,
        # Fabric cloning silently drops envs when the source articulation
        # has dangling visual references -- the converted UR3e USD has one
        # for the geometry-less ``tool0`` frame. The slower USD-based
        # cloner handles it correctly.
        clone_in_fabric=False,
    )

    # --- robot + gripper (the swap-point) ---
    robot: ArticulationCfg = UR3E_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    gripper: GripperCfg = NoGripperCfg()
    sensor_selection: SensorSelectionCfg = SensorSelectionCfg()

    # --- action shaping ---
    action_scale: float = 0.1
    """Max joint position delta per policy step, in radians."""

    ee_body_name: str = "tool0"
    """Body frame used as the end-effector for FK and reward distance.

    ``tool0`` is the canonical UR TCP frame and matches what
    ``ur_rtde.RTDEReceiveInterface.getActualTCPPose()`` reports on the real
    robot, so observations stay consistent across sim and hardware. It is
    only present in the USD when ``convert_urdf`` is run *without*
    ``--merge-joints`` (with ``--fix-base`` to anchor the arm).
    """

    # --- workspace target sampling, in robot base frame (meters) ---
    target_pos_x_range: tuple[float, float] = (-0.30, 0.30)
    target_pos_y_range: tuple[float, float] = (-0.30, 0.30)
    target_pos_z_range: tuple[float, float] = (0.10, 0.45)

    # --- reward weights ---
    distance_reward_scale: float = 1.0
    distance_reward_tanh_std: float = 0.1
    """Distance scale (m) for the tanh shaping. Smaller = sharper reward peak."""

    action_penalty_scale: float = 0.005
    joint_limit_penalty: float = 5.0
    success_threshold: float = 0.02
    """EE-to-target distance (m) below which a step counts as success."""

    success_bonus: float = 5.0
    success_steps_required: int = 5
    """Consecutive success steps required to terminate the episode."""

    # --- sim-to-real scaffolds (declared, not yet wired) ---
    randomize_dynamics: bool = False
    action_noise_std: float = 0.0
    obs_noise_std: float = 0.0

    def __post_init__(self) -> None:
        self.sim.render_interval = self.decimation
        self.action_space = 6 + self.gripper.action_dim
        self.observation_space = 25 + self.gripper.obs_dim
