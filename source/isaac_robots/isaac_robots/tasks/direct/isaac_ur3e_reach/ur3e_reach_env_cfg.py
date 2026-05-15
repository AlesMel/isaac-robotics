"""Configuration for the UR3e Reach task.

The task: move the end-effector to a randomly sampled 3D target inside the
arm's workspace. Absolute joint-position-target control, no obstacles, no gripper.
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
    # from the arm dims + observation toggles + selected gripper dims.
    action_space: int = 6
    observation_space: int = 21
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
        num_envs=1024,
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
    action_scale: float = 0.5
    """Max absolute joint-position offset from the default pose, in radians.

    This mirrors Isaac Lab's UR10 reach action term:
    ``JointPositionActionCfg(scale=0.5, use_default_offset=True)``.
    """

    include_ee_quat: bool = False
    """Include end-effector quaternion in policy observations.

    Hardware testing showed that the quaternion representation can dominate
    the policy near 180-degree TCP rotations. The default sim-to-real policy
    therefore uses the 21D layout:
    ``[q(6), qd(6), ee_pos(3), target_pos(3), target_error(3)]``.
    """

    ee_quat_sign_mode: str = "positive-max"
    """Quaternion sign convention when ``include_ee_quat`` is enabled.

    ``positive-max`` makes the largest-magnitude quaternion component
    positive, giving a deterministic representative for the q / -q pair.
    """

    ee_body_name: str = "tool0"
    """Body frame used as the end-effector for FK and reward distance.

    ``tool0`` is the canonical UR TCP frame and matches what
    ``ur_rtde.RTDEReceiveInterface.getActualTCPPose()`` reports on the real
    robot, so observations stay consistent across sim and hardware. It is
    only present in the USD when ``convert_urdf`` is run *without*
    ``--merge-joints`` (with ``--fix-base`` to anchor the arm).
    """

    # --- workspace target sampling, in UR "Base" frame (meters) ---
    # Matches the frame ur_rtde.getActualTCPPose() reports on hardware, so
    # the same target coordinates work in sim and on the real arm. The env
    # flips x and y internally when computing world positions for the
    # marker and reward (see ur3e_reach_env.py: self._ur_base_flip).
    #
    # Centered on the home TCP at roughly (-0.30, -0.13, +0.30) -- verify
    # by running the same default joint pose on hardware and reading
    # getActualTCPPose(). Box stays well within the UR3e's 0.5 m reach.
    target_pos_x_range: tuple[float, float] = (-0.42, -0.18)
    target_pos_y_range: tuple[float, float] = (-0.30, 0.05)
    target_pos_z_range: tuple[float, float] = (0.15, 0.40)

    target_current_ee_probability: float = 0.0
    """Fraction of resets whose target starts at the current end-effector pose.

    These episodes teach the policy that zero target error should produce a
    near-zero action instead of an arbitrary learned motion bias. Disabled
    while the base reach policy is still being learned -- a non-zero value
    rewards a "stand still" policy and dilutes the reaching signal.
    """

    target_current_ee_noise_std: float = 0.002
    """Meters of Gaussian target noise for target-current reset episodes."""

    # --- reward weights, matching Isaac Lab's UR10 reach reward terms ---
    position_tracking_weight: float = -0.2
    """Weight for the L2 end-effector position error term."""

    position_tracking_fine_weight: float = 0.1
    """Weight for the tanh-kernel fine position tracking term."""

    position_tracking_fine_std: float = 0.1
    """Distance scale (m) for fine position tracking."""

    orientation_tracking_weight: float = 0.0
    """Weight for the shortest-path end-effector orientation error term.

    Disabled by default: the target quaternion is set to the EE quat at reset,
    so any motion that satisfies the position target also incurs orientation
    error -- the two terms fight each other on a 6-DOF arm. Re-enable only
    once a real per-episode orientation command is sampled and the policy
    can observe the EE quaternion (``include_ee_quat=True``).
    """

    action_rate_l2_weight_initial: float = -0.0001
    """Initial action-rate penalty weight."""

    action_rate_l2_weight_final: float = -0.005
    """Action-rate penalty weight after the curriculum step."""

    joint_vel_l2_weight_initial: float = -0.0001
    """Initial joint-velocity penalty weight."""

    joint_vel_l2_weight_final: float = -0.001
    """Joint-velocity penalty weight after the curriculum step."""

    reward_curriculum_steps: int = 4500
    """Step after which action-rate and joint-velocity weights switch to final values."""

    # --- sim-to-real scaffolds (declared, not yet wired) ---
    randomize_dynamics: bool = False
    action_noise_std: float = 0.0
    obs_noise_std: float = 0.0

    def __post_init__(self) -> None:
        self.sim.render_interval = self.decimation
        self.action_space = 6 + self.gripper.action_dim
        arm_obs_dim = 25 if self.include_ee_quat else 21
        self.observation_space = arm_obs_dim + self.gripper.obs_dim
