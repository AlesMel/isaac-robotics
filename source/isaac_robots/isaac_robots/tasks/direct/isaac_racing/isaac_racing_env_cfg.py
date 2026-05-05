from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from .cfg import CRAZYFLIE_CFG, CRAZYFLIE_AI_CAMERA_CFG, SensorSelectionCfg, MULTI_RANGER_CFG, MULTI_RANGER_CFG_HARD, MULTI_RANGER_CFG_VERY_HARD
from isaac_robots.scripts.racing_track_builder import RacingTrackCfg


@configclass
class DomainRandomizationCfg:
    thrust_noise_std: float = 0.0
    sensor_noise_std: float = 0.0
    mass_randomization_pct: float = 0.0
    turbulence_std: float = 0.0   # random world-frame force per axis, as fraction of robot weight

    # Per-episode track randomization (applied via PhysX rigid-body views at reset)
    reshuffle_layouts_on_reset: bool = True   # randomly reassign track layout each episode
    gate_angle_noise_deg: float = 0.0         # rotate gate post pair ±N° around its midpoint
    crossbar_z_noise_m: float = 0.0           # jitter crossbar height ±m
    scatter_xy_noise_m: float = 0.0           # jitter scatter obstacle XY ±m
    checkpoint_height_noise_m: float = 0.0    # jitter checkpoint target Z ±m


@configclass
class RacingEnvCfg(DirectRLEnvCfg):
    """Config for the Crazyflie racing circuit environment."""

    episode_length_s: float = 60.0
    decimation: int = 2
    action_space: int = 4
    observation_space: int = 18   # 12 proprio + 6 lidar; overridden in __init__ if camera
    state_space: int = 10
    frame_stack: int = 4
    debug_vis: bool = True

    viewer: ViewerCfg = ViewerCfg(
        eye=(0.0, 0.0, 10.0),
        lookat=(0.0, 0.0, 0.0),
        origin_type="env",
        env_index=0,
        asset_name="robot",
    )

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=2,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            gpu_max_rigid_contact_count=2**21,
            gpu_max_rigid_patch_count=2**19,
            gpu_found_lost_pairs_capacity=2**19,
            gpu_found_lost_aggregate_pairs_capacity=2**22,
            gpu_total_aggregate_pairs_capacity=2**19,
            gpu_collision_stack_size=2**24,
        ),
    )

    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048,
        env_spacing=5.1,
        replicate_physics=False,
        clone_in_fabric=False,
    )

    # Racing track generation
    track: RacingTrackCfg = RacingTrackCfg(
        size=5.0,
        wall_height=1.2,
        wall_thickness=0.1,
        n_control_points=6,
        track_width=0.9,
        pillar_radius_min=0.06,
        pillar_radius_max=0.10,
        radial_noise=0.25,
        checkpoint_height=0.5,
        seed=None,
        n_layouts=10,
        spawn_walls=True,
    )

    # Sensors
    sensor_selection: SensorSelectionCfg = SensorSelectionCfg()
    lidar = MULTI_RANGER_CFG.replace(
        prim_path="/World/envs/env_.*/Robot/body",
    )
    camera: CameraCfg | None = None
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/body",
        history_length=1,
        track_air_time=False,
    )
    collision_force_threshold: float = 0.5
    grace_seconds: float = 0.3

    # Action scaling
    thrust_to_weight: float = 2.0
    moment_scale: float = 0.01

    # Reward scales
    forward_speed_scale: float = 8.0      # primary: velocity component toward checkpoint
    goal_reached_bonus: float = 15.0      # checkpoint crossing bonus
    alive_bonus: float = 0.5
    wall_proximity_reward_scale: float = -5.0
    tilt_reward_scale: float = -0.1
    action_smoothness_scale: float = -0.03
    low_altitude_penalty_scale: float = -3.0
    low_altitude_threshold: float = 0.3

    # Thresholds
    goal_reached_threshold: float = 0.35  # distance to checkpoint centre (m)
    wall_danger_distance: float = 0.3

    # Domain randomization
    spawn_noise_xy_m: float = 0.15   # random XY jitter added to spawn position (m)
    domain_rand: DomainRandomizationCfg = DomainRandomizationCfg(
        thrust_noise_std=0.05,
        sensor_noise_std=0.02,
        mass_randomization_pct=0.10,
        turbulence_std=0.08,
        reshuffle_layouts_on_reset=True,
        gate_angle_noise_deg=12.0,
        crossbar_z_noise_m=0.12,
        scatter_xy_noise_m=0.20,
        checkpoint_height_noise_m=0.12,
    )

    def __post_init__(self) -> None:
        self.sim.render_interval = self.decimation
        self.terrain.num_envs = self.scene.num_envs
        self.terrain.env_spacing = self.scene.env_spacing
        if self.camera is not None:
            self.observation_space = 12 + self.frame_stack * self.camera.height * self.camera.width
        elif self.lidar is not None:
            self.observation_space = 12 + self.sensor_selection.lidar_flat_dim


@configclass
class RacingEnvCfgHard(RacingEnvCfg):
    """Harder racing variant: smaller arena, tighter turns, scatter obstacles, turbulence."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048,
        env_spacing=6.0,   # smaller arena → tighter spacing
        replicate_physics=False,
        clone_in_fabric=False,
    )

    track: RacingTrackCfg = RacingTrackCfg(
        size=3.5,              # smaller arena = much tighter turns
        wall_height=1.2,
        wall_thickness=0.1,
        n_control_points=9,    # more gates / curves
        track_width=0.65,      # narrow gate opening
        pillar_radius_min=0.07,
        pillar_radius_max=0.13,
        radial_noise=0.40,     # more irregular, sharper corners
        checkpoint_height=0.5,
        seed=None,
        n_layouts=20,          # more variety, harder to memorize layouts
        spawn_walls=True,
        n_scatter_obstacles=8, # random obstacles scattered inside arena
        scatter_radius_min=0.05,
        scatter_radius_max=0.12,
        scatter_box_prob=0.4,
    )

    lidar = MULTI_RANGER_CFG_HARD.replace(
        prim_path="/World/envs/env_.*/Robot/body",
    )

    goal_reached_threshold: float = 0.28

    domain_rand: DomainRandomizationCfg = DomainRandomizationCfg(
        turbulence_std=0.15,   # ±15% robot weight as random force per axis
    )


@configclass
class RacingEnvCfgVeryHard(RacingEnvCfgHard):
    """Full 3D challenge: figure-8 track + gate crossbars + hanging bars + height-varied checkpoints + turbulence."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=8.0,   # larger arena (5.0 m) needs more spacing
        replicate_physics=False,
        clone_in_fabric=False,
    )

    track: RacingTrackCfg = RacingTrackCfg(
        size=5.0,
        wall_height=1.2,
        wall_thickness=0.1,
        track_topology="figure8",
        n_control_points=10,   # 5 per loop
        track_width=0.75,
        pillar_radius_min=0.07,
        pillar_radius_max=0.12,
        radial_noise=0.35,
        checkpoint_height=0.5,
        vary_checkpoint_heights=True,
        checkpoint_height_min=0.25,
        checkpoint_height_max=0.95,
        seed=None,
        n_layouts=16,
        spawn_walls=True,
        add_crossbars=True,
        crossbar_radius=0.04,
        n_hanging_bars=4,
        hanging_bar_radius=0.04,
        hanging_bar_length_min=0.4,
        hanging_bar_length_max=0.8,
        oscillate_hanging_bars=True,
        osc_amplitude_min=0.10,
        osc_amplitude_max=0.25,
        osc_frequency_min=0.3,
        osc_frequency_max=0.8,
        n_scatter_obstacles=4,
        scatter_radius_min=0.05,
        scatter_radius_max=0.10,
        scatter_box_prob=0.3,
    )

    lidar = MULTI_RANGER_CFG_VERY_HARD.replace(
        prim_path="/World/envs/env_.*/Robot/body",
    )

    goal_reached_threshold: float = 0.30
    low_altitude_threshold: float = 0.1   # allow flying low to reach low checkpoints

    domain_rand: DomainRandomizationCfg = DomainRandomizationCfg(
        turbulence_std=0.20,
        thrust_noise_std=0.05,
        mass_randomization_pct=0.10,
    )
