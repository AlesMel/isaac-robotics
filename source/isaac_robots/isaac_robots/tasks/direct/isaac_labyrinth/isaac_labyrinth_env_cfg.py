from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.envs import ViewerCfg

from .cfg import MULTI_RANGER_CFG, CRAZYFLIE_CFG, CRAZYFLIE_AI_CAMERA_CFG, SensorSelectionCfg
from isaac_robots.scripts.labyrinth_builder import LabyrinthCfg


@configclass
class DomainRandomizationCfg:
    """Domain randomization parameters (all default to off)."""
    thrust_noise_std: float = 0.0
    sensor_noise_std: float = 0.0
    mass_randomization_pct: float = 0.0


@configclass
class LabyrinthEnvCfg(DirectRLEnvCfg):
    """Base config shared by all labyrinth challenge variants."""

    episode_length_s: float = 20.0
    decimation: int = 2
    action_space: int = 4
    observation_space: int = 12
    state_space: int = 0
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

    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=5.0,
        replicate_physics=False,
        clone_in_fabric=False,
    )

    # Labyrinth generation
    labyrinth: LabyrinthCfg = LabyrinthCfg(
        size=3.5,
        wall_height=1.2,
        wall_thickness=0.1,
        seed=None,
        difficulty=0.5,
        spawn_walls=True,
        n_layouts=64,  # set equal to num_envs for a unique layout per env
    )

    # Sensors
    sensor_selection: SensorSelectionCfg = SensorSelectionCfg(enable_lidar=True)
    lidar = MULTI_RANGER_CFG.replace(
        prim_path="/World/envs/env_.*/Robot/body",
    )
    # Crazyflie AI bundle HM01B0 monochrome camera (disabled by default).
    # Enable by setting sensor_selection.enable_camera = True and assigning
    # camera = CRAZYFLIE_AI_CAMERA_CFG (or a .replace(...) of it).
    camera: TiledCameraCfg | None = None
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
    lin_vel_reward_scale: float = -0.01
    ang_vel_reward_scale: float = -0.01
    distance_to_goal_reward_scale: float = 16.0
    wall_proximity_reward_scale: float = -5.0
    goal_reached_bonus: float = 10.0
    alive_bonus: float = 0.5
    alive_bonus_min_speed: float = 0.1
    tilt_reward_scale: float = -0.1
    action_smoothness_scale: float = -0.03

    # Thresholds
    goal_reached_threshold: float = 0.25
    goal_z_range: tuple[float, float] = (0.4, 1.2)
    wall_danger_distance: float = 0.3

    # Domain randomization
    domain_rand: DomainRandomizationCfg = DomainRandomizationCfg()

    def __post_init__(self) -> None:
        self.sim.render_interval = self.decimation
        self.terrain.num_envs = self.scene.num_envs
        self.terrain.env_spacing = self.scene.env_spacing
        if self.sensor_selection.enable_lidar:
            self.observation_space = 12 + self.sensor_selection.lidar_flat_dim
        if self.sensor_selection.enable_camera and self.camera is not None:
            self.observation_space += self.sensor_selection.camera_flat_dim
            if self.scene.num_envs > 512:
                import warnings
                warnings.warn(
                    f"Camera is enabled at 324x244 with {self.scene.num_envs} envs. "
                    "This may exceed GPU memory. Consider setting num_envs <= 512.",
                    stacklevel=2,
                )
