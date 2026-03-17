from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.envs import ViewerCfg
from isaaclab.markers import VisualizationMarkersCfg

from .cfg import MULTI_RANGER_CFG, CRAZYFLIE_CFG, SensorSelectionCfg
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
        eye=(2.0, 2.0, 2.0),
        lookat=(0.0, 0.0, 0.5),
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
        env_spacing=8.0,
        replicate_physics=False,
        clone_in_fabric=False,
    )

    # Labyrinth generation — overridden per challenge variant
    labyrinth: LabyrinthCfg = LabyrinthCfg(
        challenge="corridor",
        size=6.0,
        wall_height=1.5,
        wall_thickness=0.1,
        seed=42,
        difficulty=0.5,
    )

    # Sensors
    sensor_selection: SensorSelectionCfg = SensorSelectionCfg(enable_lidar=True)
    lidar = MULTI_RANGER_CFG.replace(
        prim_path="/World/envs/env_.*/Robot/body",
    )
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/body",
        history_length=1,
        track_air_time=False,
    )
    collision_force_threshold: float = 0.5
    grace_seconds: float = 0.3

    # Waypoint markers for debug visualization
    waypoint_markers: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/ring_waypoints",
        markers={
            "waypoint": sim_utils.SphereCfg(
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0),
                ),
            ),
        },
    )

    # Action scaling
    thrust_to_weight: float = 1.9
    moment_scale: float = 0.01

    # Reward scales
    lin_vel_reward_scale: float = -0.04
    ang_vel_reward_scale: float = -0.05
    distance_to_goal_reward_scale: float = 16.0
    wall_proximity_reward_scale: float = -5.0
    goal_reached_bonus: float = 10.0
    alive_bonus: float = 0.5
    tilt_reward_scale: float = -0.5
    action_smoothness_scale: float = -0.15

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


# ---------------------------------------------------------------------------
# Per-challenge config variants
# ---------------------------------------------------------------------------


@configclass
class CorridorEnvCfg(LabyrinthEnvCfg):
    labyrinth: LabyrinthCfg = LabyrinthCfg(
        challenge="corridor", size=6.0, wall_height=1.5, seed=42, difficulty=0.5,
    )


@configclass
class GateSlalomEnvCfg(LabyrinthEnvCfg):
    labyrinth: LabyrinthCfg = LabyrinthCfg(
        challenge="gate_slalom", size=6.0, wall_height=1.5, seed=42, difficulty=0.5,
    )


@configclass
class PillarForestEnvCfg(LabyrinthEnvCfg):
    labyrinth: LabyrinthCfg = LabyrinthCfg(
        challenge="pillar_forest", size=6.0, wall_height=1.5, seed=42, difficulty=0.5,
    )


@configclass
class VerticalLayersEnvCfg(LabyrinthEnvCfg):
    labyrinth: LabyrinthCfg = LabyrinthCfg(
        challenge="vertical_layers", size=6.0, wall_height=1.5, seed=42, difficulty=0.5,
    )


@configclass
class RoomMazeEnvCfg(LabyrinthEnvCfg):
    labyrinth: LabyrinthCfg = LabyrinthCfg(
        challenge="room_maze", size=6.0, wall_height=1.5, seed=42, difficulty=0.5,
    )
