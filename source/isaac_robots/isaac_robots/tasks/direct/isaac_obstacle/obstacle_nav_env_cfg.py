from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from .cfg import CRAZYFLIE_CFG, ObstacleNavObservationCfg, ObstaclePatternCfg, SensorSelectionCfg, build_lidar_cfg


@configclass
class ObstacleNavEnvCfg(DirectRLEnvCfg):
    episode_length_s: float = 10.0
    decimation: int = 2
    action_space: int = 4
    observation_space: int = 12
    state_space: int = 0
    debug_vis = True

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
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
        env_spacing=12.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    sensor_selection: SensorSelectionCfg = SensorSelectionCfg(enable_lidar=True, enable_camera=False)
    obstacle_cfg: ObstaclePatternCfg = ObstaclePatternCfg()
    observation_cfg: ObstacleNavObservationCfg = ObstacleNavObservationCfg()
    lidar = None

    thrust_to_weight: float = 1.9
    moment_scale: float = 0.01
    lin_vel_reward_scale: float = -0.05
    ang_vel_reward_scale: float = -0.01
    distance_to_goal_reward_scale: float = 15.0
    obstacle_proximity_reward_scale: float = -5.0
    collision_penalty: float = -15.0
    goal_reached_bonus: float = 15.0
    collision_margin: float = 0.15
    goal_reached_threshold: float = 0.2
    obstacle_safety_distance: float = 1.0
    randomize_initial_episode_length: bool = True
    goal_z_range: tuple[float, float] = (0.6, 1.5)
    goal_min_distance_from_spawn: float = 2.0
    goal_clearance_margin: float = 0.75

    def __post_init__(self) -> None:
        self.sim.render_interval = self.decimation
        self.terrain.num_envs = self.scene.num_envs
        self.terrain.env_spacing = self.scene.env_spacing
        self.lidar = build_lidar_cfg(self.sensor_selection)
        self.observation_cfg.sensor_selection = self.sensor_selection
        self.observation_cfg.base_state_shape = (12,)
        self.observation_cfg.__post_init__()
        self.observation_space = 12 + self.observation_cfg.lidar_flat_dim
