from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.envs import ViewerCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

from .cfg import MULTI_RANGER_CFG, CRAZYFLIE_CFG, SensorSelectionCfg, LIDAR_CFG


@configclass
class ObstacleNavEnvCfg(DirectRLEnvCfg):
    episode_length_s: float = 10.0
    decimation: int = 2
    action_space: int = 4
    observation_space: int = 12
    state_space: int = 0
    debug_vis = True

    viewer: ViewerCfg = ViewerCfg(
        eye=(2.0, 2.0, 2.0),
        lookat=(0.0, 0.0, 0.5),
        origin_type="env",
        env_index=0,
        asset_name="robot",
    )

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
    warehouse: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Warehouse",
        spawn=UsdFileCfg(
            usd_path=os.path.join(os.path.dirname(__file__), "warehouse.usd"),
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=20.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )
    sensor_selection: SensorSelectionCfg = SensorSelectionCfg(enable_lidar=True, enable_camera=False)
    
    waypoint_markers: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/figure8_waypoints",
        markers={
            "waypoint": sim_utils.SphereCfg(
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0),  # green
                ),
            ),
        },
    )

    # lidar = LIDAR_CFG.replace(
    #     prim_path="/World/envs/env_.*/Robot/body",
    # )
    lidar = MULTI_RANGER_CFG.replace(
        prim_path="/World/envs/env_.*/Robot/body",
    )
    thrust_to_weight: float = 1.9
    moment_scale: float = 0.01
    lin_vel_reward_scale: float = -0.05
    ang_vel_reward_scale: float = -0.01
    distance_to_goal_reward_scale: float = 15.0
    goal_reached_bonus: float = 15.0
    goal_reached_threshold: float = 0.2
    randomize_initial_episode_length: bool = True
    goal_z_range: tuple[float, float] = (0.4, 1.0)
    goal_min_distance_from_spawn: float = 2.0

    def __post_init__(self) -> None:
        self.sim.render_interval = self.decimation
        self.terrain.num_envs = self.scene.num_envs
        self.terrain.env_spacing = self.scene.env_spacing
        if self.sensor_selection.enable_lidar:
            self.observation_space = 12 + self.sensor_selection.lidar_flat_dim
