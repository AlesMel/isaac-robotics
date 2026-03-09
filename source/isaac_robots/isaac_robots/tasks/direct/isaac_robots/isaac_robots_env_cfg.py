from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from .cfg import CRAZYFLIE_CFG
from .cfg import SensorSelectionCfg

@configclass
class CrazyflieEnvCfg(DirectRLEnvCfg):
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=4.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    sensor_selection: SensorSelectionCfg = SensorSelectionCfg()

    thrust_to_weight: float = 1.9
    moment_scale: float = 0.01
    lin_vel_reward_scale: float = -0.05
    ang_vel_reward_scale: float = -0.01
    distance_to_goal_reward_scale: float = 15.0
    randomize_initial_episode_length: bool = True

    def __post_init__(self) -> None:
        self.sim.render_interval = self.decimation