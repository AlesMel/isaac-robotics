from __future__ import annotations

from gymnasium import spaces

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .cfg import CRAZYFLIE_CFG, SensorSelectionCfg, WAREHOUSE_CFG


@configclass
class WarehouseEnvCfg(DirectRLEnvCfg):
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=100.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    warehouse: AssetBaseCfg = WAREHOUSE_CFG
    sensor_selection: SensorSelectionCfg = SensorSelectionCfg()

    thrust_to_weight: float = 1.9
    moment_scale: float = 0.01
    lin_vel_reward_scale: float = -0.05
    ang_vel_reward_scale: float = -0.01
    distance_to_goal_reward_scale: float = 15.0
    randomize_initial_episode_length: bool = True

    def __post_init__(self) -> None:
        self.sim.render_interval = self.decimation
        warehouse_usd_path = getattr(self.warehouse.spawn, "usd_path", "")
        if not warehouse_usd_path:
            raise ValueError(
                "WAREHOUSE_USD_PATH is not set. Point it to a valid warehouse USD before loading the task."
            )
