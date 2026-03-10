from __future__ import annotations

from dataclasses import field

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from .assets import CRAZYFLIE_CFG
from .obstacles import ObstaclePatternCfg
from .sensors import SensorSelectionCfg, build_lidar_cfg


@configclass
class ObstacleNavSceneCfg(InteractiveSceneCfg):
    robot = CRAZYFLIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ground: AssetBaseCfg = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    dome_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=2500.0),
    )

    lidar = None
    _sensor_selection: SensorSelectionCfg = field(default_factory=SensorSelectionCfg)
    _obstacle_cfg: ObstaclePatternCfg = field(default_factory=ObstaclePatternCfg)

    def configure(self, sensor_selection: SensorSelectionCfg, obstacle_cfg: ObstaclePatternCfg | None = None) -> None:
        self._sensor_selection = sensor_selection
        self._obstacle_cfg = obstacle_cfg if obstacle_cfg is not None else ObstaclePatternCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.lidar = build_lidar_cfg(self._sensor_selection)
        self._sensor_selection = None
        self._obstacle_cfg = None


WarehouseSceneCfg = ObstacleNavSceneCfg
