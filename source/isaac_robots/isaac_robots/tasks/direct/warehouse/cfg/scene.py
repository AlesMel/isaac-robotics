from __future__ import annotations

from dataclasses import field
from typing import Optional

from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from .assets import CRAZYFLIE_CFG, WAREHOUSE_CFG
from .obstacles import SpawnManagerCfg, UdsObstacleCfg
from .sensors import SensorSelectionCfg, build_camera_cfg, build_lidar_cfg


@configclass
class WarehouseSceneCfg(InteractiveSceneCfg):
    robot = CRAZYFLIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    warehouse: AssetBaseCfg = WAREHOUSE_CFG.replace(prim_path="{ENV_REGEX_NS}/Warehouse")

    lidar: Optional[object] = None
    camera: Optional[object] = None
    _sensor_selection: SensorSelectionCfg = field(default_factory=SensorSelectionCfg)
    _obstacle_cfg: UdsObstacleCfg = field(default_factory=UdsObstacleCfg)
    _spawn_manager: SpawnManagerCfg = field(default_factory=SpawnManagerCfg)

    def configure(
        self,
        sensor_selection: SensorSelectionCfg,
        obstacle_cfg: UdsObstacleCfg | None = None,
    ) -> None:
        self._sensor_selection = sensor_selection
        self._obstacle_cfg = obstacle_cfg if obstacle_cfg is not None else UdsObstacleCfg()
        self._spawn_manager = SpawnManagerCfg(obstacles=self._obstacle_cfg)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._spawn_manager = SpawnManagerCfg(obstacles=self._obstacle_cfg)

        self.lidar = build_lidar_cfg(
            self._sensor_selection,
            prim_path="{ENV_REGEX_NS}/Robot/body",
        )
        self.camera = build_camera_cfg(
            self._sensor_selection,
            prim_path="{ENV_REGEX_NS}/Robot/body/front_camera",
        )

        for name, asset_cfg in self._spawn_manager.build_assets().items():
            setattr(self, name, asset_cfg)

        self._sensor_selection = None
        self._obstacle_cfg = None
        self._spawn_manager = None
