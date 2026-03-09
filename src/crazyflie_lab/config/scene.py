from __future__ import annotations

from typing import Optional

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from .assets import CRAZYFLIE_CFG
from .obstacles import SpawnManagerCfg, UdsObstacleCfg
from .sensors import SensorSelectionCfg, build_camera_cfg, build_lidar_cfg


@configclass
class CrazyflieSceneCfg(InteractiveSceneCfg):
    robot = CRAZYFLIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    sensor_selection: SensorSelectionCfg = SensorSelectionCfg()
    obstacle_cfg: UdsObstacleCfg = UdsObstacleCfg()
    spawn_manager: SpawnManagerCfg = SpawnManagerCfg()

    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=2500.0),
    )

    lidar: Optional[object] = None
    camera: Optional[object] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.spawn_manager = SpawnManagerCfg(obstacles=self.obstacle_cfg)

        self.lidar = build_lidar_cfg(
            self.sensor_selection,
            prim_path="{ENV_REGEX_NS}/Robot/body",
        )
        self.camera = build_camera_cfg(
            self.sensor_selection,
            prim_path="{ENV_REGEX_NS}/Robot/body/front_camera",
        )

        for name, asset_cfg in self.spawn_manager.build_assets().items():
            setattr(self, name, asset_cfg)
