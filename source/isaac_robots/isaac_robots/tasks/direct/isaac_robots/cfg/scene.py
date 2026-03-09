from __future__ import annotations

from dataclasses import field
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

    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=2500.0),
    )

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

        # Clear private helper configs: Isaac Lab's entity scanner iterates over cfg.__dict__
        # and raises ValueError for any non-None value that isn't a recognized asset type.
        self._sensor_selection = None
        self._obstacle_cfg = None
        self._spawn_manager = None
