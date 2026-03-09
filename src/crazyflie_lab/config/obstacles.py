from __future__ import annotations

import os
from typing import Dict

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

from .assets import DEFAULT_NUCLEUS_ROOT


@configclass
class UdsObstacleCfg:
    count: int = 8
    prim_path_root: str = "{ENV_REGEX_NS}/World/Obstacles"
    asset_usd_paths: tuple[str, ...] = (
        f"{DEFAULT_NUCLEUS_ROOT}/Props/Blocks/concrete_block.usd",
        f"{DEFAULT_NUCLEUS_ROOT}/Props/Barrels/oil_barrel.usd",
        f"{DEFAULT_NUCLEUS_ROOT}/Props/Pallets/wood_pallet.usd",
    )
    translation_step: tuple[float, float, float] = (1.4, 0.9, 0.0)
    base_translation: tuple[float, float, float] = (1.5, -3.0, 0.0)
    collision_group: int = -1


@configclass
class SpawnManagerCfg:
    obstacles: UdsObstacleCfg = UdsObstacleCfg()

    def build_assets(self) -> Dict[str, AssetBaseCfg]:
        assets: Dict[str, AssetBaseCfg] = {}
        for index in range(self.obstacles.count):
            usd_path = self.obstacles.asset_usd_paths[index % len(self.obstacles.asset_usd_paths)]
            x = self.obstacles.base_translation[0] + index * self.obstacles.translation_step[0]
            y = self.obstacles.base_translation[1] + ((-1) ** index) * self.obstacles.translation_step[1]
            z = self.obstacles.base_translation[2] + self.obstacles.translation_step[2]
            assets[f"obstacle_{index:02d}"] = AssetBaseCfg(
                prim_path=f"{self.obstacles.prim_path_root}/obstacle_{index:02d}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=usd_path,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(x, y, z)),
                collision_group=self.obstacles.collision_group,
            )
        return assets
