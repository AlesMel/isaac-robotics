from __future__ import annotations

from typing import Dict

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

# Procedural obstacle shapes cycled across obstacle instances (no USD/Nucleus required).
_OBSTACLE_SHAPES = [
    sim_utils.CuboidCfg(
        size=(0.5, 0.5, 0.8),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    ),
    sim_utils.CylinderCfg(
        radius=0.25,
        height=0.9,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    ),
    sim_utils.SphereCfg(
        radius=0.3,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    ),
]


@configclass
class UdsObstacleCfg:
    count: int = 8
    prim_path_root: str = "/World/Obstacles"
    translation_step: tuple[float, float, float] = (1.4, 0.9, 0.0)
    base_translation: tuple[float, float, float] = (1.5, -3.0, 0.0)
    collision_group: int = -1


@configclass
class SpawnManagerCfg:
    obstacles: UdsObstacleCfg = UdsObstacleCfg()

    def build_assets(self) -> Dict[str, AssetBaseCfg]:
        assets: Dict[str, AssetBaseCfg] = {}
        for index in range(self.obstacles.count):
            spawn_cfg = _OBSTACLE_SHAPES[index % len(_OBSTACLE_SHAPES)]
            x = self.obstacles.base_translation[0] + index * self.obstacles.translation_step[0]
            y = self.obstacles.base_translation[1] + ((-1) ** index) * self.obstacles.translation_step[1]
            z = self.obstacles.base_translation[2] + self.obstacles.translation_step[2]
            assets[f"obstacle_{index:02d}"] = AssetBaseCfg(
                prim_path=f"{self.obstacles.prim_path_root}/obstacle_{index:02d}",
                spawn=spawn_cfg,
                init_state=AssetBaseCfg.InitialStateCfg(pos=(x, y, z)),
                collision_group=self.obstacles.collision_group,
            )
        return assets
