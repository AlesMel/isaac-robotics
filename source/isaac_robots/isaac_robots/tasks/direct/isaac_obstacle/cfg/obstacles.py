from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.utils import configclass


_OBSTACLE_SPAWN_CONFIGS = [
    sim_utils.CuboidCfg(
        size=(0.8, 0.8, 1.2),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.2, 0.2), metallic=0.1),
    ),
    sim_utils.CylinderCfg(
        radius=0.45,
        height=1.4,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.65, 0.9), metallic=0.1),
    ),
    sim_utils.SphereCfg(
        radius=0.5,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.8, 0.35), metallic=0.1),
    ),
]

_OBSTACLE_RADII = [
    0.5 * math.sqrt(0.8**2 + 0.8**2),
    0.45,
    0.5,
]

_OBSTACLE_HEIGHTS = [1.2, 1.4, 1.0]


@configclass
class ObstaclePatternCfg:
    count: int = 8
    layout_half_extent: float = 4.5
    min_obstacle_clearance: float = 0.7
    reserved_spawn_radius: float = 1.25
    max_sampling_attempts: int = 100


def _default_obstacle_xy(index: int, obstacle_cfg: ObstaclePatternCfg) -> tuple[float, float]:
    count = max(1, obstacle_cfg.count)
    layout_radius = max(
        obstacle_cfg.reserved_spawn_radius + 0.75,
        min(obstacle_cfg.layout_half_extent - 0.6, obstacle_cfg.layout_half_extent * 0.65),
    )
    angle = (2.0 * math.pi * index) / count
    radial_scale = 1.0 if index % 2 == 0 else 0.78
    radius = layout_radius * radial_scale
    return radius * math.cos(angle), radius * math.sin(angle)


def build_source_obstacle_cfgs(count: int, obstacle_cfg: ObstaclePatternCfg | None = None) -> dict[str, RigidObjectCfg]:
    obstacle_cfg = obstacle_cfg if obstacle_cfg is not None else ObstaclePatternCfg(count=count)
    rigid_objects: dict[str, RigidObjectCfg] = {}
    for index in range(count):
        spawn_cfg = _OBSTACLE_SPAWN_CONFIGS[index % len(_OBSTACLE_SPAWN_CONFIGS)]
        height = _OBSTACLE_HEIGHTS[index % len(_OBSTACLE_HEIGHTS)]
        x, y = _default_obstacle_xy(index, obstacle_cfg)
        rigid_objects[f"obstacle_{index:02d}"] = RigidObjectCfg(
            prim_path=f"/World/envs/env_0/Obstacle_{index:02d}",
            spawn=spawn_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, 0.5 * height)),
        )
    return rigid_objects


def build_obstacle_collection_cfg(
    count: int, obstacle_cfg: ObstaclePatternCfg | None = None
) -> RigidObjectCollectionCfg:
    obstacle_cfg = obstacle_cfg if obstacle_cfg is not None else ObstaclePatternCfg(count=count)
    rigid_objects: dict[str, RigidObjectCfg] = {}
    for index in range(count):
        height = _OBSTACLE_HEIGHTS[index % len(_OBSTACLE_HEIGHTS)]
        x, y = _default_obstacle_xy(index, obstacle_cfg)
        rigid_objects[f"obstacle_{index:02d}"] = RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/Obstacle_{index:02d}",
            spawn=None,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, 0.5 * height)),
        )
    return RigidObjectCollectionCfg(rigid_objects=rigid_objects)


def get_obstacle_dim_tensors(count: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    radii = [_OBSTACLE_RADII[index % len(_OBSTACLE_RADII)] for index in range(count)]
    heights = [_OBSTACLE_HEIGHTS[index % len(_OBSTACLE_HEIGHTS)] for index in range(count)]
    return (
        torch.tensor(radii, dtype=torch.float, device=device),
        torch.tensor(heights, dtype=torch.float, device=device),
    )
