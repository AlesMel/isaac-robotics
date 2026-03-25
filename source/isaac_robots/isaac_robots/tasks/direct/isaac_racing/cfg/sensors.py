from __future__ import annotations

from isaaclab.sensors import MultiMeshRayCasterCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass
from ..._shared.tof_pattern import CrazyflieToFPatternCfg

_RANGER_BASE_PATHS = [
    "/World/ground",
    MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="/World/envs/env_.*/racing_track/gate_.*"),
    MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="/World/envs/env_.*/racing_track/wall_.*"),
]
_RANGER_SCATTER_PATH = MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="/World/envs/env_.*/racing_track/scatter_.*")
_RANGER_CROSSBAR_PATH = MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="/World/envs/env_.*/racing_track/crossbar_.*")
_RANGER_HANGING_PATH = MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="/World/envs/env_.*/racing_track/hanging_.*")

# Base: walls + gate posts only
MULTI_RANGER_CFG = MultiMeshRayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/body",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.02)),
    pattern_cfg=CrazyflieToFPatternCfg(),
    max_distance=4.0,
    mesh_prim_paths=_RANGER_BASE_PATHS,
    debug_vis=False,
)

# Hard: adds scatter obstacles
MULTI_RANGER_CFG_HARD = MultiMeshRayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/body",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.02)),
    pattern_cfg=CrazyflieToFPatternCfg(),
    max_distance=4.0,
    mesh_prim_paths=_RANGER_BASE_PATHS + [_RANGER_SCATTER_PATH],
    debug_vis=False,
)

# VeryHard: scatter + crossbars + hanging bars
MULTI_RANGER_CFG_VERY_HARD = MultiMeshRayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/body",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.02)),
    pattern_cfg=CrazyflieToFPatternCfg(),
    max_distance=4.0,
    mesh_prim_paths=_RANGER_BASE_PATHS + [_RANGER_SCATTER_PATH, _RANGER_CROSSBAR_PATH, _RANGER_HANGING_PATH],
    debug_vis=False,
)


def _compute_ray_count(cfg: patterns.PatternBaseCfg) -> tuple[int, int]:
    if isinstance(cfg, CrazyflieToFPatternCfg):
        return 1, 6
    starts, directions = cfg.func(cfg, device="cpu")
    return 1, directions.shape[0]


_LIDAR_CHANNELS, _LIDAR_H_RAYS = _compute_ray_count(MULTI_RANGER_CFG.pattern_cfg)


@configclass
class SensorSelectionCfg:
    lidar_debug_vis: bool = False
    lidar_channels: int = _LIDAR_CHANNELS
    lidar_horizontal_rays: int = _LIDAR_H_RAYS
    lidar_max_distance_m: float = 4.0

    @property
    def lidar_scan_shape(self) -> tuple[int, int]:
        return (self.lidar_channels, self.lidar_horizontal_rays)

    @property
    def lidar_flat_dim(self) -> int:
        return self.lidar_channels * self.lidar_horizontal_rays
