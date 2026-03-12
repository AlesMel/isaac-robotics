from __future__ import annotations

import math

from isaaclab.sensors import MultiMeshRayCasterCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass
from .tof_pattern import CrazyflieToFPatternCfg

LIDAR_CFG = MultiMeshRayCasterCfg(
    attach_yaw_only=True,
    pattern_cfg=patterns.LidarPatternCfg(
        channels=16, vertical_fov_range=(-15.0, 15.0), horizontal_fov_range=(-180.0, 180.0), horizontal_res=1.0
    ),
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
    mesh_prim_paths=[
        "/World/ground",
        MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="/World/envs/env_.*/Warehouse"),
    ],
    debug_vis=True,
    max_distance=100,
)

MULTI_RANGER_CFG = MultiMeshRayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/body",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.02)), # Mounted on top
    pattern_cfg=CrazyflieToFPatternCfg(),
    max_distance=4.0, # Multi-ranger limit is 4 meters
    mesh_prim_paths=[
        "/World/ground",
        MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="/World/envs/env_.*/Warehouse"),
    ],
    debug_vis=False,
)

def _compute_ray_count(cfg: patterns.PatternBaseCfg) -> tuple[int, int]:
    """Compute (channels, rays) gracefully for any pattern."""
    
    # 1. Handle standard spinning LiDARs
    if isinstance(cfg, patterns.LidarPatternCfg):
        h_fov = cfg.horizontal_fov_range
        h_rays = math.ceil((h_fov[1] - h_fov[0]) / cfg.horizontal_res) + 1
        if abs(abs(h_fov[0] - h_fov[1]) - 360.0) < 1e-6:
            h_rays -= 1
        return cfg.channels, h_rays
        
    # 2. Handle our custom Crazyflie ToF config
    elif isinstance(cfg, CrazyflieToFPatternCfg):
        # 1 "channel" (row), 6 total rays
        return 1, 6
        
    # 3. Fallback for other custom patterns
    else:
        # If it's a generic custom pattern, you might have to hardcode 
        # or evaluate the function to get the shape.
        starts, directions = cfg.func(cfg, device="cpu")
        return 1, directions.shape[0]

_LIDAR_CHANNELS, _LIDAR_H_RAYS = _compute_ray_count(MULTI_RANGER_CFG.pattern_cfg)

@configclass
class SensorSelectionCfg:
    enable_lidar: bool = True
    enable_camera: bool = False
    lidar_debug_vis: bool = True
    lidar_channels: int = _LIDAR_CHANNELS
    lidar_horizontal_rays: int = _LIDAR_H_RAYS
    lidar_vertical_fov_deg: tuple[float, float] = (-15.0, 15.0)
    lidar_max_distance_m: float = 4.0
    camera_width: int = 84
    camera_height: int = 84
    camera_data_types: tuple[str, ...] = ("rgb", "depth")

    @property
    def lidar_scan_shape(self) -> tuple[int, int]:
        return (self.lidar_channels, self.lidar_horizontal_rays)

    @property
    def lidar_flat_dim(self) -> int:
        return self.lidar_channels * self.lidar_horizontal_rays
