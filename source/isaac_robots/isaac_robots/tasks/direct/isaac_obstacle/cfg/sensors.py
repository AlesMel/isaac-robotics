from __future__ import annotations

from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, patterns
from isaaclab.utils import configclass


@configclass
class SensorSelectionCfg:
    enable_lidar: bool = True
    enable_camera: bool = False
    lidar_debug_vis: bool = True
    lidar_channels: int = 16
    lidar_horizontal_rays: int = 180
    lidar_vertical_fov_deg: tuple[float, float] = (-15.0, 15.0)
    lidar_max_distance_m: float = 10.0
    camera_width: int = 84
    camera_height: int = 84
    camera_data_types: tuple[str, ...] = ("rgb", "depth")

    @property
    def lidar_scan_shape(self) -> tuple[int, int]:
        return (self.lidar_channels, self.lidar_horizontal_rays)

    @property
    def lidar_flat_dim(self) -> int:
        return self.lidar_channels * self.lidar_horizontal_rays


def build_lidar_cfg(sensor_cfg: SensorSelectionCfg) -> MultiMeshRayCasterCfg | None:
    if not sensor_cfg.enable_lidar:
        return None

    horizontal_resolution = 360.0 / max(1, sensor_cfg.lidar_horizontal_rays)

    return MultiMeshRayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/body",
        update_period=0.0,
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.05)),
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=sensor_cfg.lidar_channels,
            horizontal_res=horizontal_resolution,
            vertical_fov_range=list(sensor_cfg.lidar_vertical_fov_deg),
            horizontal_fov_range=[-180.0, 180.0],
        ),
        debug_vis=sensor_cfg.lidar_debug_vis,
        mesh_prim_paths=[
            "/World/ground",
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/envs/env_.*/Obstacle_.*",
                track_mesh_transforms=True,
            ),
        ],
        max_distance=sensor_cfg.lidar_max_distance_m,
    )
