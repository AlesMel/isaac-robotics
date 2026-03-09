from __future__ import annotations

from typing import Optional

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass


@configclass
class SensorSelectionCfg:
    enable_lidar: bool = False
    enable_camera: bool = False
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
    def camera_rgb_shape(self) -> tuple[int, int, int]:
        return (3, self.camera_height, self.camera_width)

    @property
    def camera_depth_shape(self) -> tuple[int, int, int]:
        return (1, self.camera_height, self.camera_width)


def build_lidar_cfg(sensor_cfg: SensorSelectionCfg, prim_path: str) -> Optional[RayCasterCfg]:
    if not sensor_cfg.enable_lidar:
        return None

    horizontal_resolution = 360.0 / max(1, sensor_cfg.lidar_horizontal_rays)

    return RayCasterCfg(
        prim_path=prim_path,
        update_period=0.0,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.05)),
        attach_yaw_only=False,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=sensor_cfg.lidar_channels,
            horizontal_res=horizontal_resolution,
            vertical_fov_range=list(sensor_cfg.lidar_vertical_fov_deg),
            horizontal_fov_range=[-180.0, 180.0],
        ),
        debug_vis=False,
        mesh_prim_paths=["{ENV_REGEX_NS}/World"],
        max_distance=sensor_cfg.lidar_max_distance_m,
    )


def build_camera_cfg(sensor_cfg: SensorSelectionCfg, prim_path: str) -> Optional[CameraCfg]:
    if not sensor_cfg.enable_camera:
        return None

    return CameraCfg(
        prim_path=prim_path,
        update_period=0.0,
        height=sensor_cfg.camera_height,
        width=sensor_cfg.camera_width,
        data_types=list(sensor_cfg.camera_data_types),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 100.0),
        ),
    )
