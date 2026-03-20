"""Crazyflie AI bundle monocular camera sensor configuration.

Camera: Himax HM01B0 (monochrome variant, shipped with AI Deck 1.1)
  - Resolution : 324 x 244 px  (window mode)
  - Pixel size : 3.6 µm x 3.6 µm
  - Sensor     : 1.1664 mm x 0.8784 mm
  - Diagonal FOV: ~80° (AI-deck standard lens)
  - h_fov      : ~68°  (derived from 80° diagonal on 324x244 sensor)

Derived pinhole parameters (consistent physical units in mm):
  focal_length      = sensor_diag / 2 / tan(diag_fov / 2)
                    = 1.460 / 2 / tan(40°) ≈ 0.870 mm
  horizontal_aperture = 324 * 3.6e-3 mm = 1.1664 mm
  vertical_aperture   = 244 * 3.6e-3 mm = 0.8784 mm
"""
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg

CRAZYFLIE_AI_CAMERA_CFG = TiledCameraCfg(
    # Prim path is a placeholder; callers must replace with the correct env regex.
    prim_path="{ENV_REGEX_NS}/Robot/body/ai_camera",
    offset=TiledCameraCfg.OffsetCfg(
        # Slightly in front of and above the drone body centre.
        # The AI deck sits on top of the Crazyflie with the camera facing forward.
        pos=(0.03, 0.0, 0.022),
        # Rotate so the camera's optical axis points along the drone's +X (forward).
        # In ROS convention: cam +Z (optical) → body +X, cam +Y (down) → body -Z.
        # Quaternion (w, x, y, z) = (0.5, -0.5, 0.5, -0.5).
        rot=(0.5, -0.5, 0.5, -0.5),
        convention="ros",
    ),
    data_types=["rgb"],
    width=324,
    height=244,
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=0.870,
        horizontal_aperture=1.1664,
        vertical_aperture=0.8784,
        clipping_range=(0.05, 10.0),
    ),
)

CRAZYFLIE_AI_CAMERA_64_CFG = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/body/ai_camera",
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.03, 0.0, 0.022),
        rot=(0.5, -0.5, 0.5, -0.5),
        convention="ros",
    ),
    data_types=["rgb"],
    width=64,
    height=64,
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=0.870,
        horizontal_aperture=1.1664,
        vertical_aperture=0.8784,
        clipping_range=(0.05, 10.0),
    ),
)
