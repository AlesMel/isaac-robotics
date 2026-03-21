"""Crazyflie AI bundle monocular camera sensor configuration.

Camera: Himax HM01B0 (monochrome variant, shipped with AI Deck 1.1)
  - Resolution : 324 x 244 px  (window mode)
  - Diagonal FOV: ~80° (AI-deck standard lens)
  - h_fov      : ~68°  (derived from 80° diagonal on 324x244 sensor)

Uses CameraCfg (not TiledCameraCfg) because TiledCameraCfg has a known bug
with OffsetCfg rotations.  See https://github.com/isaac-sim/IsaacLab/discussions/3819

Aperture / focal-length values are scaled to standard simulation range (×27.6
from the real sensor); the ratio (and therefore FOV) is preserved.
"""
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

CRAZYFLIE_AI_CAMERA_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/body/ai_camera",
    update_period=0.0,
    offset=CameraCfg.OffsetCfg(
        pos=(0.03, 0.0, 0.022),
        rot=(0.5, -0.5, 0.5, -0.5),
        convention="ros",
    ),
    data_types=["rgb"],
    width=324,
    height=244,
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=32.18,
        clipping_range=(0.05, 10.0),
    ),
)

CRAZYFLIE_AI_CAMERA_64_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/body/ai_camera",
    update_period=0.0,
    offset=CameraCfg.OffsetCfg(
        pos=(0.03, 0.0, 0.022),
        rot=(0.5, -0.5, 0.5, -0.5),
        convention="ros",
    ),
    data_types=["rgb"],
    width=64,
    height=64,
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=32.18,
        clipping_range=(0.05, 10.0),
    ),
)
