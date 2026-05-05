"""Sensor configs for the UR3e Reach task.

Reach is proprioception-only -- this file is a placeholder so future variants
(camera-based servoing, force sensing) follow the same layout as the other
tasks in this extension.
"""

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class SensorSelectionCfg:
    """No sensors enabled by default for Reach."""

    enable_wrist_camera: bool = False
    enable_force_sensor: bool = False
