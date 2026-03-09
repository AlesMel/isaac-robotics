from __future__ import annotations

from dataclasses import field
from typing import Dict

import numpy as np
from gymnasium import spaces
from isaaclab.utils import configclass

from .sensors import SensorSelectionCfg


@configclass
class ObservationTermSpec:
    key: str
    shape: tuple[int, ...]
    dtype: type[np.floating] = np.float32


@configclass
class CrazyflieObservationCfg:
    sensor_selection: SensorSelectionCfg = SensorSelectionCfg()
    base_state_shape: tuple[int, ...] = (17,)
    include_last_action: bool = True
    terms: Dict[str, ObservationTermSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.terms = {
            "state": ObservationTermSpec(key="state", shape=self.base_state_shape),
        }
        if self.sensor_selection.enable_lidar:
            self.terms["lidar"] = ObservationTermSpec(
                key="lidar",
                shape=self.sensor_selection.lidar_scan_shape,
            )
        if self.sensor_selection.enable_camera:
            if "rgb" in self.sensor_selection.camera_data_types:
                self.terms["camera_rgb"] = ObservationTermSpec(
                    key="camera_rgb",
                    shape=self.sensor_selection.camera_rgb_shape,
                )
            if "depth" in self.sensor_selection.camera_data_types:
                self.terms["camera_depth"] = ObservationTermSpec(
                    key="camera_depth",
                    shape=self.sensor_selection.camera_depth_shape,
                )

    def build_space(self) -> spaces.Dict:
        return spaces.Dict(
            {
                term.key: spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=term.shape,
                    dtype=term.dtype,
                )
                for term in self.terms.values()
            }
        )

    def flatdim(self) -> int:
        return int(sum(np.prod(term.shape, dtype=int) for term in self.terms.values()))
