from __future__ import annotations

from typing import Dict

import torch
import numpy as np
from gymnasium import spaces
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from crazyflie_lab.config.observations import CrazyflieObservationCfg
from crazyflie_lab.config.scene import CrazyflieSceneCfg
from crazyflie_lab.config.sensors import SensorSelectionCfg


@configclass
class CrazyflieEnvCfg(DirectRLEnvCfg):
    episode_length_s: float = 10.0
    decimation: int = 2
    action_scale: float = 1.0
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=2)
    sensor_selection: SensorSelectionCfg = SensorSelectionCfg()
    scene: CrazyflieSceneCfg = CrazyflieSceneCfg(num_envs=64, env_spacing=4.0)
    observations: CrazyflieObservationCfg = CrazyflieObservationCfg()

    def __post_init__(self) -> None:
        self.sim.render_interval = self.decimation
        self.scene.configure(self.sensor_selection)
        self.observations.sensor_selection = self.sensor_selection
        self.observations.__post_init__()
        self.scene.__post_init__()
        self.observation_space = self.observations.build_space_spec()
        self.action_space = 4


class CrazyflieDirectEnv(DirectRLEnv):
    cfg: CrazyflieEnvCfg

    def __init__(self, cfg: CrazyflieEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        self.observation_space = cfg.observations.build_space()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    def _setup_scene(self) -> None:
        super()._setup_scene()
        self.robot = self.scene["robot"]

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        if hasattr(self, "robot"):
            self.robot.set_joint_effort_target(self.actions * self.cfg.action_scale)

    def _get_observations(self) -> Dict[str, Dict[str, torch.Tensor]]:
        obs: Dict[str, torch.Tensor] = {
            "state": self._get_state_obs(),
        }

        if self.cfg.sensor_selection.enable_lidar and getattr(self.scene, "lidar", None) is not None:
            obs["lidar"] = self.scene.lidar.data.ray_distances

        if self.cfg.sensor_selection.enable_camera and getattr(self.scene, "camera", None) is not None:
            if "rgb" in self.cfg.sensor_selection.camera_data_types:
                obs["camera_rgb"] = self.scene.camera.data.output["rgb"].permute(0, 3, 1, 2)
            if "depth" in self.cfg.sensor_selection.camera_data_types:
                depth = self.scene.camera.data.output["depth"]
                obs["camera_depth"] = depth.unsqueeze(1) if depth.ndim == 3 else depth

        return {"policy": obs}

    def _get_state_obs(self) -> torch.Tensor:
        root_state = self.robot.data.root_state_w
        lin_vel = self.robot.data.root_lin_vel_b
        ang_vel = self.robot.data.root_ang_vel_b
        last_action = getattr(self, "actions", torch.zeros(self.num_envs, 4, device=self.device))
        return torch.cat(
            [
                root_state[:, 0:3],
                root_state[:, 3:7],
                lin_vel,
                ang_vel,
                last_action,
            ],
            dim=-1,
        )

    def _get_rewards(self) -> torch.Tensor:
        height_error = torch.square(self.robot.data.root_pos_w[:, 2] - 1.0)
        actions = getattr(self, "actions", torch.zeros(self.num_envs, 4, device=self.device))
        action_penalty = 0.01 * torch.sum(torch.square(actions), dim=-1)
        return 1.0 - height_error - action_penalty

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = self.robot.data.root_pos_w[:, 2] < 0.1
        time_outs = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_outs

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        super()._reset_idx(env_ids)
        if env_ids is None:
            return
        default_actions = torch.zeros((env_ids.shape[0], 4), device=self.device)
        if hasattr(self, "actions"):
            self.actions[env_ids] = default_actions
