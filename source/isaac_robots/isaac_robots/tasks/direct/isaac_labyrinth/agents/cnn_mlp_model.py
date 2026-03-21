"""Asymmetric actor-critic models for the Isaac Labyrinth task.

NatureCnnPolicy (actor):
    Receives proprio(12) + stacked grayscale frames (frame_stack × H × W).
    NatureCNN encoder → concat with proprio → MLP → Gaussian action distribution.

MlpCritic (critic):
    Receives privileged state: desired_pos_b(3), lin_vel_b(3), ang_vel_b(3),
    geodesic_distance(1), nearest_obstacle_dist(1) = 11D.
    Simple MLP → scalar value estimate.

The legacy CnnMlpSharedModel is kept at the bottom for loading old checkpoints.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

# ── Constants ────────────────────────────────────────────────────────────────
PROPRIO_DIM: int = 12  # lin_vel(3) + ang_vel(3) + gravity(3) + goal(3)
FRAME_STACK: int = 4
CAM_H: int = 64
CAM_W: int = 64


# ── Actor ────────────────────────────────────────────────────────────────────

class NatureCnnPolicy(GaussianMixin, Model):
    """NatureCNN actor for visual navigation with stacked frames.

    Observation layout (flat): [proprio(12) | frames(frame_stack*H*W)]
    """

    def __init__(
        self,
        observation_space=None,
        action_space=None,
        device=None,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        initial_log_std: float = 0.0,
        reduction: str = "sum",
        **kwargs,
    ) -> None:
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(
            self, clip_actions=clip_actions, clip_log_std=clip_log_std,
            min_log_std=min_log_std, max_log_std=max_log_std, reduction=reduction,
        )

        # NatureCNN: 3 conv layers → flatten → projection
        self.cnn = nn.Sequential(
            nn.Conv2d(FRAME_STACK, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute CNN output size dynamically (safe if CAM_H/CAM_W change)
        with torch.no_grad():
            dummy = torch.zeros(1, FRAME_STACK, CAM_H, CAM_W)
            cnn_out_dim = self.cnn(dummy).shape[1]
        self.img_proj = nn.Sequential(
            nn.Linear(cnn_out_dim, 512),
            nn.ReLU(),
        )

        # MLP trunk: concat(cnn_feat, proprio) → hidden → action head
        trunk_in = 512 + PROPRIO_DIM
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(128, self.num_actions)
        self.log_std_parameter = nn.Parameter(
            torch.full((self.num_actions,), initial_log_std)
        )

    def compute(self, inputs: dict, role: str):
        obs = inputs["observations"]
        proprio = obs[:, :PROPRIO_DIM]
        img = obs[:, PROPRIO_DIM:].view(-1, FRAME_STACK, CAM_H, CAM_W)
        img_feat = self.img_proj(self.cnn(img))
        features = self.trunk(torch.cat([img_feat, proprio], dim=-1))
        return self.action_head(features), {"log_std": self.log_std_parameter}


# ── Critic ───────────────────────────────────────────────────────────────────

class MlpCritic(DeterministicMixin, Model):
    """MLP critic operating on privileged state (11D)."""

    def __init__(
        self,
        observation_space=None,
        action_space=None,
        device=None,
        clip_actions: bool = False,
        **kwargs,
    ) -> None:
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        state_dim = observation_space.shape[0] if hasattr(observation_space, "shape") else observation_space
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
        )
        self.value_head = nn.Linear(256, 1)

    def compute(self, inputs: dict, role: str):
        features = self.net(inputs["states"])
        return self.value_head(features), {}


# ── Legacy model (for loading old checkpoints) ──────────────────────────────

# Old camera constants
_LEGACY_PROP_DIM: int = 18
_LEGACY_CAM_H: int = 244
_LEGACY_CAM_W: int = 324
_LEGACY_CAM_DIM: int = _LEGACY_CAM_H * _LEGACY_CAM_W


def _legacy_cnn_out_size(in_h: int, in_w: int) -> int:
    def conv(h, w, k, s): return (h - k) // s + 1, (w - k) // s + 1
    h, w = conv(in_h, in_w, 8, 4)
    h, w = conv(h,    w,    4, 2)
    h, w = conv(h,    w,    3, 1)
    return 64 * h * w


class CnnMlpSharedModel(GaussianMixin, DeterministicMixin, Model):
    """DEPRECATED: Legacy shared-trunk model for old 324x244 camera checkpoints."""

    def __init__(
        self,
        observation_space=None,
        action_space=None,
        device=None,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        initial_log_std: float = 0.0,
        reduction: str = "sum",
        **kwargs,
    ) -> None:
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(
            self, clip_actions=clip_actions, clip_log_std=clip_log_std,
            min_log_std=min_log_std, max_log_std=max_log_std, reduction=reduction,
        )
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self._has_camera: bool = (self.num_observations == _LEGACY_PROP_DIM + _LEGACY_CAM_DIM)

        if self._has_camera:
            cnn_flat = _legacy_cnn_out_size(_LEGACY_CAM_H, _LEGACY_CAM_W)
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
                nn.Flatten(),
            )
            self.img_proj = nn.Sequential(nn.Linear(cnn_flat, 256), nn.ReLU())
            trunk_in = _LEGACY_PROP_DIM + 256
        else:
            trunk_in = self.num_observations

        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
        )
        self.policy_head = nn.Linear(128, self.num_actions)
        self.value_head = nn.Linear(128, 1)
        self.log_std_parameter = nn.Parameter(
            torch.full((self.num_actions,), initial_log_std)
        )

    def _encode(self, states: torch.Tensor) -> torch.Tensor:
        if self._has_camera:
            prop = states[:, :_LEGACY_PROP_DIM]
            img = states[:, _LEGACY_PROP_DIM:].view(-1, 1, _LEGACY_CAM_H, _LEGACY_CAM_W)
            img_feat = self.img_proj(self.cnn(img))
            return self.trunk(torch.cat([prop, img_feat], dim=-1))
        return self.trunk(states)

    def act(self, inputs: dict, *, role: str = ""):
        if role == "value":
            return DeterministicMixin.act(self, inputs, role=role)
        return GaussianMixin.act(self, inputs, role=role)

    def compute(self, inputs: dict, role: str):
        obs = inputs["observations"] if role == "policy" else inputs["states"]
        features = self._encode(obs)
        if role == "policy":
            return self.policy_head(features), {"log_std": self.log_std_parameter}
        return self.value_head(features), {}
