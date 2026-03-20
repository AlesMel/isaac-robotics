"""Shared CNN+MLP actor-critic model for SKRL PPO.

Designed for the Isaac Labyrinth task with an optional Crazyflie AI-bundle
HM01B0 monochrome camera (324 x 244 px, grayscale).

Observation layout produced by LabyrinthDirectEnv:
  obs[:, :PROP_DIM]  → proprioception
                        (lin_vel_b=3, ang_vel_b=3, gravity_b=3, goal_b=3, lidar=6 → 18 total)
  obs[:, PROP_DIM:]  → flattened grayscale camera image, normalised to [0, 1]
                        (324 * 244 = 79 056 values when camera is enabled)

The model auto-detects whether camera observations are present by checking
the total observation dimension.  When no camera is available it falls back
to a plain MLP, so the YAML can stay the same regardless of whether the
camera sensor is enabled.

Architecture (camera present):
    ┌─ Camera (N,1,244,324) ──► CNN ──► Linear(256) ──► ReLU ─┐
    │                                                           ├─ cat → Trunk → policy head (actions)
    └─ Proprioception (N,18) ─────────────────────────────────┘          └─► value head (1)

Architecture (camera absent):
    Proprioception (N,18) ──► Trunk → policy head
                                   └─► value head
"""
from __future__ import annotations

import torch
import torch.nn as nn

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

# ── Constants ──────────────────────────────────────────────────────────────────
# Proprioception: lin_vel(3) + ang_vel(3) + gravity(3) + goal(3) + lidar(6)
PROP_DIM: int = 18
# Camera: HM01B0 window mode
CAM_H: int = 244
CAM_W: int = 324
CAM_DIM: int = CAM_H * CAM_W  # 79 056


def _cnn_out_size(in_h: int, in_w: int) -> int:
    """Compute flattened CNN output size without allocating GPU tensors."""
    def conv(h, w, k, s): return (h - k) // s + 1, (w - k) // s + 1
    h, w = conv(in_h, in_w, 8, 4)   # → 60, 80
    h, w = conv(h,    w,    4, 2)   # → 29, 39
    h, w = conv(h,    w,    3, 1)   # → 27, 37
    return 64 * h * w               # = 63 936


class CnnMlpSharedModel(GaussianMixin, DeterministicMixin, Model):
    """Shared trunk actor-critic for SKRL PPO with optional camera input.

    Registered in ``skrl_ppo_cfg.yaml`` via full module path so that SKRL's
    Runner can instantiate it.  Both the policy and value heads share the same
    CNN encoder and MLP trunk (``separate: False`` in the YAML).
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        initial_log_std: float = 0.0,
        reduction: str = "sum",
        **kwargs,
    ) -> None:
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction,
        )
        DeterministicMixin.__init__(self, clip_actions)

        self._has_camera: bool = (self.num_observations == PROP_DIM + CAM_DIM)

        if self._has_camera:
            cnn_flat = _cnn_out_size(CAM_H, CAM_W)  # 63 936
            self.cnn = nn.Sequential(
                # (N, 1, 244, 324) → (N, 32, 60, 80)
                nn.Conv2d(1, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                # → (N, 64, 29, 39)
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                # → (N, 64, 27, 37)
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            self.img_proj = nn.Sequential(
                nn.Linear(cnn_flat, 256),
                nn.ReLU(),
            )
            trunk_in = PROP_DIM + 256
        else:
            trunk_in = self.num_observations

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
        )

        # Separate heads
        self.policy_head = nn.Linear(128, self.num_actions)
        self.value_head = nn.Linear(128, 1)

        # Trainable log-std for the Gaussian policy
        self.log_std_parameter = nn.Parameter(
            torch.full((self.num_actions,), initial_log_std)
        )

    # ── forward ───────────────────────────────────────────────────────────────

    def _encode(self, states: torch.Tensor) -> torch.Tensor:
        """Encode raw observation into shared feature vector."""
        if self._has_camera:
            prop = states[:, :PROP_DIM]
            img = states[:, PROP_DIM:].view(-1, 1, CAM_H, CAM_W)
            img_feat = self.img_proj(self.cnn(img))
            return self.trunk(torch.cat([prop, img_feat], dim=-1))
        return self.trunk(states)

    def act(self, inputs: dict, role: str = ""):
        """Route to the correct mixin based on role.

        GaussianMixin.act() expects compute() to return (mean, log_std, extras).
        DeterministicMixin.act() expects compute() to return (value, extras).
        Without this override Python always picks GaussianMixin.act() via MRO,
        which breaks the value role.
        """
        if role == "value":
            return DeterministicMixin.act(self, inputs, role)
        return GaussianMixin.act(self, inputs, role)

    def compute(self, inputs: dict, role: str):
        features = self._encode(inputs["states"])
        if role == "policy":
            return self.policy_head(features), self.log_std_parameter, {}
        # role == "value"
        return self.value_head(features), {}
