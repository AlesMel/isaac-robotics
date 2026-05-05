"""Base interface for swappable end-effector grippers.

Tasks that want a gripper should declare ``gripper: GripperCfg`` on their env
config and instantiate it inside the env via ``cfg.gripper.class_type(cfg, env)``.
The env then forwards a slice of the action vector to ``gripper.apply_action``
and concatenates ``gripper.get_observation()`` onto its observation tensor.

Concrete implementations (NoGripper, SuctionGripper, ...) live in sibling files.

Design notes for students:
  * This is duck-typed on purpose -- no abstract base class. The contract is
    just the methods listed below. Subclassing GripperBase is encouraged but
    not enforced.
  * ``action_dim`` and ``obs_dim`` come from the config so the env can size its
    action_space / observation_space without instantiating the gripper.
  * Keeping the gripper as a separate object (instead of mixing its logic into
    the env) means swapping grippers is a one-line change in the env config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import DirectRLEnv


@configclass
class GripperCfg:
    """Base config. Subclass this for each concrete gripper type."""

    class_type: type = None
    """Concrete GripperBase subclass. Filled in by each gripper's own Cfg."""

    action_dim: int = 0
    """Number of action values this gripper consumes per step."""

    obs_dim: int = 0
    """Number of observation values this gripper appends to the policy obs."""


class GripperBase:
    """Pluggable gripper interface.

    Subclasses own any extra scene state (rigid objects, attached children,
    visual markers) and expose the four-method lifecycle below to the env.
    """

    cfg: GripperCfg

    def __init__(self, cfg: GripperCfg, env: "DirectRLEnv") -> None:
        self.cfg = cfg
        self._env = env
        self._device = env.device
        self._num_envs = env.num_envs

    @property
    def action_dim(self) -> int:
        return self.cfg.action_dim

    @property
    def obs_dim(self) -> int:
        return self.cfg.obs_dim

    def setup_scene(self) -> None:
        """Called once from the env's ``_setup_scene``.

        Use this to spawn extra prims (gripper meshes, suction-cup markers,
        graspable objects) and register them with the InteractiveScene.
        """

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset gripper state for the given environments.

        Called from the env's ``_reset_idx``. The default is a no-op.
        """

    def apply_action(self, action: torch.Tensor) -> None:
        """Apply the gripper command for one physics step.

        Args:
            action: tensor of shape ``(num_envs, action_dim)`` with values
                already clipped to ``[-1, 1]`` by the env.
        """

    def get_observation(self) -> torch.Tensor:
        """Return the gripper's contribution to the policy observation.

        Returns:
            Tensor of shape ``(num_envs, obs_dim)``.
        """
        return torch.zeros(self._num_envs, self.obs_dim, device=self._device)
