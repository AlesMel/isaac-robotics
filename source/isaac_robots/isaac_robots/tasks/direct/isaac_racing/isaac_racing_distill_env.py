"""Racing distillation environment.

Identical to RacingDirectEnv except that, when a camera is active, it additionally
exposes normalized ToF readings in ``extras["tof_obs"]`` (shape: (N, lidar_flat_dim))
so the distillation training script can query the frozen ToF teacher on the same
observations the teacher was trained on.

The existing RacingDirectEnv is NOT modified — this subclass is registered under
a separate Gym ID (``Isaac-Robots-Racing-Distill-Direct-v0``).
"""

import torch

from .isaac_racing_env import RacingDirectEnv


class RacingDistillEnv(RacingDirectEnv):
    """Racing env variant that exposes ToF observations for teacher-student distillation.

    When camera mode is active:
    - extras["tof_obs"] is set every step to normalized ToF readings (N, lidar_flat_dim).
    - extras["tof_obs"] is also set on reset (all ones = max-distance) so the key always
      exists in the info dict from the very first env.step() call.

    In non-camera mode (flat MLP / ToF-only), no extras are added — the env behaves
    identically to the base class.
    """

    def _get_observations(self) -> dict:
        result = super()._get_observations()

        # Only expose ToF obs when camera mode is active (distillation use case).
        # _lidar_ranges_raw is populated by _get_rewards() (line 526) which runs
        # before _get_observations() is called.
        if self._camera is not None:
            if self._lidar_ranges_raw is not None:
                tof = self._lidar_ranges_raw.clone()
                if self.cfg.domain_rand.sensor_noise_std > 0:
                    tof = tof + torch.randn_like(tof) * self.cfg.domain_rand.sensor_noise_std
                tof = tof.div(self.cfg.sensor_selection.lidar_max_distance_m)
                tof.nan_to_num_(nan=1.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
                self.extras["tof_obs"] = tof.reshape(self.num_envs, -1)  # (N, 6)
            else:
                # First step before _get_rewards() has run — emit safe max-distance readings.
                self.extras["tof_obs"] = torch.ones(
                    self.num_envs,
                    self.cfg.sensor_selection.lidar_flat_dim,
                    device=self.device,
                )

        return result

    def _reset_idx(self, env_ids):
        result = super()._reset_idx(env_ids)

        # Ensure key always exists in info after a reset (camera mode only).
        # Only the subset env_ids is being reset; the full-env version in
        # _get_observations() handles the remaining envs on the next step.
        if self._camera is not None:
            self.extras["tof_obs"] = torch.ones(
                len(env_ids),
                self.cfg.sensor_selection.lidar_flat_dim,
                device=self.device,
            )

        return result
