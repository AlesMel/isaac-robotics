"""Sim-to-real bridge for the UR3e Reach task -- SKELETON, NOT YET WIRED.

This script is a starting point for running a policy trained in
``Isaac-Robots-UR3E-Reach-Direct-v0`` on the physical UR3e via the
``ur_rtde`` library. It is intentionally incomplete; fill in the marked
TODOs once the simulated policy trains.

Prereqs (TODO students):
    1. ``pip install ur-rtde``  (https://sdurobotics.gitlab.io/ur_rtde/)
    2. Set ``ROBOT_IP`` to the controller IP.
    3. Install the policy weights produced by:
           python scripts/skrl/train.py --task=Isaac-Robots-UR3E-Reach-Direct-v0
       and load them with the same skrl model class as in train.py.

Sim-to-real notes:
    * The current policy is trained on UR10-style absolute joint-position
      targets: ``q_desired = sim_home_q + action * 0.5``. On hardware, do not
      send that target directly at first; move in small guarded steps toward
      it, as implemented in ``safe_bringup.py policy-step``.
    * Observations on the real arm come from
      ``rtde_receive.getActualQ`` (joint pos), ``getActualQd`` (joint vel),
      and ``getActualTCPPose`` (EE pose). Make sure the order matches the
      simulator's joint ordering -- print both and align before deploying.
    * Start with an empty workspace and the e-stop within reach. The first
      command should be a slow move to the home pose, not a policy step.

Safety:
    * Always wrap policy execution in a try/except that calls
      ``rtde_control.servoStop()`` and re-raises.
    * Cap each hardware step to a tiny joint delta while validating the policy.
      The simulator's action scale is the full absolute target offset, not a
      safe real-robot step size.
"""

from __future__ import annotations

# --- intentionally commented out so this file imports cleanly without
# --- the ur_rtde / torch dependencies installed:
# import rtde_control
# import rtde_receive
# import torch

ROBOT_IP = "192.168.1.10"  # TODO: set to your controller IP
CONTROL_FREQUENCY_HZ = 60  # match the sim policy rate
POLICY_ACTION_SCALE = 0.5  # matches cfg.action_scale for absolute targets
HARDWARE_STEP_LIMIT = 0.002


def load_policy(checkpoint_path: str):
    """Load a trained skrl policy from a checkpoint .pt file.

    TODO: instantiate the same GaussianMixin model used in skrl_ppo_cfg.yaml,
    load the state dict, set eval mode, return the callable that maps obs ->
    action mean.
    """
    raise NotImplementedError


def read_robot_obs(rtde_r) -> "torch.Tensor":  # noqa: F821
    """Build the policy observation vector from the live robot state.

    Layout must match ``UR3eReachDirectEnv._get_observations``. The current
    sim-to-real default is the 21D position-only layout::

        [joint_pos(6), joint_vel(6), ee_pos_b(3),
         target_pos_b(3), target_pos_b - ee_pos_b(3)]

    Legacy checkpoints trained with ``include_ee_quat=True`` use the older
    25D layout::

        [joint_pos(6), joint_vel(6), ee_pos_b(3), ee_quat_b(4),
         target_pos_b(3), target_pos_b - ee_pos_b(3)]

    TODO: fetch ``getActualQ``, ``getActualQd``, ``getActualTCPPose``;
    convert TCP pose (axis-angle) to a base-frame quaternion; assemble.
    """
    raise NotImplementedError


def send_joint_target(rtde_c, current_q, action) -> None:
    """Move one guarded step toward the policy's absolute joint target.

    Args:
        rtde_c: ``rtde_control.RTDEControlInterface`` instance.
        current_q: current joint positions (length 6).
        action: policy output for the arm slice, length 6, already in [-1, 1].
    """
    raise NotImplementedError


def main(checkpoint_path: str) -> None:
    """Top-level deploy loop.

    1. Connect to the robot.
    2. Move to the home pose used in sim (matches UR3E_CFG.init_state).
    3. Loop: obs <- read_robot_obs -> action <- policy(obs) -> servoJ.
    4. On Ctrl+C or exception, call rtde_control.servoStop and disconnect.
    """
    raise NotImplementedError


if __name__ == "__main__":
    raise NotImplementedError(
        "This script is a skeleton. Fill in the TODOs in load_policy, "
        "read_robot_obs, send_joint_target, and main before running."
    )
