"""Hand-E gripper actuation sanity check.

Holds the UR3e arm at its home pose and toggles the Hand-E close command in a
square wave so the slider positions can be observed directly. Designed to fail
loud when the sliders are not actually driven by PhysX (e.g. missing DriveAPI
on the Slider_* joints), instead of silently looking like "the policy just
hasn't learned to grasp yet".

Checks performed:
  1. After init, dumps PhysX DOF stiffness / damping / max-force for the
     finger sliders. If any are zero, the drive is not live and PhysX will
     ignore the position target.
  2. On every print, compares the *previous* slider position to the current
     one so a stuck slider is obvious.
  3. Flags whenever the env auto-resets, so a slider snapped back to the
     open target by reset() is not misread as "drive doesn't work".
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Hand-E gripper actuation sanity check.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Robots-UR3e-Lift-Cube-HandE-Direct-v0",
    help="Gym task id (must use the Hand-E gripper).",
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--period_steps",
    type=int,
    default=120,
    help="Policy steps per open/close half-cycle (~2 s at 60 Hz).",
)
parser.add_argument(
    "--print_every",
    type=int,
    default=10,
    help="Print joint trace every N policy steps.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import isaac_robots.tasks  # noqa: F401


def _fmt(t: torch.Tensor) -> str:
    return "  ".join(f"{v:+.4f}" for v in t.tolist())


def main() -> None:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)
    env.reset()

    inner = env.unwrapped
    robot = inner._robot
    n_envs = inner.num_envs
    device = inner.device

    slider_ids, slider_names = robot.find_joints(["Slider_.*"])
    if not slider_ids:
        raise RuntimeError(
            f"No Slider_* joints found. Available joints: {robot.joint_names}"
        )

    print()
    print("=" * 78)
    print("Hand-E actuation sanity check")
    print("=" * 78)
    print(f"task        : {args.task}")
    print(f"num_envs    : {n_envs}")
    print(f"action_dim  : {env.action_space.shape[0]}")
    print(f"sliders     : ids={slider_ids} names={slider_names}")

    # ---- 1. Actuator group bookkeeping (what Isaac Lab thinks it wrote) ----
    print()
    print("[init] actuator groups:")
    for name, actuator in robot.actuators.items():
        joint_names = [robot.joint_names[i] for i in actuator.joint_indices]
        print(
            f"   {name:>16}: joints={joint_names} "
            f"stiffness={float(actuator.stiffness[0, 0]):.1f} "
            f"damping={float(actuator.damping[0, 0]):.1f} "
            f"effort_limit={float(actuator.effort_limit[0, 0]):.3f}"
        )

    # ---- 2. What PhysX actually sees on the finger DOFs ----
    # If any of these read back as ~0, the joint has no live drive and
    # set_joint_position_target will be silently ignored regardless of what
    # the actuator config says.
    physx = robot.root_physx_view
    stiff_all = physx.get_dof_stiffnesses().to(device)
    damp_all = physx.get_dof_dampings().to(device)
    maxf_all = physx.get_dof_max_forces().to(device)
    print()
    print("[physx] live DOF gains on finger sliders (env 0):")
    for sid, sname in zip(slider_ids, slider_names):
        s = float(stiff_all[0, sid])
        d = float(damp_all[0, sid])
        m = float(maxf_all[0, sid])
        flag = "  <-- DRIVE NOT LIVE" if max(s, d, m) <= 1.0e-6 else ""
        print(f"   {sname}: stiffness={s:.1f}  damping={d:.1f}  max_force={m:.3f}{flag}")

    # ---- 3. Square-wave the gripper command ----
    actions = torch.zeros(n_envs, env.action_space.shape[0], device=device)
    prev_pos = robot.data.joint_pos[:, slider_ids].clone()
    prev_episode_len = inner.episode_length_buf.clone()

    print()
    print(f"[run] square wave: every {args.period_steps} policy steps; "
          f"printing trace every {args.print_every} steps; Ctrl-C to quit")
    print()

    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions.zero_()  # arm stays put -- IK delta = 0
            phase = (step // args.period_steps) % 2
            close = phase == 0
            actions[:, 6] = 1.0 if close else -1.0

            env.step(actions)

            if step % args.print_every == 0:
                cur_pos = robot.data.joint_pos[:, slider_ids]
                cur_vel = robot.data.joint_vel[:, slider_ids]
                cur_tgt = robot.data.joint_pos_target[:, slider_ids]
                delta = cur_pos - prev_pos

                # Detect an auto-reset in env 0 between prints: episode length
                # counter drops back near zero.
                reset_flag = ""
                ep_len = inner.episode_length_buf
                if ep_len[0].item() < prev_episode_len[0].item():
                    reset_flag = "  [env0 RESET]"
                prev_episode_len = ep_len.clone()

                state = "CLOSE" if close else "OPEN "
                print(
                    f"[t={step:>5}] cmd={state}  "
                    f"pos=[{_fmt(cur_pos[0])}]  "
                    f"vel=[{_fmt(cur_vel[0])}]  "
                    f"tgt=[{_fmt(cur_tgt[0])}]  "
                    f"d_pos=[{_fmt(delta[0])}]{reset_flag}"
                )
                prev_pos = cur_pos.clone()

            step += 1

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
