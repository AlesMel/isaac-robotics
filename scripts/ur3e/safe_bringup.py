"""Conservative UR3e real-hardware bring-up helpers.

The default commands are read-only. Commands that can affect the arm require
both ``--execute`` and ``--accept-risk`` so a typo cannot start motion.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import rtde_control
    import rtde_receive
except ImportError as exc:  # pragma: no cover - exercised on machines without ur-rtde
    raise SystemExit(
        "Missing ur-rtde. Install it with:\n"
        "  python -m pip install -r requirements-real-ur3e.txt"
    ) from exc


JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

SIM_HOME_Q = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0], dtype=np.float64)
TRAIN_TARGET_X_RANGE = (-0.36, -0.24)
TRAIN_TARGET_Y_RANGE = (-0.20, -0.06)
TRAIN_TARGET_Z_RANGE = (0.12, 0.24)
OBS_SEGMENTS_WITH_QUAT = (
    ("joint_pos", 0, 6),
    ("joint_vel", 6, 12),
    ("ee_pos_b", 12, 15),
    ("ee_quat_b", 15, 19),
    ("target_pos_b", 19, 22),
    ("target_error", 22, 25),
)
OBS_SEGMENTS_POSITION_ONLY = (
    ("joint_pos", 0, 6),
    ("joint_vel", 6, 12),
    ("ee_pos_b", 12, 15),
    ("target_pos_b", 15, 18),
    ("target_error", 18, 21),
)

ROBOT_MODE_NAMES = {
    -1: "NO_CONTROLLER",
    0: "DISCONNECTED",
    1: "CONFIRM_SAFETY",
    2: "BOOTING",
    3: "POWER_OFF",
    4: "POWER_ON",
    5: "IDLE",
    6: "BACKDRIVE",
    7: "RUNNING",
    8: "UPDATING_FIRMWARE",
}

SAFETY_MODE_NAMES = {
    1: "NORMAL",
    2: "REDUCED",
    3: "PROTECTIVE_STOP",
    4: "RECOVERY",
    5: "SAFEGUARD_STOP",
    6: "SYSTEM_EMERGENCY_STOP",
    7: "ROBOT_EMERGENCY_STOP",
    8: "VIOLATION",
    9: "FAULT",
    10: "VALIDATE_JOINT_ID",
    11: "UNDEFINED",
    12: "AUTOMATIC_MODE_SAFEGUARD_STOP",
    13: "THREE_POSITION_ENABLING_STOP",
}


@dataclass
class RobotState:
    q: np.ndarray
    qd: np.ndarray
    tcp_pose: np.ndarray
    tcp_speed: np.ndarray
    robot_mode: int | None
    safety_mode: int | None
    speed_scaling: float | None
    speed_scaling_combined: float | None
    protective_stopped: bool | None
    emergency_stopped: bool | None


@dataclass
class PolicyActionPlan:
    clipped_action: np.ndarray
    desired_q: np.ndarray
    guarded_delta_q: np.ndarray
    target_q: np.ndarray


def _safe_call(obj: object, method_name: str):
    method = getattr(obj, method_name, None)
    if method is None:
        return None
    try:
        return method()
    except Exception:
        return None


def fmt_vec(values: Sequence[float], precision: int = 4) -> str:
    return "[" + ", ".join(f"{float(v): .{precision}f}" for v in values) + "]"


def mode_name(value: int | None, names: dict[int, str]) -> str:
    if value is None:
        return "unknown"
    return f"{names.get(value, 'UNKNOWN')}({value})"


def connect_receive(robot_ip: str):
    print(f"[connect] RTDE receive -> {robot_ip}")
    return rtde_receive.RTDEReceiveInterface(robot_ip)


def connect_control(robot_ip: str):
    print(f"[connect] RTDE control -> {robot_ip}")
    return rtde_control.RTDEControlInterface(robot_ip)


def read_state(rtde_r) -> RobotState:
    return RobotState(
        q=np.asarray(rtde_r.getActualQ(), dtype=np.float64),
        qd=np.asarray(rtde_r.getActualQd(), dtype=np.float64),
        tcp_pose=np.asarray(rtde_r.getActualTCPPose(), dtype=np.float64),
        tcp_speed=np.asarray(_safe_call(rtde_r, "getActualTCPSpeed") or [math.nan] * 6, dtype=np.float64),
        robot_mode=_safe_call(rtde_r, "getRobotMode"),
        safety_mode=_safe_call(rtde_r, "getSafetyMode"),
        speed_scaling=_safe_call(rtde_r, "getSpeedScaling"),
        speed_scaling_combined=_safe_call(rtde_r, "getSpeedScalingCombined"),
        protective_stopped=_safe_call(rtde_r, "isProtectiveStopped"),
        emergency_stopped=_safe_call(rtde_r, "isEmergencyStopped"),
    )


def print_state(state: RobotState, *, prefix: str = "[state]") -> None:
    tcp_linear_speed = float(np.linalg.norm(state.tcp_speed[:3]))
    print(f"{prefix} robot_mode={mode_name(state.robot_mode, ROBOT_MODE_NAMES)}")
    print(f"{prefix} safety_mode={mode_name(state.safety_mode, SAFETY_MODE_NAMES)}")
    print(
        f"{prefix} protective_stop={state.protective_stopped} "
        f"emergency_stop={state.emergency_stopped}"
    )
    print(
        f"{prefix} speed_scaling={state.speed_scaling} "
        f"combined={state.speed_scaling_combined}"
    )
    print(f"{prefix} q={fmt_vec(state.q)}")
    print(f"{prefix} qd={fmt_vec(state.qd)}")
    print(f"{prefix} tcp_pose_xyz_rotvec={fmt_vec(state.tcp_pose)}")
    print(f"{prefix} tcp_linear_speed_m_s={tcp_linear_speed:.5f}")
    print(f"{prefix} sim_home_delta={fmt_vec(state.q - SIM_HOME_Q)}")


def assert_stationary_and_safe(
    state: RobotState,
    *,
    allow_reduced_mode: bool,
    max_start_joint_speed: float,
    max_start_tcp_speed: float,
) -> None:
    if state.emergency_stopped:
        raise RuntimeError("Robot reports emergency stop. Refusing control.")
    if state.protective_stopped:
        raise RuntimeError("Robot reports protective stop. Refusing control.")

    allowed_safety_modes = {1}
    if allow_reduced_mode:
        allowed_safety_modes.add(2)
    if state.safety_mode not in allowed_safety_modes:
        raise RuntimeError(
            "Robot safety mode is not allowed for this script: "
            f"{mode_name(state.safety_mode, SAFETY_MODE_NAMES)}"
        )

    max_joint_speed = float(np.max(np.abs(state.qd)))
    if max_joint_speed > max_start_joint_speed:
        raise RuntimeError(
            f"Robot joints are already moving ({max_joint_speed:.4f} rad/s). Refusing control."
        )

    tcp_linear_speed = float(np.linalg.norm(state.tcp_speed[:3]))
    if tcp_linear_speed > max_start_tcp_speed:
        raise RuntimeError(
            f"TCP is already moving ({tcp_linear_speed:.4f} m/s). Refusing control."
        )


def require_explicit_execution(args: argparse.Namespace) -> None:
    if not args.execute or not args.accept_risk:
        raise SystemExit(
            "Control command not sent. Re-run with both --execute and --accept-risk "
            "after confirming the workspace is clear and the e-stop is reachable."
        )


def assert_small_delta(delta: np.ndarray, max_abs_delta: float) -> None:
    largest = float(np.max(np.abs(delta)))
    if largest > max_abs_delta:
        raise RuntimeError(
            f"Requested joint delta {largest:.5f} rad exceeds limit {max_abs_delta:.5f} rad."
        )


def assert_robot_accepts_target(rtde_c, target_q: np.ndarray) -> None:
    accepted = rtde_c.isJointsWithinSafetyLimits(target_q.tolist())
    if not accepted:
        raise RuntimeError(f"Robot rejected target as outside safety limits: {fmt_vec(target_q)}")


def cleanup_control(rtde_c, *, servo: bool) -> None:
    if rtde_c is None:
        return
    if servo:
        try:
            rtde_c.servoStop()
        except Exception:
            pass
    try:
        rtde_c.stopScript()
    except Exception:
        pass
    try:
        rtde_c.disconnect()
    except Exception:
        pass


def rotvec_to_quat_wxyz(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    axis = rotvec / angle
    half = 0.5 * angle
    return np.array(
        [math.cos(half), *(axis * math.sin(half))],
        dtype=np.float32,
    )


def canonicalize_quat_wxyz(quat: np.ndarray, mode: str) -> np.ndarray:
    """Return a deterministic representative of a quaternion sign pair.

    ``q`` and ``-q`` encode the same orientation, but a neural net sees them
    as different observations. ``positive-max`` is useful near 180 degree
    rotations where ``w`` is close to zero and the usual ``w >= 0`` rule does
    not prevent vector-part sign flips.
    """
    quat = np.asarray(quat, dtype=np.float32).copy()
    if mode == "raw":
        return quat
    if mode == "negated":
        return -quat
    if mode == "positive-w":
        return -quat if quat[0] < 0.0 else quat
    if mode == "positive-max":
        idx = int(np.argmax(np.abs(quat)))
        return -quat if quat[idx] < 0.0 else quat
    if mode == "negative-max":
        idx = int(np.argmax(np.abs(quat)))
        return -quat if quat[idx] > 0.0 else quat
    raise ValueError(f"Unknown quaternion sign mode: {mode}")


def build_reach_observation(
    state: RobotState,
    target_pos_b: np.ndarray,
    *,
    include_quat: bool = True,
    quat_sign: str = "raw",
) -> np.ndarray:
    ee_pos_b = state.tcp_pose[:3].astype(np.float32)
    ee_quat_b = canonicalize_quat_wxyz(rotvec_to_quat_wxyz(state.tcp_pose[3:]), quat_sign)
    target = target_pos_b.astype(np.float32)
    obs_parts = [
        state.q.astype(np.float32),
        state.qd.astype(np.float32),
        ee_pos_b,
    ]
    if include_quat:
        obs_parts.append(ee_quat_b)
    obs_parts.extend((target, target - ee_pos_b))
    obs = np.concatenate(obs_parts)
    expected_shape = (25,) if include_quat else (21,)
    if obs.shape != expected_shape:
        raise RuntimeError(
            f"Internal observation shape error: expected {expected_shape}, got {obs.shape}"
        )
    return obs


def load_reach_policy(checkpoint_path: str):
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Policy preview requires torch in this Python environment.") from exc

    class ReachPolicy(nn.Module):
        def __init__(self, input_dim: int) -> None:
            super().__init__()
            self.log_std_parameter = nn.Parameter(torch.zeros(6))
            self.net_container = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ELU(),
                nn.Linear(128, 128),
                nn.ELU(),
            )
            self.policy_layer = nn.Linear(128, 6)
            self.value_layer = nn.Linear(128, 1)

        def forward(self, obs):
            return self.policy_layer(self.net_container(obs))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "policy" not in checkpoint or "state_preprocessor" not in checkpoint:
        raise RuntimeError(
            "Expected a skrl checkpoint with 'policy' and 'state_preprocessor' entries."
        )
    input_dim = int(checkpoint["policy"]["net_container.0.weight"].shape[1])
    if input_dim not in (21, 25):
        raise RuntimeError(f"Unsupported UR3e reach policy input dimension: {input_dim}")

    model = ReachPolicy(input_dim)
    model.load_state_dict(checkpoint["policy"], strict=True)
    model.eval()

    preprocessor = checkpoint["state_preprocessor"]
    mean = preprocessor["running_mean"].detach().cpu().float()
    variance = preprocessor["running_variance"].detach().cpu().float().clamp_min(1e-8)
    std = torch.sqrt(variance)
    mean_np = mean.detach().cpu().numpy().astype(np.float64)
    std_np = std.detach().cpu().numpy().astype(np.float64)

    def normalize(obs_np: np.ndarray) -> np.ndarray:
        return (obs_np.astype(np.float64) - mean_np) / std_np

    def act(obs_np: np.ndarray) -> np.ndarray:
        obs = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
        obs = (obs - mean) / std
        with torch.inference_mode():
            action = model(obs)[0]
        return action.detach().cpu().numpy().astype(np.float64)

    act.normalize = normalize  # type: ignore[attr-defined]
    act.mean = mean_np  # type: ignore[attr-defined]
    act.std = std_np  # type: ignore[attr-defined]
    act.input_dim = input_dim  # type: ignore[attr-defined]
    act.include_quat = input_dim == 25  # type: ignore[attr-defined]
    return act


def plan_policy_action(
    raw_action: np.ndarray,
    current_q: np.ndarray,
    *,
    policy_action_mode: str,
    max_policy_action_abs: float,
    hardware_action_scale: float,
    policy_action_scale: float,
    max_filtered_delta_rad: float,
) -> PolicyActionPlan:
    if raw_action.shape != (6,):
        raise RuntimeError(f"Expected 6D action, got shape {raw_action.shape}")
    if not np.all(np.isfinite(raw_action)):
        raise RuntimeError(f"Policy produced non-finite action: {raw_action}")
    clipped_action = np.clip(raw_action, -max_policy_action_abs, max_policy_action_abs)
    if policy_action_mode == "delta":
        desired_q = current_q + clipped_action * hardware_action_scale
    elif policy_action_mode == "absolute":
        desired_q = SIM_HOME_Q + clipped_action * policy_action_scale
    else:
        raise RuntimeError(f"Unsupported policy action mode: {policy_action_mode}")

    desired_delta_q = desired_q - current_q
    guarded_delta_q = np.clip(
        desired_delta_q,
        -max_filtered_delta_rad,
        max_filtered_delta_rad,
    )
    assert_small_delta(guarded_delta_q, max_filtered_delta_rad)
    return PolicyActionPlan(
        clipped_action=clipped_action,
        desired_q=desired_q,
        guarded_delta_q=guarded_delta_q,
        target_q=current_q + guarded_delta_q,
    )


def resolve_target(args: argparse.Namespace, state: RobotState) -> np.ndarray:
    if args.target is not None and args.target_offset is not None:
        raise SystemExit("Use only one of --target or --target-offset.")
    if args.target_offset is not None:
        offset = np.asarray(args.target_offset, dtype=np.float64)
        offset_norm = float(np.linalg.norm(offset))
        if offset_norm > args.max_target_offset_m:
            raise SystemExit(
                f"Target offset norm {offset_norm:.4f}m exceeds "
                f"--max-target-offset-m {args.max_target_offset_m:.4f}m."
            )
        return state.tcp_pose[:3].astype(np.float64) + offset
    if args.target is None:
        return state.tcp_pose[:3].astype(np.float64)
    return np.asarray(args.target, dtype=np.float64)


def run_status(args: argparse.Namespace) -> None:
    rtde_r = connect_receive(args.robot_ip)
    try:
        print_state(read_state(rtde_r))
    finally:
        rtde_r.disconnect()


def run_watch(args: argparse.Namespace) -> None:
    rtde_r = connect_receive(args.robot_ip)
    try:
        for i in range(args.samples):
            print_state(read_state(rtde_r), prefix=f"[sample {i + 1:03d}]")
            if i + 1 < args.samples:
                time.sleep(args.interval)
    finally:
        rtde_r.disconnect()


def run_hold(args: argparse.Namespace) -> None:
    rtde_r = connect_receive(args.robot_ip)
    rtde_c = None
    try:
        state = read_state(rtde_r)
        print_state(state)
        print(f"[plan] hold current q for {args.seconds:.2f}s at {args.hz:.1f}Hz")
        require_explicit_execution(args)
        assert_stationary_and_safe(
            state,
            allow_reduced_mode=args.allow_reduced_mode,
            max_start_joint_speed=args.max_start_joint_speed,
            max_start_tcp_speed=args.max_start_tcp_speed,
        )
        rtde_c = connect_control(args.robot_ip)
        assert_robot_accepts_target(rtde_c, state.q)

        dt = 1.0 / args.hz
        steps = max(1, int(round(args.seconds * args.hz)))
        target_q = state.q.copy()
        for _ in range(steps):
            t_start = rtde_c.initPeriod()
            latest = read_state(rtde_r)
            assert_small_delta(latest.q - target_q, args.max_hold_error_rad)
            rtde_c.servoJ(target_q.tolist(), 0.0, 0.0, dt, args.lookahead_time, args.gain)
            rtde_c.waitPeriod(t_start)
        print("[done] hold complete; stopping servo")
    finally:
        cleanup_control(rtde_c, servo=True)
        rtde_r.disconnect()


def run_tiny_movej(args: argparse.Namespace) -> None:
    rtde_r = connect_receive(args.robot_ip)
    rtde_c = None
    try:
        state = read_state(rtde_r)
        delta = np.zeros(6, dtype=np.float64)
        delta[args.joint_index] = args.delta_rad
        assert_small_delta(delta, args.max_delta_rad)
        target_q = state.q + delta
        print_state(state)
        print(
            "[plan] moveJ "
            f"{JOINT_NAMES[args.joint_index]} by {args.delta_rad:.5f} rad "
            f"at speed={args.speed:.4f} rad/s accel={args.accel:.4f} rad/s^2"
        )
        print(f"[plan] target_q={fmt_vec(target_q)}")
        require_explicit_execution(args)
        assert_stationary_and_safe(
            state,
            allow_reduced_mode=args.allow_reduced_mode,
            max_start_joint_speed=args.max_start_joint_speed,
            max_start_tcp_speed=args.max_start_tcp_speed,
        )
        rtde_c = connect_control(args.robot_ip)
        assert_robot_accepts_target(rtde_c, target_q)
        ok = rtde_c.moveJ(target_q.tolist(), args.speed, args.accel)
        if ok is False:
            raise RuntimeError("moveJ returned False")
        print("[done] tiny moveJ complete")
    except Exception:
        if rtde_c is not None:
            try:
                rtde_c.stopJ(args.accel)
            except Exception:
                pass
        raise
    finally:
        cleanup_control(rtde_c, servo=False)
        rtde_r.disconnect()


def run_tiny_servoj(args: argparse.Namespace) -> None:
    rtde_r = connect_receive(args.robot_ip)
    rtde_c = None
    try:
        state = read_state(rtde_r)
        delta = np.zeros(6, dtype=np.float64)
        delta[args.joint_index] = args.delta_rad
        assert_small_delta(delta, args.max_delta_rad)
        final_q = state.q + delta
        print_state(state)
        print(
            "[plan] servoJ ramp "
            f"{JOINT_NAMES[args.joint_index]} by {args.delta_rad:.5f} rad "
            f"over {args.seconds:.2f}s at {args.hz:.1f}Hz"
        )
        print(f"[plan] final_q={fmt_vec(final_q)}")
        require_explicit_execution(args)
        assert_stationary_and_safe(
            state,
            allow_reduced_mode=args.allow_reduced_mode,
            max_start_joint_speed=args.max_start_joint_speed,
            max_start_tcp_speed=args.max_start_tcp_speed,
        )
        rtde_c = connect_control(args.robot_ip)
        assert_robot_accepts_target(rtde_c, final_q)

        dt = 1.0 / args.hz
        steps = max(2, int(round(args.seconds * args.hz)))
        for i in range(steps):
            alpha = (i + 1) / steps
            target_q = state.q + alpha * delta
            t_start = rtde_c.initPeriod()
            rtde_c.servoJ(target_q.tolist(), 0.0, 0.0, dt, args.lookahead_time, args.gain)
            rtde_c.waitPeriod(t_start)
        print("[done] tiny servoJ ramp complete; stopping servo")
    finally:
        cleanup_control(rtde_c, servo=True)
        rtde_r.disconnect()


def run_policy_preview(args: argparse.Namespace) -> None:
    rtde_r = connect_receive(args.robot_ip)
    try:
        state = read_state(rtde_r)
        target = resolve_target(args, state)
        policy = load_reach_policy(args.checkpoint)
        obs = build_reach_observation(
            state,
            target,
            include_quat=policy.include_quat,  # type: ignore[attr-defined]
            quat_sign=args.quat_sign,
        )
        raw_action = policy(obs)
        action_plan = plan_policy_action(
            raw_action,
            state.q,
            policy_action_mode=args.policy_action_mode,
            max_policy_action_abs=args.max_policy_action_abs,
            hardware_action_scale=args.hardware_action_scale,
            policy_action_scale=args.policy_action_scale,
            max_filtered_delta_rad=args.max_filtered_delta_rad,
        )
        print_state(state)
        print(f"[policy] checkpoint={Path(args.checkpoint).resolve()}")
        print(f"[policy] input_dim={policy.input_dim}")  # type: ignore[attr-defined]
        print(f"[policy] quat_sign={args.quat_sign}")
        print(f"[policy] action_mode={args.policy_action_mode}")
        print(f"[policy] target_pos_b={fmt_vec(target)}")
        print(f"[policy] raw_action={fmt_vec(raw_action)}")
        print(f"[policy] clipped_action={fmt_vec(action_plan.clipped_action)}")
        print(f"[policy] desired_policy_q={fmt_vec(action_plan.desired_q)}")
        print(f"[policy] guarded_delta_q={fmt_vec(action_plan.guarded_delta_q, precision=6)}")
        print(f"[policy] one_step_target_q={fmt_vec(action_plan.target_q)}")
        print("[policy] no control command sent")
    finally:
        rtde_r.disconnect()


def run_policy_step(args: argparse.Namespace) -> None:
    rtde_r = connect_receive(args.robot_ip)
    rtde_c = None
    try:
        state = read_state(rtde_r)
        target = resolve_target(args, state)
        policy = load_reach_policy(args.checkpoint)
        obs = build_reach_observation(
            state,
            target,
            include_quat=policy.include_quat,  # type: ignore[attr-defined]
            quat_sign=args.quat_sign,
        )
        raw_action = policy(obs)
        action_plan = plan_policy_action(
            raw_action,
            state.q,
            policy_action_mode=args.policy_action_mode,
            max_policy_action_abs=args.max_policy_action_abs,
            hardware_action_scale=args.hardware_action_scale,
            policy_action_scale=args.policy_action_scale,
            max_filtered_delta_rad=args.max_filtered_delta_rad,
        )
        print_state(state)
        print(f"[policy] input_dim={policy.input_dim}")  # type: ignore[attr-defined]
        print(f"[policy] quat_sign={args.quat_sign}")
        print(f"[policy] action_mode={args.policy_action_mode}")
        print(f"[policy] target_pos_b={fmt_vec(target)}")
        print(f"[policy] raw_action={fmt_vec(raw_action)}")
        print(f"[policy] clipped_action={fmt_vec(action_plan.clipped_action)}")
        print(f"[policy] desired_policy_q={fmt_vec(action_plan.desired_q)}")
        print(f"[policy] guarded_delta_q={fmt_vec(action_plan.guarded_delta_q, precision=6)}")
        print(
            "[plan] one guarded policy step via moveJ "
            f"speed={args.speed:.4f} rad/s accel={args.accel:.4f} rad/s^2"
        )
        print(f"[plan] target_q={fmt_vec(action_plan.target_q)}")
        require_explicit_execution(args)
        assert_stationary_and_safe(
            state,
            allow_reduced_mode=args.allow_reduced_mode,
            max_start_joint_speed=args.max_start_joint_speed,
            max_start_tcp_speed=args.max_start_tcp_speed,
        )
        rtde_c = connect_control(args.robot_ip)
        assert_robot_accepts_target(rtde_c, action_plan.target_q)
        ok = rtde_c.moveJ(action_plan.target_q.tolist(), args.speed, args.accel)
        if ok is False:
            raise RuntimeError("policy-step moveJ returned False")
        print("[done] guarded policy step complete")
    except Exception:
        if rtde_c is not None:
            try:
                rtde_c.stopJ(args.accel)
            except Exception:
                pass
        raise
    finally:
        cleanup_control(rtde_c, servo=False)
        rtde_r.disconnect()


def print_workspace_check(name: str, value: float, limits: tuple[float, float]) -> None:
    ok = limits[0] <= value <= limits[1]
    print(f"[debug] {name}={value:.5f} train_range={limits} in_range={ok}")


def print_obs_segments(
    obs: np.ndarray,
    norm_obs: np.ndarray,
    *,
    prefix: str,
    include_quat: bool,
) -> None:
    segments = OBS_SEGMENTS_WITH_QUAT if include_quat else OBS_SEGMENTS_POSITION_ONLY
    for name, start, end in segments:
        print(f"{prefix} obs.{name}={fmt_vec(obs[start:end], precision=6)}")
        print(f"{prefix} norm.{name}={fmt_vec(norm_obs[start:end], precision=3)}")


def print_norm_outliers(norm_obs: np.ndarray, *, threshold: float) -> None:
    indices = np.flatnonzero(np.abs(norm_obs) >= threshold)
    if len(indices) == 0:
        print(f"[debug] normalized_obs_outliers(abs>={threshold})=none")
        return
    parts = [f"{int(i)}:{float(norm_obs[i]):.3f}" for i in indices]
    print(f"[debug] normalized_obs_outliers(abs>={threshold})={', '.join(parts)}")


def run_policy_debug(args: argparse.Namespace) -> None:
    rtde_r = connect_receive(args.robot_ip)
    try:
        state = read_state(rtde_r)
        target = resolve_target(args, state)
        policy = load_reach_policy(args.checkpoint)
        print_state(state)
        print(f"[debug] checkpoint={Path(args.checkpoint).resolve()}")
        print(f"[debug] input_dim={policy.input_dim}")  # type: ignore[attr-defined]
        print(f"[debug] action_mode={args.policy_action_mode}")
        print(f"[debug] target_pos_b={fmt_vec(target, precision=6)}")
        print(f"[debug] target_error={fmt_vec(target - state.tcp_pose[:3], precision=6)}")
        print_workspace_check("target_x", float(target[0]), TRAIN_TARGET_X_RANGE)
        print_workspace_check("target_y", float(target[1]), TRAIN_TARGET_Y_RANGE)
        print_workspace_check("target_z", float(target[2]), TRAIN_TARGET_Z_RANGE)

        modes = (
            ("raw", "negated", "positive-w", "positive-max", "negative-max")
            if policy.include_quat  # type: ignore[attr-defined]
            else ("position-only",)
        )
        for mode in modes:
            quat_sign = "raw" if mode == "position-only" else mode
            obs = build_reach_observation(
                state,
                target,
                include_quat=policy.include_quat,  # type: ignore[attr-defined]
                quat_sign=quat_sign,
            )
            norm_obs = policy.normalize(obs)  # type: ignore[attr-defined]
            raw_action = policy(obs)
            action_plan = plan_policy_action(
                raw_action,
                state.q,
                policy_action_mode=args.policy_action_mode,
                max_policy_action_abs=args.max_policy_action_abs,
                hardware_action_scale=args.hardware_action_scale,
                policy_action_scale=args.policy_action_scale,
                max_filtered_delta_rad=args.max_filtered_delta_rad,
            )
            ee_quat_b = canonicalize_quat_wxyz(
                rotvec_to_quat_wxyz(state.tcp_pose[3:]),
                quat_sign,
            )
            print(f"[debug:{mode}] ee_quat_b={fmt_vec(ee_quat_b, precision=8)}")
            print(f"[debug:{mode}] max_abs_norm_obs={float(np.max(np.abs(norm_obs))):.3f}")
            print(f"[debug:{mode}] raw_action={fmt_vec(raw_action)}")
            print(f"[debug:{mode}] clipped_action={fmt_vec(action_plan.clipped_action)}")
            print(f"[debug:{mode}] desired_policy_q={fmt_vec(action_plan.desired_q)}")
            print(f"[debug:{mode}] guarded_delta_q={fmt_vec(action_plan.guarded_delta_q, precision=6)}")
            print_norm_outliers(norm_obs, threshold=args.z_threshold)
            if args.full:
                print_obs_segments(
                    obs,
                    norm_obs,
                    prefix=f"[debug:{mode}]",
                    include_quat=policy.include_quat,  # type: ignore[attr-defined]
                )
        print("[debug] no control command sent")
    finally:
        rtde_r.disconnect()


def resolve_home_q(args: argparse.Namespace) -> np.ndarray:
    if args.home_q is None:
        return SIM_HOME_Q.copy()
    home_q = np.asarray(args.home_q, dtype=np.float64)
    if home_q.shape != (6,):
        raise SystemExit(f"Expected 6 joint values for --home-q, got {home_q.shape}.")
    return home_q


def print_home_plan(
    state: RobotState,
    target_q: np.ndarray,
    *,
    max_step_rad: float,
    done_tolerance_rad: float,
) -> None:
    delta = target_q - state.q
    max_abs_delta = float(np.max(np.abs(delta)))
    remaining_steps = int(math.ceil(max_abs_delta / max_step_rad)) if max_step_rad > 0 else 0
    print(f"[home] target_q={fmt_vec(target_q)}")
    print(f"[home] delta_q={fmt_vec(delta)}")
    print(f"[home] max_abs_delta_rad={max_abs_delta:.5f}")
    print(f"[home] max_step_rad={max_step_rad:.5f}")
    print(f"[home] estimated_steps_remaining={remaining_steps}")
    if max_abs_delta <= done_tolerance_rad:
        print("[home] already within tolerance")
    for i, name in enumerate(JOINT_NAMES):
        print(
            f"[home] joint {i} {name}: "
            f"current={state.q[i]: .5f} target={target_q[i]: .5f} delta={delta[i]: .5f}"
        )


def run_home_plan(args: argparse.Namespace) -> None:
    rtde_r = connect_receive(args.robot_ip)
    try:
        state = read_state(rtde_r)
        target_q = resolve_home_q(args)
        print_state(state)
        print_home_plan(
            state,
            target_q,
            max_step_rad=args.max_step_rad,
            done_tolerance_rad=args.done_tolerance_rad,
        )
        print("[home] no control command sent")
    finally:
        rtde_r.disconnect()


def run_home_step(args: argparse.Namespace) -> None:
    rtde_r = connect_receive(args.robot_ip)
    rtde_c = None
    try:
        state = read_state(rtde_r)
        target_q = resolve_home_q(args)
        delta = target_q - state.q
        max_abs_delta = float(np.max(np.abs(delta)))
        print_state(state)
        print_home_plan(
            state,
            target_q,
            max_step_rad=args.max_step_rad,
            done_tolerance_rad=args.done_tolerance_rad,
        )
        if max_abs_delta <= args.done_tolerance_rad:
            print("[done] home target already reached within tolerance")
            return

        step_delta = np.clip(delta, -args.max_step_rad, args.max_step_rad)
        next_q = state.q + step_delta
        assert_small_delta(step_delta, args.max_step_rad)
        print(
            "[plan] one bounded home step via moveJ "
            f"speed={args.speed:.4f} rad/s accel={args.accel:.4f} rad/s^2"
        )
        print(f"[plan] step_delta_q={fmt_vec(step_delta)}")
        print(f"[plan] next_q={fmt_vec(next_q)}")
        require_explicit_execution(args)
        assert_stationary_and_safe(
            state,
            allow_reduced_mode=args.allow_reduced_mode,
            max_start_joint_speed=args.max_start_joint_speed,
            max_start_tcp_speed=args.max_start_tcp_speed,
        )
        rtde_c = connect_control(args.robot_ip)
        assert_robot_accepts_target(rtde_c, next_q)
        ok = rtde_c.moveJ(next_q.tolist(), args.speed, args.accel)
        if ok is False:
            raise RuntimeError("home-step moveJ returned False")
        print("[done] one bounded home step complete")
    except Exception:
        if rtde_c is not None:
            try:
                rtde_c.stopJ(args.accel)
            except Exception:
                pass
        raise
    finally:
        cleanup_control(rtde_c, servo=False)
        rtde_r.disconnect()


def add_common_control_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--execute", action="store_true", help="Actually send the control command.")
    parser.add_argument(
        "--accept-risk",
        action="store_true",
        help="Confirm that the workspace is clear and an e-stop is reachable.",
    )
    parser.add_argument("--max-start-joint-speed", type=float, default=0.02)
    parser.add_argument("--max-start-tcp-speed", type=float, default=0.01)


def add_policy_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--checkpoint",
        default="logs/skrl/ur3e_reach_abs_ur10_reward_direct/2026-05-12_12-02-33_ppo_torch/checkpoints/best_agent.pt",
        help="Path to the skrl UR3e reach checkpoint.",
    )
    parser.add_argument(
        "--target",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Target in the robot base frame. Defaults to the current TCP position.",
    )
    parser.add_argument(
        "--target-offset",
        nargs=3,
        type=float,
        default=None,
        metavar=("DX", "DY", "DZ"),
        help="Small target offset from the current TCP position.",
    )
    parser.add_argument("--max-target-offset-m", type=float, default=0.02)
    parser.add_argument("--max-policy-action-abs", type=float, default=0.25)
    parser.add_argument("--hardware-action-scale", type=float, default=0.002)
    parser.add_argument(
        "--policy-action-mode",
        choices=("absolute", "delta"),
        default="absolute",
        help=(
            "How to interpret the policy action. New UR3e checkpoints use "
            "absolute: q_desired = sim_home_q + action * --policy-action-scale. "
            "Use delta only for legacy delta-action checkpoints."
        ),
    )
    parser.add_argument(
        "--policy-action-scale",
        type=float,
        default=0.5,
        help="Training action scale in radians for --policy-action-mode absolute.",
    )
    parser.add_argument("--max-filtered-delta-rad", type=float, default=0.002)
    parser.add_argument(
        "--quat-sign",
        choices=("raw", "negated", "positive-w", "positive-max", "negative-max"),
        default="raw",
        help=(
            "Quaternion sign handling before policy input. Default raw "
            "reproduces the original behavior."
        ),
    )


def add_home_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--home-q",
        nargs=6,
        type=float,
        default=None,
        metavar=("Q0", "Q1", "Q2", "Q3", "Q4", "Q5"),
        help="Home joint target in radians. Defaults to the sim home pose.",
    )
    parser.add_argument("--max-step-rad", type=float, default=0.05)
    parser.add_argument("--done-tolerance-rad", type=float, default=0.005)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot-ip",
        default=os.getenv("ROBOT_IP", "147.175.108.138"),
        help="UR controller IP. Defaults to ROBOT_IP or 147.175.108.138.",
    )
    parser.add_argument(
        "--allow-reduced-mode",
        action="store_true",
        help="Allow control when the robot reports reduced safety mode.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="Read and print one robot state sample.")
    status.set_defaults(func=run_status)

    watch = subparsers.add_parser("watch", help="Read-only state samples.")
    watch.add_argument("--samples", type=int, default=10)
    watch.add_argument("--interval", type=float, default=0.5)
    watch.set_defaults(func=run_watch)

    hold = subparsers.add_parser("hold", help="Servo-hold the current joint position.")
    add_common_control_args(hold)
    hold.add_argument("--seconds", type=float, default=2.0)
    hold.add_argument("--hz", type=float, default=20.0)
    hold.add_argument("--lookahead-time", type=float, default=0.12)
    hold.add_argument("--gain", type=float, default=100.0)
    hold.add_argument("--max-hold-error-rad", type=float, default=0.015)
    hold.set_defaults(func=run_hold)

    home_plan = subparsers.add_parser(
        "home-plan",
        help="Read-only plan for moving toward a home joint target.",
    )
    add_home_args(home_plan)
    home_plan.set_defaults(func=run_home_plan)

    home_step = subparsers.add_parser(
        "home-step",
        help="Move one bounded step toward a home joint target.",
    )
    add_common_control_args(home_step)
    add_home_args(home_step)
    home_step.add_argument("--speed", type=float, default=0.03)
    home_step.add_argument("--accel", type=float, default=0.06)
    home_step.set_defaults(func=run_home_step)

    tiny_movej = subparsers.add_parser("tiny-movej", help="One very small joint move using moveJ.")
    add_common_control_args(tiny_movej)
    tiny_movej.add_argument("--joint-index", type=int, choices=range(6), default=5)
    tiny_movej.add_argument("--delta-rad", type=float, default=0.005)
    tiny_movej.add_argument("--max-delta-rad", type=float, default=0.01)
    tiny_movej.add_argument("--speed", type=float, default=0.02)
    tiny_movej.add_argument("--accel", type=float, default=0.05)
    tiny_movej.set_defaults(func=run_tiny_movej)

    tiny_servoj = subparsers.add_parser("tiny-servoj", help="One tiny joint ramp using servoJ.")
    add_common_control_args(tiny_servoj)
    tiny_servoj.add_argument("--joint-index", type=int, choices=range(6), default=5)
    tiny_servoj.add_argument("--delta-rad", type=float, default=0.003)
    tiny_servoj.add_argument("--max-delta-rad", type=float, default=0.006)
    tiny_servoj.add_argument("--seconds", type=float, default=2.0)
    tiny_servoj.add_argument("--hz", type=float, default=20.0)
    tiny_servoj.add_argument("--lookahead-time", type=float, default=0.12)
    tiny_servoj.add_argument("--gain", type=float, default=100.0)
    tiny_servoj.set_defaults(func=run_tiny_servoj)

    policy_preview = subparsers.add_parser(
        "policy-preview",
        help="Read robot state, run the learned policy once, print the guarded action only.",
    )
    add_policy_args(policy_preview)
    policy_preview.set_defaults(func=run_policy_preview)

    policy_debug = subparsers.add_parser(
        "policy-debug",
        help="Read-only policy observation/action diagnostic across quaternion sign modes.",
    )
    add_policy_args(policy_debug)
    policy_debug.add_argument(
        "--full",
        action="store_true",
        help="Print all raw and normalized observation segments for every quaternion mode.",
    )
    policy_debug.add_argument(
        "--z-threshold",
        type=float,
        default=3.0,
        help="Report normalized observation entries with absolute value at or above this threshold.",
    )
    policy_debug.set_defaults(func=run_policy_debug)

    policy_step = subparsers.add_parser(
        "policy-step",
        help="Execute exactly one guarded learned-policy step via a slow moveJ.",
    )
    add_common_control_args(policy_step)
    add_policy_args(policy_step)
    policy_step.add_argument("--speed", type=float, default=0.015)
    policy_step.add_argument("--accel", type=float, default=0.04)
    policy_step.set_defaults(func=run_policy_step)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
