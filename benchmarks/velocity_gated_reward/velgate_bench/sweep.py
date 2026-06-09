# Copyright (c) 2026, Ales Melichar
# SPDX-License-Identifier: BSD-3-Clause

"""Sweep orchestrator for the velocity-gated reward benchmark.

Runs the cartesian product (task × kernel × seed) by dispatching one
training subprocess per cell. Each subprocess invokes the project's
existing ``scripts/skrl/train.py`` with the appropriate gym ID and seed.

After all training runs complete, optionally runs eval rollouts on each
``best_agent.pt`` to collect hold metrics.

Usage::

    # Default sweep (tight benchmark - 3 kernels × 5 seeds × 2 tasks = 30 runs)
    python -m velgate_bench.sweep --config configs/sweep.yaml

    # Custom sweep
    python -m velgate_bench.sweep --tasks franka_lift ur3e_lift \\
        --kernels tanh gaussian velocity_gated_tanh \\
        --seeds 0 1 2 3 4 --num_envs 4096 --max_iterations 10000

    # Eval only (skip training, run rollouts on existing checkpoints)
    python -m velgate_bench.sweep --config configs/sweep.yaml --eval_only

The sweep is *resumable*: if a run's output directory already contains a
``best_agent.pt``, it is skipped (unless ``--force`` is given).
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Repo root inferred from this file location.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_BENCH_DIR = _REPO_ROOT / "benchmarks" / "velocity_gated_reward"
_RESULTS_DIR = _BENCH_DIR / "results"


# -----------------------------------------------------------------------------
# Task definitions: gym ID template + arm-joint pattern for eval metrics
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskSpec:
    """Defines one benchmark task."""
    name: str
    gym_id_template: str
    arm_joint_pattern: str
    default_num_envs: int = 4096
    default_iterations: int | None = None  # uses YAML's default if None


TASKS: dict[str, TaskSpec] = {
    "franka_lift": TaskSpec(
        name="franka_lift",
        gym_id_template="Velgate-Bench-Franka-Lift-{kernel_display}-v0",
        arm_joint_pattern=r"panda_joint.*",
    ),
    "ur3e_lift": TaskSpec(
        name="ur3e_lift",
        gym_id_template="Velgate-Bench-UR3e-HandE-Lift-{kernel_display}-v0",
        arm_joint_pattern=r"(shoulder_.*|elbow_joint|wrist_.*)",
        default_num_envs=8192,
    ),
}

KERNEL_DISPLAY: dict[str, str] = {
    "tanh": "Tanh",
    "gaussian": "Gaussian",
    "velocity_gated_tanh": "VelocityGatedTanh",
}


# -----------------------------------------------------------------------------
# Sweep config (loadable from YAML)
# -----------------------------------------------------------------------------


@dataclass
class SweepCfg:
    tasks: list[str] = field(default_factory=lambda: list(TASKS.keys()))
    kernels: list[str] = field(default_factory=lambda: list(KERNEL_DISPLAY.keys()))
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    num_envs: int | None = None              # None -> task default
    max_iterations: int | None = None        # None -> skrl YAML default

    # Phases
    eval_only: bool = False
    skip_eval: bool = False
    eval_num_envs: int = 64
    eval_num_episodes: int = 100
    force: bool = False                      # rerun even if best_agent.pt exists

    # Video recording during TRAINING.
    # Videos save to <run_dir>/<skrl_timestamp_dir>/videos/train/*.mp4.
    # Adds ~10-20% overhead (renders intermittently). Off by default for sweeps.
    record_video_train: bool = False
    video_length: int = 500                  # frames per recorded clip
    video_interval: int = 5000               # record every N env steps

    def cells(self):
        return list(itertools.product(self.tasks, self.kernels, self.seeds))


# -----------------------------------------------------------------------------
# Run dispatch
# -----------------------------------------------------------------------------


def run_dir(task: str, kernel: str, seed: int) -> Path:
    return _RESULTS_DIR / "runs" / task / kernel / f"seed_{seed}"


def eval_path(task: str, kernel: str, seed: int) -> Path:
    return _RESULTS_DIR / "eval" / task / kernel / f"seed_{seed}.json"


def best_checkpoint(run_dir_: Path) -> Path | None:
    """Return path to best_agent.pt under skrl's nested timestamp dir, if any."""
    if not run_dir_.exists():
        return None
    for cand in run_dir_.rglob("best_agent.pt"):
        return cand
    return None


def _train_command(task_spec: TaskSpec, kernel: str, seed: int, cfg: SweepCfg) -> list[str]:
    gym_id = task_spec.gym_id_template.format(kernel_display=KERNEL_DISPLAY[kernel])
    num_envs = cfg.num_envs if cfg.num_envs is not None else task_spec.default_num_envs
    out_dir = run_dir(task_spec.name, kernel, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Use the benchmark's train wrapper so velgate_bench env variants get
    # registered before the project's train.py looks them up via gym.spec.
    cmd = [
        sys.executable, str(_BENCH_DIR / "scripts" / "train.py"),
        "--task", gym_id,
        "--num_envs", str(num_envs),
        "--seed", str(seed),
        "--headless",
        # Hydra overrides redirect skrl's TB + checkpoint writes into our
        # per-cell directory instead of the YAML default. Without these, all
        # cells of the same base task share one skrl log dir and our sweep
        # can't find best_agent.pt afterward.
        f"agent.experiment.directory={out_dir}",
        f"agent.experiment.experiment_name=skrl",
    ]
    if cfg.max_iterations is not None:
        cmd += ["--max_iterations", str(cfg.max_iterations)]
    if cfg.record_video_train:
        cmd += [
            "--video",
            "--video_length", str(cfg.video_length),
            "--video_interval", str(cfg.video_interval),
        ]
    return cmd, out_dir


def _eval_command(task_spec: TaskSpec, kernel: str, seed: int, ckpt: Path, cfg: SweepCfg) -> list[str]:
    gym_id = task_spec.gym_id_template.format(kernel_display=KERNEL_DISPLAY[kernel])
    out_json = eval_path(task_spec.name, kernel, seed)
    return [
        sys.executable, "-m", "velgate_bench.metrics",
        "--task", gym_id,
        "--checkpoint", str(ckpt),
        "--num_envs", str(cfg.eval_num_envs),
        "--num_episodes", str(cfg.eval_num_episodes),
        "--seed", str(seed),
        "--arm_joint_pattern", task_spec.arm_joint_pattern,
        "--out", str(out_json),
    ]


def _run_subprocess(cmd: list[str], log_path: Path, timeout: int | None = None) -> int:
    """Run a subprocess in its own process group, capture output to log_path.

    Process group isolation lets us kill the WHOLE process tree if the
    subprocess hangs (Isaac Sim's kit sometimes leaves orphan children that
    leak CUDA memory). Optional timeout enforces a max wall-clock per cell.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pretty = " ".join(shlex.quote(c) for c in cmd)
    print(f"\n[velgate-sweep] $ {pretty}")
    print(f"[velgate-sweep]   log: {log_path}")
    with open(log_path, "w") as logf:
        logf.write(f"# {pretty}\n\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd, stdout=logf, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,  # new process group -> we can killpg later
        )
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"[velgate-sweep] TIMEOUT after {timeout}s, killing process group")
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            time.sleep(2)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            rc = -9
    # Cleanup: even on normal exit, sweep occasionally leaves orphan kit
    # processes that hold CUDA memory. Try a best-effort orphan kill.
    _cleanup_orphan_kit_processes()
    return rc


def _cleanup_orphan_kit_processes() -> None:
    """Best-effort kill of any orphan Isaac Sim kit processes from this user.

    Isaac Sim's kit/python sometimes leaves children behind after a normal
    exit, holding CUDA memory and breaking the next run. We pattern-match
    on process name and only kill processes owned by the current user.
    """
    try:
        # List user's processes matching kit/isaac-sim patterns
        out = subprocess.check_output(
            ["pgrep", "-u", str(os.getuid()), "-f", "kit"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except subprocess.CalledProcessError:
        return  # no orphans
    if not out:
        return
    # Don't kill ourselves or our parent shells.
    self_pid = os.getpid()
    parent_pid = os.getppid()
    killed = []
    for line in out.splitlines():
        try:
            pid = int(line.strip())
        except ValueError:
            continue
        if pid in (self_pid, parent_pid):
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append(pid)
        except (ProcessLookupError, PermissionError):
            pass
    if killed:
        print(f"[velgate-sweep]   killed orphan kit PIDs: {killed}")


def _wait_for_gpu_memory(min_free_mb: int = 4000, max_wait_s: int = 30) -> None:
    """Block until at least ``min_free_mb`` of GPU memory is free, or timeout."""
    deadline = time.monotonic() + max_wait_s
    while time.monotonic() < deadline:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            free_mb = max(int(x) for x in out.splitlines() if x.strip())
            if free_mb >= min_free_mb:
                return
            print(f"[velgate-sweep]   GPU free {free_mb} MB < {min_free_mb} MB, waiting...")
            time.sleep(3)
        except Exception:
            return  # nvidia-smi unavailable; just continue


# -----------------------------------------------------------------------------
# Sweep entry
# -----------------------------------------------------------------------------


def run_sweep(cfg: SweepCfg) -> dict:
    """Execute the sweep. Returns a manifest dict listing each cell + status."""
    manifest = {"cells": [], "config": dataclasses.asdict(cfg)}
    for task_name, kernel, seed in cfg.cells():
        task_spec = TASKS[task_name]
        cell = {"task": task_name, "kernel": kernel, "seed": seed,
                "train_status": "skipped", "eval_status": "skipped"}

        # --- training phase ---
        out_dir = run_dir(task_name, kernel, seed)
        ckpt = best_checkpoint(out_dir)
        if cfg.eval_only:
            cell["train_status"] = "skipped(eval_only)"
        elif ckpt and not cfg.force:
            cell["train_status"] = "cached"
        else:
            # Block until GPU has enough free memory (handles slow CUDA teardown
            # from the previous cell). Conservative threshold: 4 GB.
            _wait_for_gpu_memory(min_free_mb=4000, max_wait_s=60)
            cmd, _ = _train_command(task_spec, kernel, seed, cfg)
            log_path = out_dir / "train.log"
            rc = _run_subprocess(cmd, log_path)
            cell["train_status"] = "ok" if rc == 0 else f"fail(rc={rc})"
            ckpt = best_checkpoint(out_dir)

        # --- eval phase ---
        if cfg.skip_eval:
            cell["eval_status"] = "skipped(--skip_eval)"
        elif ckpt is None:
            cell["eval_status"] = "no_checkpoint"
        else:
            cell["checkpoint"] = str(ckpt)
            eval_log = eval_path(task_name, kernel, seed).with_suffix(".log")
            eval_cmd = _eval_command(task_spec, kernel, seed, ckpt, cfg)
            rc = _run_subprocess(eval_cmd, eval_log)
            cell["eval_status"] = "ok" if rc == 0 else f"fail(rc={rc})"
            cell["eval_json"] = str(eval_path(task_name, kernel, seed))

        manifest["cells"].append(cell)

    manifest_path = _RESULTS_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n[velgate-sweep] Manifest written to {manifest_path}")
    return manifest


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _load_yaml(path: Path) -> SweepCfg:
    import yaml
    raw = yaml.safe_load(path.read_text())
    return SweepCfg(**raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run velocity-gated reward sweep")
    parser.add_argument("--config", type=Path, help="YAML sweep config")
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS.keys()))
    parser.add_argument("--kernels", nargs="+", choices=list(KERNEL_DISPLAY.keys()))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--max_iterations", type=int)
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training; run only eval rollouts on existing checkpoints")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Train only; skip eval rollouts")
    parser.add_argument("--eval_num_envs", type=int, default=64)
    parser.add_argument("--eval_num_episodes", type=int, default=100)
    parser.add_argument("--force", action="store_true",
                        help="Force retrain even if best_agent.pt exists")
    parser.add_argument("--record_video_train", action="store_true",
                        help="Record videos during training (~10-20%% overhead)")
    parser.add_argument("--video_length", type=int,
                        help="Frames per recorded video clip")
    parser.add_argument("--video_interval", type=int,
                        help="Env steps between video clips")
    args = parser.parse_args(argv)

    if args.config:
        cfg = _load_yaml(args.config)
        # CLI flags override YAML
        for k, v in vars(args).items():
            if k == "config" or v is None or v is False:
                continue
            setattr(cfg, k, v)
    else:
        cfg_kwargs = {k: v for k, v in vars(args).items()
                      if k != "config" and v is not None and v is not False}
        cfg = SweepCfg(**cfg_kwargs)

    manifest = run_sweep(cfg)
    n_train_ok = sum(1 for c in manifest["cells"] if c["train_status"] in ("ok", "cached"))
    n_eval_ok = sum(1 for c in manifest["cells"] if c["eval_status"] == "ok")
    print(f"\n[velgate-sweep] Summary: {n_train_ok}/{len(manifest['cells'])} training OK, "
          f"{n_eval_ok}/{len(manifest['cells'])} eval OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
