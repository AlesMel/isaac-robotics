# Copyright (c) 2026, Ales Melichar
# SPDX-License-Identifier: BSD-3-Clause

"""Sweep orchestrator for the velocity-gated reward benchmark.

Runs the cartesian product (task × kernel × seed) by dispatching one
training subprocess per cell. Each subprocess invokes the project's
existing ``scripts/skrl/train.py`` with the appropriate gym ID and seed.

After all training runs complete, optionally runs eval rollouts on each
``best_agent.pt`` to collect hold metrics.

Usage::

    # Default sweep (5 kernels franka + 4 kernels ur3e, 5 seeds each = 45 runs)
    python -m velgate_bench.sweep --config configs/sweep_default.yaml

    # Custom sweep
    python -m velgate_bench.sweep --tasks franka_lift ur3e_lift \\
        --kernels tanh gaussian velocity_gated_tanh \\
        --seeds 0 1 2 3 4 --num_envs 4096 --max_timesteps 500000

    # Eval only (skip training, run rollouts on existing checkpoints)
    python -m velgate_bench.sweep --config configs/sweep.yaml --eval_only

The sweep is *resumable*: if a run's output directory already contains a
``best_agent.pt``, it is skipped (unless ``--force`` is given).
"""

from __future__ import annotations

import argparse
import dataclasses
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
    # Upstream skrl YAML's default ``agent.experiment.directory`` (the leaf name
    # skrl uses under ``logs/skrl/`` when no override is given). We now override
    # the experiment directory per-cell (see _train_command) so skrl writes
    # straight into the results tree; this field is kept only for reference and
    # for the one-shot ``relocate_existing_logs.py`` migration tool.
    skrl_basename: str = ""
    default_num_envs: int = 4096
    # Per-task training budget in skrl *trainer timesteps* (vector-env steps).
    # None -> the task's skrl YAML default. NOTE: this is NOT the same unit as
    # the upstream train.py ``--max_iterations`` flag, which gets multiplied by
    # ``rollouts`` (24/32) before becoming trainer timesteps -- that silent
    # multiplication once turned an intended 500k-step budget into a 12M-step
    # request that every cell timeout-killed. The sweep therefore sets
    # ``++agent.trainer.timesteps`` directly and never uses --max_iterations.
    default_max_timesteps: int | None = None


TASKS: dict[str, TaskSpec] = {
    "franka_lift": TaskSpec(
        name="franka_lift",
        gym_id_template="Velgate-Bench-Franka-Lift-{kernel_display}-v0",
        arm_joint_pattern=r"panda_joint.*",
        skrl_basename="franka_lift",
        # Upstream YAML default is 36k, far short of convergence at 4096 envs.
        # Seed-0 plateau analysis: reward peaks ~240k-360k timesteps; 500k
        # covers the peak plus margin to expose late-training drift.
        default_max_timesteps=500_000,
    ),
    "ur3e_lift": TaskSpec(
        name="ur3e_lift",
        gym_id_template="Velgate-Bench-UR3e-HandE-Lift-{kernel_display}-v0",
        arm_joint_pattern=r"(shoulder_.*|elbow_joint|wrist_.*)",
        default_num_envs=8192,
        skrl_basename="ur3e_hande_lift_cube",
        default_max_timesteps=300_000,  # matches the tuned ur3e skrl YAML
    ),
}

KERNEL_DISPLAY: dict[str, str] = {
    "tanh": "Tanh",
    "gaussian": "Gaussian",
    "velocity_gated_tanh": "VelocityGatedTanh",
    "tanh_additive_velpen": "TanhAdditiveVelPen",
    "velocity_gated_tanh_smooth": "VelocityGatedTanhSmooth",
}


# -----------------------------------------------------------------------------
# Sweep config (loadable from YAML)
# -----------------------------------------------------------------------------


@dataclass
class SweepCfg:
    tasks: list[str] = field(default_factory=lambda: list(TASKS.keys()))
    kernels: list[str] = field(default_factory=lambda: list(KERNEL_DISPLAY.keys()))
    # Per-task kernel override (e.g. the smooth-gate ablation runs on franka
    # only). Tasks absent from this dict fall back to ``kernels``.
    kernels_per_task: dict[str, list[str]] | None = None
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    num_envs: int | None = None              # None -> task default
    # Training budget in skrl trainer timesteps (see TaskSpec note); None ->
    # the per-task default_max_timesteps, which itself falls back to the YAML.
    max_timesteps: int | None = None
    # skrl TB scalar write interval (trainer timesteps between points). The
    # YAML default "auto" = timesteps/100 gave curves too sparse for the drift
    # metric; 500 yields 1000 points for franka's 500k / 600 for ur3e's 300k.
    tb_write_interval: int | None = 500

    # Phases
    eval_only: bool = False
    skip_eval: bool = False
    eval_num_envs: int = 64
    eval_num_episodes: int = 100
    force: bool = False                      # rerun even if best_agent.pt exists

    # Per-cell wall-clock ceilings. Isaac Sim occasionally deadlocks on
    # startup/shutdown (a hung eval has been observed spinning a full CPU core
    # for hours with no progress). Without a timeout, _run_subprocess waits
    # forever and the whole sweep stalls on one bad cell. On timeout the
    # subprocess' process group is killed (freeing the GPU), the cell is marked
    # failed, and the sweep advances to the next cell.
    train_timeout_s: int = 14400             # 4 h  (slowest: full ur3e run)
    eval_timeout_s: int = 1800               # 30 min

    # Video recording during TRAINING.
    # Videos save to <run_dir>/<skrl_timestamp_dir>/videos/train/*.mp4.
    # Adds ~10-20% overhead (renders intermittently). The kit-crash-dialog
    # watchdog in _run_subprocess limits the blast radius of render crashes.
    record_video_train: bool = False
    video_length: int = 500                  # frames per recorded clip
    # Trainer timesteps between clips. None -> auto: one clip every 10% of
    # the cell's training budget (~11 clips/run incl. step 0), so coverage is
    # uniform in training progress regardless of per-task budgets.
    video_interval: int | None = None

    def kernels_for(self, task: str) -> list[str]:
        return (self.kernels_per_task or {}).get(task, self.kernels)

    def cells(self):
        return [
            (task, kernel, seed)
            for task in self.tasks
            for kernel in self.kernels_for(task)
            for seed in self.seeds
        ]


# -----------------------------------------------------------------------------
# Run dispatch
# -----------------------------------------------------------------------------


def _resolved_max_timesteps(task_spec: TaskSpec, cfg: SweepCfg) -> int | None:
    return cfg.max_timesteps if cfg.max_timesteps is not None else task_spec.default_max_timesteps


def _resolved_video_interval(task_spec: TaskSpec, cfg: SweepCfg) -> int:
    """Trainer timesteps between video clips; auto = 10% of the cell budget."""
    if cfg.video_interval is not None:
        return cfg.video_interval
    max_ts = _resolved_max_timesteps(task_spec, cfg)
    return max(1, max_ts // 10) if max_ts else 10_000


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


def _train_command(task_spec: TaskSpec, kernel: str, seed: int, cfg: SweepCfg) -> tuple[list[str], Path]:
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
        # Redirect skrl's TB + checkpoint writes straight into our per-cell
        # directory. IsaacLab's hydra_task_config wraps the whole skrl YAML
        # under a top-level ``agent`` node (cfg = {"env": ..., "agent": ...}),
        # and the skrl YAML *itself* has an ``agent`` block -- so the hydra path
        # to skrl's experiment settings is ``agent.agent.experiment.*`` (note
        # the doubled ``agent``). The earlier single-``agent`` key silently
        # created a phantom node that train.py never read, which is why logs
        # used to land in the default dir and needed relocating afterward.
        # ``directory`` is absolute, so train.py's os.path.join("logs","skrl",
        # dir) collapses to out_dir; the run lands at
        # out_dir/<timestamp>_ppo_torch_skrl/ -- no relocation required.
        f"++agent.agent.experiment.directory={out_dir}",
        f"++agent.agent.experiment.experiment_name=skrl",
    ]
    # Training budget, expressed directly in skrl trainer timesteps. The
    # ``trainer`` block is top-level in the skrl YAML, so its hydra path gets a
    # SINGLE ``agent`` prefix (unlike the experiment block above). We never use
    # train.py's --max_iterations flag: it multiplies by ``rollouts`` before
    # writing trainer.timesteps, which silently inflates the budget 24-32x.
    max_ts = _resolved_max_timesteps(task_spec, cfg)
    if max_ts is not None:
        cmd += [f"++agent.trainer.timesteps={max_ts}"]
    # Dense TB scalars so the drift metric has a real curve to work with
    # (skrl's "auto" = timesteps/100 is far too sparse).
    if cfg.tb_write_interval is not None:
        cmd += [f"++agent.agent.experiment.write_interval={cfg.tb_write_interval}"]
    if cfg.record_video_train:
        cmd += [
            "--video",
            "--video_length", str(cfg.video_length),
            "--video_interval", str(_resolved_video_interval(task_spec, cfg)),
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


# Log markers that mean the subprocess is dead in the water and will never
# exit on its own. Isaac Sim's kit pops a (headless-invisible) crash dialog
# and blocks forever waiting for a button press; one such hang burned a full
# 8h timeout. Seen verbatim in a real train.log.
_FATAL_LOG_MARKERS = ("Press ABORT to exit",)


def _tail_bytes(path: Path, n: int = 4096) -> str:
    """Return the last ``n`` bytes of ``path`` as text (best-effort)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - n))
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _run_subprocess(cmd: list[str], log_path: Path, timeout: int | None = None) -> tuple[int, str]:
    """Run a subprocess in its own process group, capture output to log_path.

    Process group isolation lets us kill the WHOLE process tree if the
    subprocess hangs (Isaac Sim's kit sometimes leaves orphan children that
    leak CUDA memory). Optional timeout enforces a max wall-clock per cell.

    Returns ``(rc, reason)``; ``reason`` is "" on a clean exit, else
    "timeout" or "kit_crash_dialog" so callers can record WHY a cell was
    killed instead of an opaque rc=-9.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pretty = " ".join(shlex.quote(c) for c in cmd)
    print(f"\n[velgate-sweep] $ {pretty}")
    print(f"[velgate-sweep]   log: {log_path}")

    def _killpg(proc: subprocess.Popen) -> None:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        time.sleep(2)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass

    reason = ""
    poll_s = 30
    with open(log_path, "w") as logf:
        logf.write(f"# {pretty}\n\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd, stdout=logf, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,  # new process group -> we can killpg later
        )
        deadline = time.monotonic() + timeout if timeout else None
        while True:
            try:
                rc = proc.wait(timeout=poll_s)
                break
            except subprocess.TimeoutExpired:
                pass
            if deadline is not None and time.monotonic() > deadline:
                print(f"[velgate-sweep] TIMEOUT after {timeout}s, killing process group")
                _killpg(proc)
                rc, reason = -9, "timeout"
                break
            tail = _tail_bytes(log_path)
            marker = next((m for m in _FATAL_LOG_MARKERS if m in tail), None)
            if marker is not None:
                print(f"[velgate-sweep] FATAL log marker {marker!r} detected "
                      f"(kit crash dialog hang), killing process group early")
                _killpg(proc)
                rc, reason = -9, "kit_crash_dialog"
                break
    # Cleanup: even on normal exit, sweep occasionally leaves orphan kit
    # processes that hold CUDA memory. Try a best-effort orphan kill.
    _cleanup_orphan_kit_processes()
    return rc, reason


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


_KERNEL_DESCRIPTIONS = {
    "tanh": {
        "formula": "R(d) = lifted * (1 - tanh(d / std))",
        "gradient_at_d0": "-1/std (nonzero - causes limit cycle at goal)",
        "expected_failure": "limit cycle / oscillation when holding cube at goal",
        "params": {"std": 0.05},
    },
    "gaussian": {
        "formula": "R(d) = lifted * exp(-(d / std)^2)",
        "gradient_at_d0": "0 (smooth peak)",
        "expected_failure": "late-stage policy drift (no centering pull at d=0)",
        "params": {"std": 0.05},
    },
    "velocity_gated_tanh": {
        "formula": "R(d, q_dot) = lifted * (1 - tanh(d/std)) * G(d, q_dot)",
        "gate_formula": "G = inside(d<r) * clip(1 - |q_dot|/v_thresh, 0, 1) + outside(d>=r)",
        "gradient_at_d0": "-1/std (preserved tanh centering pull)",
        "expected_behavior": "no limit cycle (gate penalizes own oscillation), no drift (centering preserved)",
        "params": {"std": 0.05, "velocity_thresh": 0.5, "neighborhood": 0.10},
    },
    "tanh_additive_velpen": {
        "formula": "R(d, q_dot) = lifted * (1 - tanh(d/std)) - scale * 1[d<r] * clip(|q_dot|/v_thresh, 0, 1)",
        "gradient_at_d0": "-1/std (tanh centering pull preserved)",
        "expected_behavior": (
            "additive counterpart of the multiplicative velocity gate; the key "
            "baseline for 'is multiplicative gating better than a plain "
            "distance-gated velocity penalty?'"
        ),
        "params": {"std": 0.05, "velocity_thresh": 0.5, "neighborhood": 0.10, "penalty_scale": 1.0},
    },
    "velocity_gated_tanh_smooth": {
        "formula": "R(d, q_dot) = lifted * (1 - tanh(d/std)) * G_smooth(d, q_dot)",
        "gate_formula": "G_smooth = 1 - sigmoid((r - d)/tau) * (1 - clip(1 - |q_dot|/v_thresh, 0, 1))",
        "gradient_at_d0": "-1/std (preserved tanh centering pull)",
        "expected_behavior": (
            "ablation of velocity_gated_tanh with the hard neighborhood "
            "indicator replaced by a sigmoid blend -- removes the reward "
            "discontinuity at d=r that could create a boundary-orbiting local "
            "optimum"
        ),
        "params": {"std": 0.05, "velocity_thresh": 0.5, "neighborhood": 0.10, "tau": 0.02},
    },
}


def _write_cell_info(out_dir: Path, task_spec: TaskSpec, kernel: str, seed: int,
                     cfg: SweepCfg, gym_id: str, train_cmd: list[str]) -> None:
    """Write a JSON + plain-text marker describing what this cell tests.

    Two files:

    1. ``KERNEL.txt`` - single line, ``cat`` / ``grep`` friendly, for
       at-a-glance identification while monitoring the sweep.

    2. ``cell_info.json`` - structured metadata for downstream analysis.

    Both go alongside skrl logs so months later you can look at any
    directory and immediately know what differs from sibling cells.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_line = f"task={task_spec.name}  kernel={kernel}  seed={seed}  gym_id={gym_id}\n"
    (out_dir / "KERNEL.txt").write_text(summary_line)
    info = {
        "cell": {
            "task": task_spec.name,
            "kernel": kernel,
            "seed": seed,
            "gym_id": gym_id,
        },
        "kernel_under_test": _KERNEL_DESCRIPTIONS.get(kernel, {"unknown": kernel}),
        "training": {
            "num_envs": cfg.num_envs if cfg.num_envs is not None else task_spec.default_num_envs,
            "max_timesteps": _resolved_max_timesteps(task_spec, cfg),
            "tb_write_interval": cfg.tb_write_interval,
            "video_recording_enabled": cfg.record_video_train,
            "video_length_frames": cfg.video_length if cfg.record_video_train else None,
            "video_interval_steps": _resolved_video_interval(task_spec, cfg) if cfg.record_video_train else None,
        },
        "what_is_unique_about_this_cell": (
            f"This is one of {len(cfg.kernels_for(task_spec.name))} kernels x {len(cfg.seeds)} seeds for {task_spec.name}. "
            f"What differs from sibling cells of the same (task, seed) but different kernel: "
            f"ONLY the fine-grained goal-tracking reward function. All other hyperparameters "
            f"(network, PPO settings, action space, observation space, simulator) are identical "
            f"across the kernel conditions for this task. "
            f"What differs from sibling cells of the same (task, kernel) but different seed: "
            f"random seed only ({seed} here). PPO is stochastic, so the same kernel + same task "
            f"+ different seed produces a different policy trajectory; this captures variance."
        ),
        "outputs": {
            "skrl_run_dir": "<timestamp>_ppo_torch_skrl/ (skrl writes here directly)",
            "tensorboard_events": "<timestamp>_ppo_torch_skrl/events.out.tfevents.*",
            "checkpoints": "<timestamp>_ppo_torch_skrl/checkpoints/{agent_<step>.pt,best_agent.pt}",
            "params_snapshot": "<timestamp>_ppo_torch_skrl/params/{env.yaml,agent.yaml}",
            "videos": "<timestamp>_ppo_torch_skrl/videos/train/*.mp4" if cfg.record_video_train else "(not recorded; cfg.record_video_train=False)",
            "train_stdout_log": "train.log",
            "cell_metadata": "cell_info.json (this file)",
        },
        "subprocess_cmd": train_cmd,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cell_info.json").write_text(json.dumps(info, indent=2))


def _write_velgate_tb_metadata(skrl_dir: Path, task_spec: TaskSpec, kernel: str,
                               seed: int, cfg: SweepCfg) -> None:
    """Write velgate-specific text + hparams to the TB event stream.

    SkRL already writes its own events.out.tfevents.* with training scalars
    into ``skrl_dir``. We add a SECOND event file (filename_suffix=".velgate")
    in the same directory; TB will discover both and merge them in the UI.

    What we add:
      - Text tab entries: kernel name, formula, expected behavior, params.
        Shows up in TB at http://localhost:6006 -> "Text" pane.
      - HParams entry: structured (kernel, task, seed) so the TB HParams
        plugin can compare runs across the sweep on the same scalar axis.
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("[velgate-sweep]   torch.utils.tensorboard not installed; skipping TB metadata")
        return

    skrl_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(skrl_dir), filename_suffix=".velgate")

    info = _KERNEL_DESCRIPTIONS.get(kernel, {})

    # Text annotations - human-readable, visible under "Text" tab in TB
    writer.add_text("velgate/00_kernel", kernel, global_step=0)
    writer.add_text("velgate/01_task", task_spec.name, global_step=0)
    writer.add_text("velgate/02_seed", str(seed), global_step=0)
    writer.add_text("velgate/03_formula", info.get("formula", "<unknown>"), global_step=0)
    expectation = info.get("expected_failure") or info.get("expected_behavior") or ""
    writer.add_text("velgate/04_expectation", expectation, global_step=0)
    writer.add_text("velgate/05_params", json.dumps(info.get("params", {})), global_step=0)
    if "gate_formula" in info:
        writer.add_text("velgate/06_gate_formula", info["gate_formula"], global_step=0)
    if "gradient_at_d0" in info:
        writer.add_text("velgate/07_gradient_at_d0", info["gradient_at_d0"], global_step=0)

    # HParams - the TB HParams plugin uses these to build a comparison view
    # across runs. Numeric metrics get filled in by analyze.py post-sweep.
    hparams = {
        "kernel": kernel,
        "task": task_spec.name,
        "seed": seed,
        "num_envs": cfg.num_envs if cfg.num_envs is not None else task_spec.default_num_envs,
        "kernel_std": info.get("params", {}).get("std", 0.0),
    }
    kernel_params = info.get("params", {})
    for extra in ("velocity_thresh", "neighborhood", "penalty_scale", "tau"):
        if extra in kernel_params:
            hparams[extra] = kernel_params[extra]
    # Placeholder metrics; real values populated by analyze.py later.
    writer.add_hparams(hparams, metric_dict={"placeholder": 0.0}, run_name=".")

    writer.close()
    print(f"[velgate-sweep]   wrote velgate TB metadata to {skrl_dir}")


def _skrl_run_dir(out_dir: Path) -> Path | None:
    """Return the skrl run directory created under ``out_dir``.

    With ``agent.agent.experiment.directory`` overridden to ``out_dir`` (see
    _train_command), skrl writes straight here, creating a single
    ``<timestamp>_<algo>_<framework>_skrl`` subdir that holds ``checkpoints/``
    and the TB ``events.out.tfevents.*`` stream. We return the newest such
    subdir (there is normally exactly one, since _clear_skrl_runs wipes stale
    ones before each (re)train).
    """
    if not out_dir.exists():
        return None
    candidates = [
        p for p in out_dir.iterdir()
        if p.is_dir() and ((p / "checkpoints").exists() or any(p.glob("events.out.tfevents.*")))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _clear_skrl_runs(out_dir: Path) -> None:
    """Wipe any prior skrl run dir(s) under ``out_dir`` before a (re)train.

    Only invoked when we are about to train this cell (it was not cached, or
    ``--force`` was given), so a completed cell's results are never touched.
    Sibling metadata (cell_info.json, KERNEL.txt, train.log) is left intact;
    only the timestamped skrl run dir(s) are removed, guaranteeing
    best_checkpoint() finds a single unambiguous best_agent.pt afterward.
    """
    if not out_dir.exists():
        return
    import shutil as _sh
    for p in out_dir.iterdir():
        if p.is_dir() and ((p / "checkpoints").exists() or any(p.glob("events.out.tfevents.*"))):
            _sh.rmtree(p, ignore_errors=True)
            print(f"[velgate-sweep]   cleared stale skrl run dir: {p.name}")


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
            # Wipe any stale skrl run dir from a previous crashed/forced attempt
            # so this cell (re)trains into a single unambiguous output dir.
            _clear_skrl_runs(out_dir)
            cmd, _ = _train_command(task_spec, kernel, seed, cfg)
            gym_id = task_spec.gym_id_template.format(kernel_display=KERNEL_DISPLAY[kernel])
            _write_cell_info(out_dir, task_spec, kernel, seed, cfg, gym_id, cmd)
            log_path = out_dir / "train.log"
            rc, kill_reason = _run_subprocess(cmd, log_path, timeout=cfg.train_timeout_s)
            cell["train_status"] = "ok" if rc == 0 else f"fail({kill_reason or f'rc={rc}'})"
            # skrl wrote straight into out_dir/<timestamp>_ppo_torch_skrl/ via the
            # agent.agent.experiment.directory override -- no relocation needed.
            skrl_dir = _skrl_run_dir(out_dir)
            if skrl_dir is not None:
                cell["skrl_dir"] = str(skrl_dir)
                # Add velgate metadata (kernel name, formula, hparams) into the
                # same TB event dir so it shows up in TB's Text + HParams tabs.
                _write_velgate_tb_metadata(skrl_dir, task_spec, kernel, seed, cfg)
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
            rc, kill_reason = _run_subprocess(eval_cmd, eval_log, timeout=cfg.eval_timeout_s)
            cell["eval_status"] = "ok" if rc == 0 else f"fail({kill_reason or f'rc={rc}'})"
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
    parser.add_argument("--max_timesteps", type=int,
                        help="Training budget in skrl trainer timesteps (NOT iterations; "
                             "never multiplied by rollouts). Default: per-task budget.")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training; run only eval rollouts on existing checkpoints")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Train only; skip eval rollouts")
    # NOTE: every optional flag must default to None/False -- the merge loop
    # below treats None/False as "not given" so YAML values survive. A non-None
    # default here would silently clobber the YAML (this happened with the old
    # default=64/100 eval flags).
    parser.add_argument("--eval_num_envs", type=int, default=None)
    parser.add_argument("--eval_num_episodes", type=int, default=None)
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

    # Record the orchestrator's own PID so a human can find/kill the sweep
    # later (previously this file was written by the launch shell and went
    # stale across restarts).
    pid_path = _BENCH_DIR / "sweep.pid"
    pid_path.write_text(f"{os.getpid()}\n")
    try:
        manifest = run_sweep(cfg)
    finally:
        pid_path.unlink(missing_ok=True)
    n_train_ok = sum(1 for c in manifest["cells"] if c["train_status"] in ("ok", "cached"))
    n_eval_ok = sum(1 for c in manifest["cells"] if c["eval_status"] == "ok")
    print(f"\n[velgate-sweep] Summary: {n_train_ok}/{len(manifest['cells'])} training OK, "
          f"{n_eval_ok}/{len(manifest['cells'])} eval OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
