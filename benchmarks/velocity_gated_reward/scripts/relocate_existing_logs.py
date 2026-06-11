#!/usr/bin/env python
# Copyright (c) 2026, Ales Melichar
# SPDX-License-Identifier: BSD-3-Clause

"""Relocate skrl logs that ended up in ``logs/skrl/<basename>/`` into the
per-cell ``results/runs/<task>/<kernel>/seed_N/skrl/`` layout the sweep
expects.

Use case: an earlier sweep was run without the relocation logic in
``sweep.py``. All training succeeded, but TB events and checkpoints landed
in the skrl default location. This script matches them to cells by reading
each skrl run's ``params/env.yaml`` (which lists the gym ID) and ``params/agent.yaml``
(which has the experiment_name) plus directory mtime ordering.

Usage::

    python scripts/relocate_existing_logs.py [--dry-run]
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

_BENCH_DIR = Path(__file__).resolve().parents[1]
_LOGS_SKRL = _BENCH_DIR / "logs" / "skrl"
_RESULTS_RUNS = _BENCH_DIR / "results" / "runs"

# Map: skrl_basename -> (task_name, gym_id_pattern)
_TASK_INFO = {
    "franka_lift": (
        "franka_lift",
        re.compile(r"Velgate-Bench-Franka-Lift-(Tanh|Gaussian|VelocityGatedTanh)-v0"),
    ),
    "ur3e_hande_lift_cube": (
        "ur3e_lift",
        re.compile(r"Velgate-Bench-UR3e-HandE-Lift-(Tanh|Gaussian|VelocityGatedTanh)-v0"),
    ),
}

_KERNEL_FROM_DISPLAY = {
    "Tanh": "tanh",
    "Gaussian": "gaussian",
    "VelocityGatedTanh": "velocity_gated_tanh",
}


def _read_env_yaml_task(env_yaml_path: Path) -> str | None:
    """Extract gym ID from skrl's params/env.yaml. Naive grep, no YAML parser needed."""
    try:
        text = env_yaml_path.read_text()
    except OSError:
        return None
    # Hydra dumps the task as `task_name: Velgate-Bench-...` somewhere
    m = re.search(r"Velgate-Bench-\S+", text)
    return m.group(0) if m else None


def _read_seed(env_yaml_path: Path) -> int | None:
    try:
        text = env_yaml_path.read_text()
    except OSError:
        return None
    m = re.search(r"^seed:\s*(\d+)", text, flags=re.MULTILINE)
    return int(m.group(1)) if m else None


def _match_cell(skrl_run: Path, task_pattern: re.Pattern, task_name: str) -> tuple[str, str, int] | None:
    """Given a single skrl timestamp dir, identify (task, kernel, seed)."""
    params_dir = skrl_run / "params"
    if not params_dir.exists():
        return None
    env_yaml = params_dir / "env.yaml"
    gym_id = _read_env_yaml_task(env_yaml)
    if not gym_id:
        return None
    m = task_pattern.search(gym_id)
    if not m:
        return None
    kernel_display = m.group(1)
    kernel = _KERNEL_FROM_DISPLAY[kernel_display]
    # Seed might be in agent.yaml or env.yaml
    seed = _read_seed(env_yaml)
    if seed is None:
        seed = _read_seed(params_dir / "agent.yaml")
    if seed is None:
        return None
    return task_name, kernel, seed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be moved, don't actually move.")
    args = parser.parse_args(argv)

    if not _LOGS_SKRL.exists():
        print(f"No skrl logs at {_LOGS_SKRL} -- nothing to relocate.")
        return 0

    moves: list[tuple[Path, Path]] = []
    unmatched: list[Path] = []

    for basename, (task_name, pattern) in _TASK_INFO.items():
        task_dir = _LOGS_SKRL / basename
        if not task_dir.exists():
            continue
        for run_dir in sorted(task_dir.iterdir(), key=lambda p: p.stat().st_mtime):
            if not run_dir.is_dir():
                continue
            match = _match_cell(run_dir, pattern, task_name)
            if not match:
                unmatched.append(run_dir)
                continue
            _, kernel, seed = match
            target = _RESULTS_RUNS / task_name / kernel / f"seed_{seed}" / "skrl"
            if target.exists():
                # Stash existing one as backup
                backup = target.parent / f"skrl_prior_{int(run_dir.stat().st_mtime)}"
                moves.append((target, backup))
            moves.append((run_dir, target))

    print(f"Plan: {len(moves)} move(s), {len(unmatched)} unmatched.")
    for src, dst in moves:
        print(f"  {src}  ->  {dst}")
    for u in unmatched:
        print(f"  UNMATCHED: {u}  (no params/env.yaml or no gym ID match)")

    if args.dry_run:
        print("\n--dry-run: no changes made.")
        return 0

    for src, dst in moves:
        if not src.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    print(f"\nMoved {len(moves)} dir(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
