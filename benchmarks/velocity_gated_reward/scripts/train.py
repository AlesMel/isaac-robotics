#!/usr/bin/env python
# Copyright (c) 2026, Ales Melichar
# SPDX-License-Identifier: BSD-3-Clause

"""Thin wrapper around the project's ``scripts/skrl/train.py`` that ensures
``velgate_bench.envs`` is imported (and its gym IDs registered) BEFORE the
hydra task lookup happens.

Why a wrapper?
--------------

The project's ``scripts/skrl/train.py`` imports ``isaac_robots.tasks`` at
module top-level, which auto-registers all the project's gym IDs via
``isaaclab_tasks.utils.import_packages``. But ``velgate_bench.envs`` lives
outside that auto-discovery tree, so its IDs (``Velgate-Bench-...``) are
not visible to gym.spec when the hydra wrapper resolves the ``--task``
argument.

Two options to fix:
  (a) Modify ``isaac_robots.tasks.__init__`` to opt-in load
      ``velgate_bench.envs``. Cleanest but couples the main project to the
      benchmark.
  (b) Wrap ``train.py`` so the benchmark loads its own envs before the
      hydra lookup. This file.

This wrapper uses a post-import hook on ``isaac_robots.tasks``: when train.py
imports it, the hook fires our ``register_all()`` right after. After
registration, control returns to train.py untouched.
"""

from __future__ import annotations

import builtins
import re
import runpy
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BENCH_DIR = _REPO_ROOT / "benchmarks" / "velocity_gated_reward"

# Make velgate_bench importable.
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))


# ---------------------------------------------------------------------------
# Banner: print a big visible header at the start of every train.log so the
# user can tell at a glance which (task, kernel, seed) cell is running. The
# banner is parsed from sys.argv (--task, --seed) before train.py touches it.
# ---------------------------------------------------------------------------

_KERNEL_SUMMARY = {
    "Tanh": (
        "tanh",
        "R(d) = (1 - tanh(d / 0.05))",
        "BASELINE - tanh has nonzero gradient at d=0; expect limit cycle at goal hold",
    ),
    "Gaussian": (
        "gaussian",
        "R(d) = exp(-(d / 0.05)^2)",
        "AVOID - zero gradient at d=0; expect late-stage drift after peak",
    ),
    "VelocityGatedTanh": (
        "velocity_gated_tanh",
        "R(d, qdot) = (1 - tanh(d/0.05)) * inside_gate(d<0.10) * clip(1 - |qdot|/0.5, 0, 1)",
        "OUR PROPOSAL - tanh centering pull preserved + velocity gate penalizes own oscillation",
    ),
    "TanhAdditiveVelPen": (
        "tanh_additive_velpen",
        "R(d, qdot) = (1 - tanh(d/0.05)) - 1.0 * 1[d<0.10] * clip(|qdot|/0.5, 0, 1)",
        "BASELINE - additive counterpart of the multiplicative velocity gate",
    ),
    "VelocityGatedTanhSmooth": (
        "velocity_gated_tanh_smooth",
        "R(d, qdot) = (1 - tanh(d/0.05)) * G_smooth, gate blend sigmoid((0.10 - d)/0.02)",
        "ABLATION - velocity gate with smooth (sigmoid) neighborhood boundary",
    ),
}


def _print_banner() -> None:
    argv = sys.argv
    task = None
    seed = None
    num_envs = None
    for i, a in enumerate(argv):
        if a == "--task" and i + 1 < len(argv):
            task = argv[i + 1]
        elif a == "--seed" and i + 1 < len(argv):
            seed = argv[i + 1]
        elif a == "--num_envs" and i + 1 < len(argv):
            num_envs = argv[i + 1]

    if not task:
        return  # not a sweep cell; skip banner

    # Extract robot, kernel from task name. e.g. Velgate-Bench-Franka-Lift-Tanh-v0
    # NOTE: longer alternatives must come first, otherwise VelocityGatedTanhSmooth
    # would partially match VelocityGatedTanh and leave "-Smooth-v0" dangling.
    kernel_alt = "|".join(sorted(_KERNEL_SUMMARY, key=len, reverse=True))
    m = re.match(rf"Velgate-Bench-(.+?)-Lift-({kernel_alt})-v0", task)
    if not m:
        return
    robot_display, kernel_display = m.group(1), m.group(2)
    kernel_name, formula, expected = _KERNEL_SUMMARY[kernel_display]

    line = "=" * 80
    bar = "+" + "-" * 78 + "+"
    print(line, file=sys.stderr, flush=True)
    print(f"  VELGATE BENCH CELL", file=sys.stderr, flush=True)
    print(bar, file=sys.stderr, flush=True)
    print(f"  Task         : {task}", file=sys.stderr, flush=True)
    print(f"  Robot        : {robot_display}", file=sys.stderr, flush=True)
    print(f"  KERNEL       : {kernel_name}   ({kernel_display})", file=sys.stderr, flush=True)
    print(f"  Seed         : {seed}", file=sys.stderr, flush=True)
    print(f"  num_envs     : {num_envs}", file=sys.stderr, flush=True)
    print(f"  Reward fn    : {formula}", file=sys.stderr, flush=True)
    print(f"  Expectation  : {expected}", file=sys.stderr, flush=True)
    print(line, file=sys.stderr, flush=True)


_print_banner()

# Install a post-import hook on isaac_robots.tasks. After that module
# finishes loading, we trigger our benchmark's gym registrations. The hook
# fires once and then short-circuits.
_REGISTERED = False
_original_import = builtins.__import__


def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
    global _REGISTERED
    module = _original_import(name, globals, locals, fromlist, level)
    # `import isaac_robots.tasks` passes name="isaac_robots.tasks", but
    # `from isaac_robots.tasks import X` may pass name="isaac_robots" with
    # fromlist=("tasks",). Either way, by the time builtins.__import__
    # returns, the submodule is in sys.modules.
    if not _REGISTERED and "isaac_robots.tasks" in sys.modules:
        # Set _REGISTERED BEFORE the import so the hook doesn't re-enter
        # while velgate_bench.envs is mid-loading (which would trigger
        # "partially initialized module" import errors as its submodules
        # transitively re-trigger the hook).
        _REGISTERED = True
        try:
            from velgate_bench.envs import register_all
            register_all()
            print("[velgate-train-wrapper] Registered velgate_bench env variants.",
                  file=sys.stderr)
        except Exception as exc:
            print(f"[velgate-train-wrapper] WARN: failed to register velgate envs: {exc}",
                  file=sys.stderr)
    return module


builtins.__import__ = _patched_import

# Delegate to the project's train.py.
_target = _REPO_ROOT / "scripts" / "skrl" / "train.py"
runpy.run_path(str(_target), run_name="__main__")
