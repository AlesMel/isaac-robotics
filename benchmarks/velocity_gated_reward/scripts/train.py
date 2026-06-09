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
import runpy
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BENCH_DIR = _REPO_ROOT / "benchmarks" / "velocity_gated_reward"

# Make velgate_bench importable.
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

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
