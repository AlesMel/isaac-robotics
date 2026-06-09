# Copyright (c) 2026, Ales Melichar
# SPDX-License-Identifier: BSD-3-Clause

"""Analysis pipeline for the velocity-gated reward benchmark.

Given the ``results/`` directory populated by ``sweep.py``, produces:

  1. Per-cell TB metric summary (peak/final reward, drift ratio).
  2. Aggregated tables (mean ± std across seeds) → CSV + LaTeX.
  3. Publication plots:
     - Training curves (reward over timesteps, lines = kernel, shaded = ±std)
     - Drift ratio bar chart (kernel × task)
     - Hold joint velocity bar chart (kernel × task)
     - Pareto scatter (peak_reward vs hold_joint_vel_l2)
  4. Statistical tests (Welch's t-test, Cohen's d for vel-gated vs tanh/gaussian).
  5. A regenerated ``docs/results.md`` with all numbers embedded.

Usage::

    python -m velgate_bench.analyze --results-dir results/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

_BENCH_DIR = Path(__file__).resolve().parents[1]
_RESULTS_DIR = _BENCH_DIR / "results"


# -----------------------------------------------------------------------------
# TB log parsing
# -----------------------------------------------------------------------------


def find_tb_log_dir(run_dir: Path) -> Path | None:
    """Find the skrl nested timestamp directory containing event files."""
    for cand in run_dir.rglob("events.out.tfevents.*"):
        return cand.parent
    return None


def parse_tb_reward_curve(log_dir: Path, tag: str = "Reward / Total reward (mean)") -> tuple[np.ndarray, np.ndarray]:
    """Return (steps, values) for ``tag`` from the TB log."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return np.array([]), np.array([])
    events = ea.Scalars(tag)
    steps = np.asarray([e.step for e in events], dtype=int)
    vals = np.asarray([e.value for e in events], dtype=float)
    return steps, vals


def summarize_tb(log_dir: Path) -> dict:
    """Compute peak/final/drift metrics from a TB log."""
    steps, vals = parse_tb_reward_curve(log_dir)
    if len(vals) == 0:
        return {"peak_reward": float("nan"), "final_reward": float("nan"),
                "drift_ratio": float("nan"), "n_points": 0}
    peak = float(np.max(vals))
    tail_n = max(1, len(vals) // 10)
    final = float(np.mean(vals[-tail_n:]))
    drift = (peak - final) / peak if peak > 0 else 0.0
    return {"peak_reward": peak, "final_reward": final, "drift_ratio": float(drift),
            "n_points": int(len(vals))}


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------


def load_cell(task: str, kernel: str, seed: int, results_dir: Path) -> dict:
    """Return all per-cell metrics: TB summary + eval rollout summary."""
    run_dir = results_dir / "runs" / task / kernel / f"seed_{seed}"
    log_dir = find_tb_log_dir(run_dir)
    if log_dir is None:
        tb = {"peak_reward": float("nan"), "final_reward": float("nan"),
              "drift_ratio": float("nan"), "n_points": 0}
    else:
        tb = summarize_tb(log_dir)

    eval_path = results_dir / "eval" / task / kernel / f"seed_{seed}.json"
    if eval_path.exists():
        with open(eval_path) as f:
            eval_data = json.load(f)
    else:
        eval_data = {}

    return {
        "task": task,
        "kernel": kernel,
        "seed": seed,
        **{f"tb_{k}": v for k, v in tb.items()},
        "eval_hold_joint_vel_l2_mean": eval_data.get("hold_joint_vel_l2_mean", float("nan")),
        "eval_hold_joint_vel_l2_std": eval_data.get("hold_joint_vel_l2_std", float("nan")),
        "eval_hold_ee_z_std_mean": eval_data.get("hold_ee_z_std_mean", float("nan")),
        "eval_success_mean": eval_data.get("success_mean", float("nan")),
        "eval_time_at_goal_s_mean": eval_data.get("time_at_goal_s_mean", float("nan")),
    }


def aggregate_across_seeds(cells: list[dict]) -> list[dict]:
    """Aggregate per-seed cells -> per-(task, kernel) rows with mean / std / n."""
    grouped: dict[tuple[str, str], list[dict]] = {}
    for c in cells:
        key = (c["task"], c["kernel"])
        grouped.setdefault(key, []).append(c)

    rows = []
    for (task, kernel), seeds in grouped.items():
        row = {"task": task, "kernel": kernel, "n_seeds": len(seeds)}
        # Aggregate every numeric column
        any_cell = seeds[0]
        for key in any_cell:
            if key in ("task", "kernel", "seed"):
                continue
            vals = np.asarray([s[key] for s in seeds if not _is_nan(s[key])], dtype=float)
            row[f"{key}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
            row[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return rows


def _is_nan(v) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


# -----------------------------------------------------------------------------
# Statistical tests
# -----------------------------------------------------------------------------


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled = math.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    if pooled == 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled)


def welch_t_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return (t_statistic, two_sided_p_value) for Welch's t-test."""
    from scipy.stats import ttest_ind

    res = ttest_ind(a, b, equal_var=False)
    return float(res.statistic), float(res.pvalue)


def pairwise_stats(cells: list[dict], metric_key: str, baseline_kernel: str, candidate_kernel: str) -> dict:
    """Per-task comparison between candidate vs baseline kernel on `metric_key`."""
    by_task: dict[str, dict[str, list[float]]] = {}
    for c in cells:
        if c["kernel"] not in (baseline_kernel, candidate_kernel):
            continue
        v = c.get(metric_key, float("nan"))
        if _is_nan(v):
            continue
        by_task.setdefault(c["task"], {}).setdefault(c["kernel"], []).append(float(v))

    out = {}
    for task, by_kernel in by_task.items():
        a = np.asarray(by_kernel.get(candidate_kernel, []))
        b = np.asarray(by_kernel.get(baseline_kernel, []))
        if len(a) < 2 or len(b) < 2:
            out[task] = {"n_candidate": len(a), "n_baseline": len(b), "skipped": True}
            continue
        t, p = welch_t_test(a, b)
        d = cohens_d(a, b)
        out[task] = {
            "n_candidate": len(a), "n_baseline": len(b),
            "candidate_mean": float(np.mean(a)), "candidate_std": float(np.std(a, ddof=1)),
            "baseline_mean": float(np.mean(b)), "baseline_std": float(np.std(b, ddof=1)),
            "t_statistic": t, "p_value": p, "cohens_d": d,
        }
    return out


# -----------------------------------------------------------------------------
# Output writers (CSV, LaTeX, results.md)
# -----------------------------------------------------------------------------


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_results_md(
    aggregates: list[dict],
    stats_peak: dict,
    stats_drift: dict,
    stats_hold_vel: dict,
    out_path: Path,
) -> None:
    lines = ["# Velocity-Gated Reward Benchmark — Results\n"]
    lines.append("*Auto-generated by `velgate_bench.analyze`. Do not edit by hand.*\n")

    lines.append("## Aggregate (mean ± std across seeds)\n")
    lines.append("| Task | Kernel | n | Peak reward | Drift ratio | Hold joint vel (L2, rad/s) | Success rate |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in sorted(aggregates, key=lambda x: (x["task"], x["kernel"])):
        lines.append(
            f"| {r['task']} | {r['kernel']} | {r['n_seeds']} | "
            f"{r.get('tb_peak_reward_mean', float('nan')):.1f} ± {r.get('tb_peak_reward_std', 0):.1f} | "
            f"{r.get('tb_drift_ratio_mean', float('nan')):.3f} ± {r.get('tb_drift_ratio_std', 0):.3f} | "
            f"{r.get('eval_hold_joint_vel_l2_mean_mean', float('nan')):.3f} ± {r.get('eval_hold_joint_vel_l2_mean_std', 0):.3f} | "
            f"{r.get('eval_success_mean_mean', float('nan')):.2f} ± {r.get('eval_success_mean_std', 0):.2f} |"
        )

    def _fmt_stats(stats: dict, candidate: str, baseline: str, lower_is_better: bool) -> list[str]:
        out = [f"\n### {candidate} vs {baseline}\n"]
        out.append("| Task | n | Mean candidate | Mean baseline | t | p | Cohen's d | Verdict |")
        out.append("|---|---|---|---|---|---|---|---|")
        for task, st in stats.items():
            if st.get("skipped"):
                out.append(f"| {task} | — | — | — | — | — | — | (insufficient n) |")
                continue
            diff = st["candidate_mean"] - st["baseline_mean"]
            improved = (diff < 0) if lower_is_better else (diff > 0)
            significant = st["p_value"] < 0.05
            verdict = (
                "✓ significant improvement" if (improved and significant) else
                "✗ significant regression" if (not improved and significant) else
                "no significant difference"
            )
            out.append(
                f"| {task} | {st['n_candidate']}/{st['n_baseline']} | "
                f"{st['candidate_mean']:.3f} ± {st['candidate_std']:.3f} | "
                f"{st['baseline_mean']:.3f} ± {st['baseline_std']:.3f} | "
                f"{st['t_statistic']:.2f} | {st['p_value']:.4f} | {st['cohens_d']:.2f} | {verdict} |"
            )
        return out

    lines.append("\n## Pairwise comparisons (Welch's t-test, two-sided)\n")
    lines.append("\n### Peak reward (higher is better)\n")
    lines.extend(_fmt_stats(stats_peak, "velocity_gated_tanh", "tanh", lower_is_better=False))
    lines.append("\n### Drift ratio (lower is better)\n")
    lines.extend(_fmt_stats(stats_drift, "velocity_gated_tanh", "gaussian", lower_is_better=True))
    lines.append("\n### Hold joint velocity (lower is better)\n")
    lines.extend(_fmt_stats(stats_hold_vel, "velocity_gated_tanh", "tanh", lower_is_better=True))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------


def plot_training_curves(cells: list[dict], results_dir: Path, plots_dir: Path) -> None:
    """One PNG per task: x=timestep, y=reward, lines=kernel, shaded=±std across seeds."""
    import matplotlib.pyplot as plt

    by_task: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = {}
    for c in cells:
        run_dir = results_dir / "runs" / c["task"] / c["kernel"] / f"seed_{c['seed']}"
        log_dir = find_tb_log_dir(run_dir)
        if log_dir is None:
            continue
        steps, vals = parse_tb_reward_curve(log_dir)
        if len(steps) == 0:
            continue
        by_task.setdefault(c["task"], {}).setdefault(c["kernel"], []).append((steps, vals))

    for task, by_kernel in by_task.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        for kernel, runs in by_kernel.items():
            steps_ref = runs[0][0]
            stacked = np.stack([np.interp(steps_ref, s, v) for s, v in runs])
            mean = stacked.mean(axis=0)
            std = stacked.std(axis=0, ddof=1) if stacked.shape[0] > 1 else np.zeros_like(mean)
            ax.plot(steps_ref, mean, label=kernel, linewidth=2)
            ax.fill_between(steps_ref, mean - std, mean + std, alpha=0.2)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Episode reward (mean across envs)")
        ax.set_title(f"Training curves — {task}")
        ax.legend()
        ax.grid(alpha=0.3)
        out_path = plots_dir / f"training_curves_{task}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[velgate-analyze] wrote {out_path}")


def plot_bar(aggregates: list[dict], metric_mean_key: str, metric_std_key: str,
              ylabel: str, title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    tasks = sorted({r["task"] for r in aggregates})
    kernels = ["tanh", "gaussian", "velocity_gated_tanh"]
    x = np.arange(len(tasks))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, kernel in enumerate(kernels):
        means, stds = [], []
        for task in tasks:
            match = [r for r in aggregates if r["task"] == task and r["kernel"] == kernel]
            if not match:
                means.append(float("nan"))
                stds.append(0)
            else:
                means.append(match[0].get(metric_mean_key, float("nan")))
                stds.append(match[0].get(metric_std_key, 0))
        ax.bar(x + i * width, means, width, yerr=stds, capsize=4, label=kernel)
    ax.set_xticks(x + width)
    ax.set_xticklabels(tasks)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[velgate-analyze] wrote {out_path}")


def plot_pareto(cells: list[dict], plots_dir: Path) -> None:
    """Scatter peak_reward (y) vs hold_joint_vel_l2 (x), colored by kernel."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {"tanh": "C0", "gaussian": "C1", "velocity_gated_tanh": "C2"}
    for kernel in colors:
        xs = [c["eval_hold_joint_vel_l2_mean"] for c in cells if c["kernel"] == kernel and not _is_nan(c.get("eval_hold_joint_vel_l2_mean", float("nan")))]
        ys = [c["tb_peak_reward"] for c in cells if c["kernel"] == kernel and not _is_nan(c.get("tb_peak_reward", float("nan")))]
        if not xs:
            continue
        ax.scatter(xs, ys, c=colors[kernel], label=kernel, alpha=0.7, s=60)
    ax.set_xlabel("Hold joint velocity (L2, rad/s) — lower is better")
    ax.set_ylabel("Peak reward — higher is better")
    ax.set_title("Pareto: hold stability vs. peak performance\n(top-left = best)")
    ax.legend()
    ax.grid(alpha=0.3)
    out_path = plots_dir / "pareto_hold_vs_peak.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[velgate-analyze] wrote {out_path}")


# -----------------------------------------------------------------------------
# Entry
# -----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze velocity-gated reward benchmark")
    parser.add_argument("--results-dir", type=Path, default=_RESULTS_DIR)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--kernels", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    args = parser.parse_args(argv)

    results_dir = args.results_dir
    runs_dir = results_dir / "runs"
    if not runs_dir.exists():
        print(f"[velgate-analyze] No runs at {runs_dir}. Did you run sweep.py?", file=__import__("sys").stderr)
        return 1

    # Discover cells from the runs/ tree
    tasks = args.tasks or sorted(p.name for p in runs_dir.iterdir() if p.is_dir())
    cells = []
    for task in tasks:
        kernel_dir = runs_dir / task
        kernels = args.kernels or sorted(p.name for p in kernel_dir.iterdir() if p.is_dir())
        for kernel in kernels:
            seed_dir = kernel_dir / kernel
            seeds = args.seeds or sorted(int(p.name.split("_")[-1]) for p in seed_dir.iterdir() if p.name.startswith("seed_"))
            for seed in seeds:
                cells.append(load_cell(task, kernel, seed, results_dir))

    if not cells:
        print("[velgate-analyze] No cells found.", file=__import__("sys").stderr)
        return 1

    aggregates = aggregate_across_seeds(cells)

    # Tables
    tables_dir = results_dir / "tables"
    write_csv(cells, tables_dir / "per_seed.csv")
    write_csv(aggregates, tables_dir / "aggregates.csv")
    print(f"[velgate-analyze] wrote {tables_dir}/per_seed.csv and aggregates.csv")

    # Stats
    stats_peak = pairwise_stats(cells, "tb_peak_reward", baseline_kernel="tanh", candidate_kernel="velocity_gated_tanh")
    stats_drift = pairwise_stats(cells, "tb_drift_ratio", baseline_kernel="gaussian", candidate_kernel="velocity_gated_tanh")
    stats_hold_vel = pairwise_stats(cells, "eval_hold_joint_vel_l2_mean", baseline_kernel="tanh", candidate_kernel="velocity_gated_tanh")

    # Plots
    plots_dir = results_dir / "plots"
    plot_training_curves(cells, results_dir, plots_dir)
    plot_bar(aggregates, "tb_drift_ratio_mean", "tb_drift_ratio_std",
             ylabel="Drift ratio (peak − final) / peak", title="Late-stage training drift by kernel",
             out_path=plots_dir / "drift_by_kernel.png")
    plot_bar(aggregates, "eval_hold_joint_vel_l2_mean_mean", "eval_hold_joint_vel_l2_mean_std",
             ylabel="Hold joint velocity (L2, rad/s)", title="Goal-hold stability by kernel",
             out_path=plots_dir / "hold_vel_by_kernel.png")
    plot_bar(aggregates, "tb_peak_reward_mean", "tb_peak_reward_std",
             ylabel="Peak reward", title="Peak training reward by kernel",
             out_path=plots_dir / "peak_reward_by_kernel.png")
    plot_pareto(cells, plots_dir)

    # results.md
    docs_dir = _BENCH_DIR / "docs"
    write_results_md(aggregates, stats_peak, stats_drift, stats_hold_vel, docs_dir / "results.md")
    print(f"[velgate-analyze] wrote {docs_dir}/results.md")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
