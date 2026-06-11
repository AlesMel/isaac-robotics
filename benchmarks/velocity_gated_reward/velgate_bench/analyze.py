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
    """Compute peak/final/drift metrics from a TB log.

    Definitions (robust against curve noise and sparse logging):

    - The curve is first smoothed with a centered rolling mean over ~5% of
      the points. Taking ``max`` of the RAW curve would be biased upward
      (max-of-noise), inflating drift even for a flat noisy curve.
    - ``peak_reward``: max of the smoothed curve.
    - ``final_reward``: mean of smoothed values in the last 10% of
      TIMESTEPS (not points -- sparse logs once made "last 10% of points"
      a single sample).
    - ``drift_ratio``: (peak - final) / peak.
    - Curves with fewer than 20 points cannot support these estimates;
      drift is reported as NaN rather than a silently garbage value.
    """
    steps, vals = parse_tb_reward_curve(log_dir)
    n = int(len(vals))
    if n == 0:
        return {"peak_reward": float("nan"), "final_reward": float("nan"),
                "drift_ratio": float("nan"), "n_points": 0}
    if n < 20:
        return {"peak_reward": float(np.max(vals)), "final_reward": float(vals[-1]),
                "drift_ratio": float("nan"), "n_points": n}
    w = max(1, round(0.05 * n))
    smooth = np.convolve(vals, np.ones(w) / w, mode="valid")
    smooth_steps = steps[(w - 1) // 2:(w - 1) // 2 + len(smooth)]
    peak = float(np.max(smooth))
    tail = smooth[smooth_steps >= 0.9 * steps.max()]
    final = float(np.mean(tail)) if len(tail) else float(smooth[-1])
    drift = (peak - final) / peak if peak > 0 else 0.0
    return {"peak_reward": peak, "final_reward": final, "drift_ratio": float(drift),
            "n_points": n}


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
        # Per-episode key is "hold_obj_goal_dist_mean"; aggregate_metrics in
        # metrics.py suffixes it again, hence the double "_mean".
        "eval_hold_obj_goal_dist_mean": eval_data.get("hold_obj_goal_dist_mean_mean", float("nan")),
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
            if key in _HEADLINE_METRICS:
                row[f"{key}_iqm"] = iqm(vals)
                ci_lo, ci_hi = bootstrap_ci(vals)
                row[f"{key}_ci_lo"] = ci_lo
                row[f"{key}_ci_hi"] = ci_hi
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


# Headline metrics that additionally get IQM + 95% bootstrap CI columns in
# the aggregate table (rliable-style robust aggregates over seeds).
_HEADLINE_METRICS = ("eval_success_mean", "eval_hold_joint_vel_l2_mean", "tb_drift_ratio")


def iqm(values) -> float:
    """Interquartile mean: mean of the middle 50% of sorted values.

    More robust than the mean for small-n RL seed aggregates (Agarwal et al.,
    "Deep RL at the Edge of the Statistical Precipice", NeurIPS 2021).
    """
    v = np.sort(np.asarray(values, dtype=float))
    n = len(v)
    if n == 0:
        return float("nan")
    lo, hi = int(np.floor(n * 0.25)), int(np.ceil(n * 0.75))
    trimmed = v[lo:hi]
    return float(np.mean(trimmed)) if len(trimmed) else float(np.mean(v))


def bootstrap_ci(values, stat=iqm, n_boot: int = 10_000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for ``stat`` over ``values`` (seeded for repro)."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    samples = np.asarray([stat(v[row]) for row in idx])
    return (float(np.quantile(samples, alpha / 2)),
            float(np.quantile(samples, 1 - alpha / 2)))


def holm_bonferroni(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (step-down), order-preserving."""
    m = len(pvals)
    if m == 0:
        return []
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


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


# The pairwise test family. Every (comparison, task) p-value computed from
# these specs forms ONE family for Holm-Bonferroni correction; verdicts in
# results.md use the adjusted p. Eval-rollout metrics are primary -- training
# reward is kernel-specific (each kernel CHANGES the reward function) and is
# reported descriptively only.
COMPARISONS: list[dict] = [
    dict(title="H1a: Success rate, velgated vs tanh (higher is better)",
         metric="eval_success_mean", candidate="velocity_gated_tanh", baseline="tanh", lower_is_better=False),
    dict(title="H1a: Success rate, velgated vs gaussian (higher is better)",
         metric="eval_success_mean", candidate="velocity_gated_tanh", baseline="gaussian", lower_is_better=False),
    dict(title="H1a: Hold distance to goal (m), velgated vs tanh (lower is better)",
         metric="eval_hold_obj_goal_dist_mean", candidate="velocity_gated_tanh", baseline="tanh", lower_is_better=True),
    dict(title="H1a: Time at goal (s), velgated vs tanh (higher is better)",
         metric="eval_time_at_goal_s_mean", candidate="velocity_gated_tanh", baseline="tanh", lower_is_better=False),
    dict(title="H1b: Hold joint velocity (rad/s), velgated vs tanh (lower is better)",
         metric="eval_hold_joint_vel_l2_mean", candidate="velocity_gated_tanh", baseline="tanh", lower_is_better=True),
    dict(title="H1c: Drift ratio, velgated vs gaussian (lower is better)",
         metric="tb_drift_ratio", candidate="velocity_gated_tanh", baseline="gaussian", lower_is_better=True),
    dict(title="Baseline: Success rate, velgated vs additive velocity penalty (higher is better)",
         metric="eval_success_mean", candidate="velocity_gated_tanh", baseline="tanh_additive_velpen", lower_is_better=False),
    dict(title="Baseline: Hold joint velocity, velgated vs additive velocity penalty (lower is better)",
         metric="eval_hold_joint_vel_l2_mean", candidate="velocity_gated_tanh", baseline="tanh_additive_velpen", lower_is_better=True),
    dict(title="Ablation: Success rate, smooth vs hard gate (higher is better)",
         metric="eval_success_mean", candidate="velocity_gated_tanh_smooth", baseline="velocity_gated_tanh", lower_is_better=False),
    dict(title="Ablation: Hold joint velocity, smooth vs hard gate (lower is better)",
         metric="eval_hold_joint_vel_l2_mean", candidate="velocity_gated_tanh_smooth", baseline="velocity_gated_tanh", lower_is_better=True),
]


def run_comparisons(cells: list[dict]) -> list[dict]:
    """Compute all COMPARISONS, then apply Holm-Bonferroni across the family.

    Returns the spec list with a ``stats`` dict (per task) attached; each
    non-skipped entry gains a ``p_holm`` field.
    """
    results = []
    flat_refs: list[dict] = []
    for spec in COMPARISONS:
        stats = pairwise_stats(cells, spec["metric"],
                               baseline_kernel=spec["baseline"],
                               candidate_kernel=spec["candidate"])
        results.append({**spec, "stats": stats})
        for st in stats.values():
            if not st.get("skipped") and not _is_nan(st.get("p_value", float("nan"))):
                flat_refs.append(st)
    adjusted = holm_bonferroni([st["p_value"] for st in flat_refs])
    for st, p_adj in zip(flat_refs, adjusted):
        st["p_holm"] = p_adj
    return results


def write_results_md(
    aggregates: list[dict],
    comparisons: list[dict],
    out_path: Path,
) -> None:
    lines = ["# Velocity-Gated Reward Benchmark — Results\n"]
    lines.append("*Auto-generated by `velgate_bench.analyze`. Do not edit by hand.*\n")

    lines.append("## Aggregate (mean ± std across seeds)\n")
    lines.append("*Peak reward is in kernel-specific units (each kernel changes the "
                 "reward function) — do not compare it across kernels; eval columns "
                 "are kernel-independent.*\n")
    lines.append("| Task | Kernel | n | Peak reward† | Drift ratio | Hold joint vel (L2, rad/s) | Success rate |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in sorted(aggregates, key=lambda x: (x["task"], x["kernel"])):
        lines.append(
            f"| {r['task']} | {r['kernel']} | {r['n_seeds']} | "
            f"{r.get('tb_peak_reward_mean', float('nan')):.1f} ± {r.get('tb_peak_reward_std', 0):.1f} | "
            f"{r.get('tb_drift_ratio_mean', float('nan')):.3f} ± {r.get('tb_drift_ratio_std', 0):.3f} | "
            f"{r.get('eval_hold_joint_vel_l2_mean_mean', float('nan')):.3f} ± {r.get('eval_hold_joint_vel_l2_mean_std', 0):.3f} | "
            f"{r.get('eval_success_mean_mean', float('nan')):.2f} ± {r.get('eval_success_mean_std', 0):.2f} |"
        )
    lines.append("\n† kernel-specific units; descriptive only.\n")

    lines.append("## Robust aggregates — IQM [95% bootstrap CI] over seeds\n")
    lines.append("*Interquartile mean with seeded percentile bootstrap (B=10 000); "
                 "robust to outlier seeds at small n.*\n")
    lines.append("| Task | Kernel | Success rate | Hold joint vel (rad/s) | Drift ratio |")
    lines.append("|---|---|---|---|---|")
    for r in sorted(aggregates, key=lambda x: (x["task"], x["kernel"])):
        def _fmt_iqm(key: str, fmt: str) -> str:
            v = r.get(f"{key}_iqm", float("nan"))
            lo = r.get(f"{key}_ci_lo", float("nan"))
            hi = r.get(f"{key}_ci_hi", float("nan"))
            return f"{v:{fmt}} [{lo:{fmt}}, {hi:{fmt}}]"
        lines.append(
            f"| {r['task']} | {r['kernel']} | "
            f"{_fmt_iqm('eval_success_mean', '.2f')} | "
            f"{_fmt_iqm('eval_hold_joint_vel_l2_mean', '.3f')} | "
            f"{_fmt_iqm('tb_drift_ratio', '.3f')} |"
        )

    lines.append("\n## Pairwise comparisons (Welch's t-test, two-sided)\n")
    lines.append("*p_holm = Holm-Bonferroni adjusted p across the whole test family; "
                 "verdicts use p_holm < 0.05.*\n")
    for comp in comparisons:
        lines.append(f"\n### {comp['title']}\n")
        lines.append(f"*candidate = `{comp['candidate']}`, baseline = `{comp['baseline']}`, "
                     f"metric = `{comp['metric']}`*\n")
        lines.append("| Task | n | Mean candidate | Mean baseline | t | p | p_holm | Cohen's d | Verdict |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for task, st in comp["stats"].items():
            if st.get("skipped"):
                lines.append(f"| {task} | {st.get('n_candidate', 0)}/{st.get('n_baseline', 0)} "
                             f"| — | — | — | — | — | — | (insufficient n) |")
                continue
            diff = st["candidate_mean"] - st["baseline_mean"]
            improved = (diff < 0) if comp["lower_is_better"] else (diff > 0)
            p_holm = st.get("p_holm", float("nan"))
            significant = p_holm < 0.05
            verdict = (
                "✓ significant improvement" if (improved and significant) else
                "✗ significant regression" if (not improved and significant) else
                "no significant difference"
            )
            lines.append(
                f"| {task} | {st['n_candidate']}/{st['n_baseline']} | "
                f"{st['candidate_mean']:.3f} ± {st['candidate_std']:.3f} | "
                f"{st['baseline_mean']:.3f} ± {st['baseline_std']:.3f} | "
                f"{st['t_statistic']:.2f} | {st['p_value']:.4f} | {p_holm:.4f} | "
                f"{st['cohens_d']:.2f} | {verdict} |"
            )

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
    kernels = sorted({r["kernel"] for r in aggregates})
    x = np.arange(len(tasks))
    width = 0.8 / max(1, len(kernels))

    fig, ax = plt.subplots(figsize=(8, 4))
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
    ax.set_xticks(x + 0.4 - width / 2)
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
    """Scatter success rate (y) vs hold_joint_vel_l2 (x), colored by kernel.

    Both axes are kernel-independent eval metrics (training reward would be
    incommensurable across kernels).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    kernels = sorted({c["kernel"] for c in cells})
    for i, kernel in enumerate(kernels):
        pts = [
            (c["eval_hold_joint_vel_l2_mean"], c["eval_success_mean"])
            for c in cells
            if c["kernel"] == kernel
            and not _is_nan(c.get("eval_hold_joint_vel_l2_mean", float("nan")))
            and not _is_nan(c.get("eval_success_mean", float("nan")))
        ]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, c=f"C{i}", label=kernel, alpha=0.7, s=60)
    ax.set_xlabel("Hold joint velocity (L2, rad/s) — lower is better")
    ax.set_ylabel("Success rate — higher is better")
    ax.set_title("Pareto: hold stability vs. task success\n(top-left = best)")
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

    # Stats: full comparison family + Holm-Bonferroni correction across it
    comparisons = run_comparisons(cells)

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
             ylabel="Peak reward (kernel-specific units)",
             title="Peak training reward by kernel (descriptive only)",
             out_path=plots_dir / "peak_reward_by_kernel.png")
    plot_bar(aggregates, "eval_success_mean_mean", "eval_success_mean_std",
             ylabel="Success rate", title="Eval success rate by kernel",
             out_path=plots_dir / "success_by_kernel.png")
    plot_pareto(cells, plots_dir)

    # results.md
    docs_dir = _BENCH_DIR / "docs"
    write_results_md(aggregates, comparisons, docs_dir / "results.md")
    print(f"[velgate-analyze] wrote {docs_dir}/results.md")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
