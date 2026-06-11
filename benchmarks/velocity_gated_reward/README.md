# Velocity-Gated Reward Shaping for PPO Continuous-Control "Reach-and-Hold" Tasks

> Benchmark accompanying the claim: **velocity-gated tanh** reward kernels
> outperform both vanilla **tanh** (which produces a limit cycle at the goal
> pose) and **Gaussian** (which produces late-stage policy drift) across PPO
> manipulation tasks with a "reach + hold at goal" requirement.

---

## TL;DR

For PPO continuous-control tasks where the policy must **reach** a goal and
**hold** there:

| Kernel | At goal: gradient ∂R/∂d | Failure mode |
|---|---|---|
| `1 - tanh(d/σ)` (upstream baseline) | -1/σ (nonzero) | **Limit cycle**: policy keeps trying to "improve" → noise amplification → visible oscillation |
| `exp(-(d/σ)²)` (Gaussian) | 0 | **Drift**: no anchor → noise dispersion → policy slowly degrades after peak |
| **`(1 - tanh(d/σ)) × velgate(q̇)`** (ours) | -1/σ × velgate (preserved) | **None observed**: centering pull intact, plus the gate penalizes the policy's own corrective motion |

Where `velgate(q̇) = clip(1 - ‖q̇‖_arm / v_thresh, 0, 1)` is applied **only
inside the goal neighborhood** (distance < r_neighborhood). Outside the
neighborhood, the reward is identical to vanilla tanh — so reach behavior
is unchanged. Inside, the gate makes the reward sensitive to the policy's
**own oscillation** rather than to micro-perturbations from the environment.

The benchmark additionally runs two hardening conditions (see
`docs/methodology.md`):

- **`tanh_additive_velpen`** — the *additive* counterpart (tanh tracking
  minus a distance-gated velocity penalty, same parameters). The key
  competing baseline: is multiplicative gating actually better than a plain
  penalty?
- **`velocity_gated_tanh_smooth`** — sigmoid-blended gate boundary
  (ablation of the hard `d < r` indicator; Franka only).

---

## What this benchmark contains

```
benchmarks/velocity_gated_reward/
├── README.md                 # this file
├── docs/
│   ├── claim.md              # formal hypothesis + sub-claims
│   ├── methodology.md        # experimental design, metrics, statistics
│   └── results.md            # filled after sweep runs (auto-generated section)
├── velgate_bench/
│   ├── kernels.py            # the five reward functions (parameterized)
│   ├── envs/                 # task wrappers (Franka lift, UR3e Hand-E lift, Franka reach)
│   ├── metrics.py            # eval-rollout metrics (hold velocity, drift, success)
│   ├── sweep.py              # orchestrator: runs (task × kernel × seed) cartesian product
│   └── analyze.py            # TB log aggregation + plot/table generation
├── configs/                  # YAML configs for sweep matrices and hyperparameters
├── scripts/                  # shell wrappers (run_sweep.sh, make_figures.sh)
└── results/                  # outputs (.gitignored except .gitkeep)
    ├── runs/                 # per-run TB logs + checkpoints
    ├── eval/                 # per-run eval rollout data
    ├── plots/                # generated figures
    └── tables/               # generated tables (CSV + LaTeX)
```

## Quick start

```bash
conda activate env_isaaclab
cd benchmarks/velocity_gated_reward

# 0. SMOKE TEST first (~75 min): all 5 kernels x 1 seed at a tiny budget.
#    Gates the hydra overrides, both new kernels, eval keys, and analyze.
bash scripts/run_sweep.sh configs/sweep_smoke.yaml
python -m velgate_bench.analyze --results-dir results/
#    ...then archive smoke output before the real matrix:
mv results/runs results/_archive_smoke && rm -rf results/eval results/plots results/tables

# 1. Phase 1: Franka (25 cells, ~3 days)
bash scripts/run_sweep.sh configs/sweep_franka.yaml

# 2. Phase 2: UR3e (20 cells, ~3-5 days), after Franka completes
bash scripts/run_sweep.sh configs/sweep_ur3e.yaml

# 3. Eval catch-up pass (idempotent; re-runs any cell whose eval failed)
bash scripts/run_eval.sh

# 4. Generate publication-ready plots and tables
python -m velgate_bench.analyze --results-dir results/

# 5. Read the results
open docs/results.md
```

Training budgets are set in **skrl trainer timesteps** via the
`max_timesteps` config key (`++agent.trainer.timesteps` under the hood).
Do **not** use train.py's `--max_iterations` — it silently multiplies by
`rollouts` (24–32×) before becoming trainer timesteps.

### Recording training videos

Training videos are **on** in the benchmark configs: one ~12 s clip every
10 % of the cell's training budget (`video_interval: null` → auto
`max_timesteps // 10`, i.e. every 50 k trainer timesteps for Franka, 30 k
for UR3e → ~11 clips per run covering 0 %…100 % of training progress).

```bash
# Videos save to:
#   results/runs/<task>/<kernel>/seed_<N>/<skrl_timestamp_dir>/videos/train/*.mp4
# Override per invocation (CLI beats YAML):
python -m velgate_bench.sweep --config configs/sweep_franka.yaml \
    --video_interval 25000 --video_length 500
```

Rendering adds ~10-20 % training overhead. A kit crash-dialog hang was once
observed during a video-recording run; the sweep's log watchdog now kills
such hangs within ~30 s and labels the cell `fail(kit_crash_dialog)`.
`configs/sweep_videos.yaml` (seed 99, videos only) remains available for
extra qualitative clips without retraining the matrix.

## The claim

> **H1**: Among the three reward kernels considered, the velocity-gated tanh
> achieves the best trade-off between final policy performance, goal-hold
> stability, and late-stage training stability — across multiple manipulation
> tasks and random seeds.

Sub-claims (each testable; see `docs/claim.md` for falsification criteria).
Performance comparisons use **kernel-independent eval metrics** — training
reward is in kernel-specific units (each condition changes the reward
function) and is reported descriptively only:

- **H1a** *(performance)*: `success_rate(vel_gated) ≥ success_rate(tanh) − ε`
  and hold distance not significantly worse.
- **H1b** *(no limit cycle)*: `hold_joint_vel_l2(vel_gated) <
  hold_joint_vel_l2(tanh)`, with effect size Cohen's *d* > 0.8. Meaningful
  only jointly with H1a (the gate directly optimizes this quantity).
- **H1c** *(no drift)*: `drift_ratio(vel_gated) < drift_ratio(gaussian)`,
  ideally `drift_ratio(vel_gated) < 0.05` (smoothed-curve estimator).
- **H1d** *(generalization)*: H1a-c hold for both Franka and UR3e platforms.
- **H2** *(vs additive baseline)*: multiplicative gating beats the
  equally-parameterized additive velocity penalty on the
  success/hold-velocity trade-off.

## Reproducibility

- Fixed seeds (0, 1, 2, 3, 4) per (task, kernel) combination.
- Same PPO hyperparameters across all conditions (only the reward kernel
  differs).
- All env configs and kernel implementations checked into git.
- TB logs + eval data archived in `results/` (excluded from git but
  reproducible from scripts).
- Hardware: tested on RTX 4070 Ti, 12 GB VRAM, num_envs=8192. Should
  reproduce on any GPU ≥ 8 GB VRAM with proportionally adjusted num_envs.
- Pinned software versions (Isaac Sim 5.1.0.0, Isaac Lab 0.54.3, skrl
  1.4.3, torch 2.7.0+cu128, Python 3.11): full table in
  `docs/methodology.md`.

## Citing

If you use this benchmark or its findings, please cite as:

```bibtex
@misc{melichar2026velgated,
  title  = {Velocity-Gated Reward Shaping for PPO Reach-and-Hold Manipulation},
  author = {Melichar, Ales},
  year   = {2026},
  url    = {https://github.com/<TBD>/velocity_gated_reward},
  note   = {Benchmark study with companion technical report}
}
```

(Citation block to be updated when published.)

## Related work

- **Isaac Lab manipulation tasks** ([Mittal et al. 2023]) — provides the
  baseline tanh kernel and the task suite this benchmark builds on.
- **Reward shaping in RL** ([Ng, Harada & Russell 1999]) — theoretical
  foundation. Note: potential-based shaping is policy-invariant by
  construction, so it *cannot* fix hold behavior — this work necessarily
  changes the optimal policy. This work is empirical reward engineering,
  not theory.
- **CAPS** ([Mysore et al. 2021, arXiv:2012.06644]) — temporal + spatial
  action-smoothness regularization at the *loss* level; the main
  loss-level alternative to reward-level velocity gating. Orthogonal and
  composable; not benchmarked here.
- **PPO** ([Schulman et al. 2017]) — the on-policy algorithm whose failure
  modes this benchmark characterizes.

## Open questions for follow-up work

1. Does the same kernel shape help SAC / TD3 / TQC? (Off-policy methods may
  have different failure modes at the goal.)
2. Can the velocity threshold `v_thresh` be made state-dependent (e.g. via
  a small neural net) for better task-agnosticism?
3. Generalization to bimanual / non-prehensile tasks?
4. Sim-to-real transfer: does smoother sim behavior translate to smoother
  real-robot behavior?
