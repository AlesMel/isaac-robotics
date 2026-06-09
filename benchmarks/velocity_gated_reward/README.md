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
│   ├── kernels.py            # the three reward functions (parameterized)
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

# 1. Run the full sweep (3 kernels × 5 seeds × N tasks)
#    Default: Franka lift + UR3e Hand-E lift, ~10 GPU-hours on RTX 4070 Ti
bash scripts/run_sweep.sh

# 2. Run deterministic eval rollouts on all trained checkpoints
bash scripts/run_eval.sh

# 3. Generate publication-ready plots and tables
python -m velgate_bench.analyze --results-dir results/

# 4. Read the results
open docs/results.md
```

### Recording training videos

Training videos are off by default for the full sweep (disk + 10-20% overhead).
Two ways to enable:

```bash
# (a) Per-invocation CLI flag (overrides YAML):
python -m velgate_bench.sweep --config configs/sweep_default.yaml \
    --record_video_train --video_interval 5000 --video_length 500

# (b) Edit configs/sweep_default.yaml: set record_video_train: true

# Videos save to:
#   results/runs/<task>/<kernel>/seed_<N>/<skrl_timestamp_dir>/videos/train/*.mp4
# One clip per `video_interval` env steps, each `video_length` frames long.
```

The smoke-test config (`configs/sweep_quick.yaml`) **enables videos by default**
so you can visually confirm policies are doing something sensible before
committing to the full sweep.

## The claim

> **H1**: Among the three reward kernels considered, the velocity-gated tanh
> achieves the best trade-off between final policy performance, goal-hold
> stability, and late-stage training stability — across multiple manipulation
> tasks and random seeds.

Sub-claims (each testable):

- **H1a** *(performance)*: `peak_reward(vel_gated) ≥ peak_reward(tanh)` and
  `peak_reward(vel_gated) > peak_reward(gaussian)`.
- **H1b** *(no limit cycle)*: `hold_joint_vel_l2(vel_gated) <
  hold_joint_vel_l2(tanh)`, ideally with effect size Cohen's *d* > 0.8.
- **H1c** *(no drift)*: `drift_ratio(vel_gated) < drift_ratio(gaussian)`,
  ideally `drift_ratio(vel_gated) < 0.05` (i.e. < 5% reward loss after peak).
- **H1d** *(generalization)*: H1a-c hold for both Franka and UR3e platforms.

## Reproducibility

- Fixed seeds (0, 1, 2, 3, 4) per (task, kernel) combination.
- Same PPO hyperparameters across all conditions (only the reward kernel
  differs).
- All env configs and kernel implementations checked into git.
- TB logs + eval data archived in `results/` (excluded from git but
  reproducible from scripts).
- Hardware: tested on RTX 4070 Ti, 12 GB VRAM, num_envs=8192. Should
  reproduce on any GPU ≥ 8 GB VRAM with proportionally adjusted num_envs.

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
  foundation; this work is empirical reward engineering, not theory.
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
