# Experimental Methodology

## Tasks

| Name | Robot | DoF | Gripper | Source |
|---|---|---|---|---|
| `franka_lift` | Franka Panda | 7 | Parallel jaws (~80 mm stroke) | Upstream `isaaclab_tasks.manager_based.manipulation.lift.config.franka.FrankaCubeLiftEnvCfg` |
| `ur3e_lift` | UR3e | 6 | Robotiq Hand-E (~40 mm stroke) | This project's `isaac_robots.tasks.manager_based.lift` |
| *(stretch)* `franka_reach` | Franka Panda | 7 | n/a (EE-to-pose) | Upstream `isaaclab_tasks.manager_based.manipulation.reach` |

Each task is wrapped by `velgate_bench/envs/*.py` to swap **only** the
`object_goal_tracking_fine_grained` reward term while leaving all other env
parameters, action spaces, observation spaces, and PPO hyperparameters
identical to the baseline.

## Conditions (independent variable)

Three reward kernels, all replacing the same reward term:

1. **`tanh`** — `R(d) = 1 - tanh(d / σ)` (upstream baseline)
2. **`gaussian`** — `R(d) = exp(-(d / σ)²)`
3. **`velocity_gated_tanh`** — `R(d, q̇) = (1 - tanh(d / σ)) · G(d, q̇)`,
   where `G` is the velocity gate (see `velgate_bench/kernels.py`).

All kernels use σ = 0.05, weight = 5.0 (matching upstream lift's
`fine_grained` term). The coarse tracking term (σ = 0.3, weight = 16) is
left on vanilla tanh in all conditions, since it shapes approach behavior
and is not the limit-cycle source.

## Seeds

5 seeds per (task, kernel) combination → 30 training runs for the 2-task
default, 45 for the 3-task stretch.

Seeds: 0, 1, 2, 3, 4 (fixed and reproducible).

## PPO hyperparameters

Identical across all conditions. Sourced from each task's existing
`agents/skrl_ppo_cfg.yaml`:

| Param | Franka lift | UR3e lift |
|---|---|---|
| Network | [256, 128, 64] elu | [256, 128, 64] elu |
| `min_log_std` / `max_log_std` | upstream defaults | -2.0 / 0.0 |
| `entropy_loss_scale` | 0.001 | 0.005 |
| `learning_rate` | 3e-4 | 3e-4 |
| `learning_rate_scheduler` | KLAdaptive | KLAdaptive (min_lr 1e-5) |
| `rollouts` / `epochs` / `minibatches` | 24 / 8 / 4 | 32 / 8 / 4 |
| `entropy_loss_scale` | 0.001 | 0.005 |
| `timesteps` | 36 000 | 300 000 |
| `num_envs` | 4 096 (default) | 8 192 (default) |
| `episode_length_s` | 5.0 | 8.0 |

Note: per-task hyperparameters differ between Franka and UR3e, but **within
a task all three kernels see identical hyperparameters**. This isolates the
kernel as the independent variable.

## Metrics

### From TB scalars (no extra eval needed)

1. **`peak_reward`** — `max(Episode_Reward/total)` across training.
2. **`final_reward`** — mean of last 10% of `Episode_Reward/total` values.
3. **`drift_ratio`** — `(peak_reward − final_reward) / peak_reward`.

### From deterministic eval rollouts (post-training, 100 episodes / seed)

Eval rollouts use `mean_actions` (no policy noise) on `best_agent.pt`.

4. **`hold_joint_vel_l2`** — RMS magnitude of *arm* joint velocities during
   the last 2 seconds of each episode (the "hold window"). Filtered to arm
   joints only via the regex pattern in `velgate_bench.sweep.TASKS`.
5. **`hold_ee_z_std`** — std of EE z-coordinate during the hold window.
   Captures vertical oscillation amplitude directly.
6. **`hold_obj_goal_dist_mean`** / `_std` — cube-to-goal distance during
   hold window. Captures both bias and precision.
7. **`success_rate`** — fraction of episodes ending with cube within
   3 cm of goal.
8. **`time_at_goal_s`** — cumulative seconds cube was within 5 cm of goal.

## Statistical analysis

For each (task, metric) pair:

- **Aggregate**: mean ± std across the 5 seeds.
- **Compare**: Welch's t-test (two-sided, α = 0.05) between
  `velocity_gated_tanh` and each baseline (`tanh`, `gaussian`).
- **Effect size**: Cohen's d, interpreted as small/medium/large per
  Cohen 1988 thresholds (0.2 / 0.5 / 0.8).

The three primary tests:

| Test | Metric | Baseline | Direction | Threshold for "win" |
|---|---|---|---|---|
| **Peak reward** | `peak_reward` | `tanh` | Higher | mean diff ≥ -5 pts (i.e., not significantly worse) |
| **Hold stability** | `hold_joint_vel_l2` | `tanh` | Lower | p < 0.05 AND Cohen's d > 0.8 |
| **No drift** | `drift_ratio` | `gaussian` | Lower | velgated < 5% AND p < 0.05 |

## Computational budget

| Sweep | # runs | Wall-clock (single RTX 4070 Ti) |
|---|---|---|
| Default (2 tasks × 3 kernels × 5 seeds) | 30 | ~10 hours |
| Stretch (3 tasks × 3 kernels × 5 seeds) | 45 | ~15 hours |
| Hyperparam ablation on velocity-gated (v_thresh × neighborhood × 3 seeds) | 27 | ~9 hours |
| **Total publication-grade** | ~72 | **~24 hours of GPU time** |

Eval rollouts add ~2 minutes per run (100 episodes × 64 envs deterministic).

## Reproducibility checklist

- [x] Fixed seeds (0..4)
- [x] All hyperparameters declared in checked-in YAML / Python config
- [x] Single command runs the entire sweep (`scripts/run_sweep.sh`)
- [x] Single command regenerates all plots/tables (`scripts/make_figures.sh`)
- [x] Hardware + software versions documented in top-level README
- [x] Resumable sweep (skip already-trained cells)
- [ ] Pinned dependency versions in `pyproject.toml` (TBD — depends on
      whether benchmark ships standalone or as part of `isaac_robots`)
