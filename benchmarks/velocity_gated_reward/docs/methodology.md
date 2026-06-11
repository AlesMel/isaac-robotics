# Experimental Methodology

## Tasks

| Name | Robot | DoF | Gripper | Source |
|---|---|---|---|---|
| `franka_lift` | Franka Panda | 7 | Parallel jaws (~80 mm stroke) | Upstream `isaaclab_tasks.manager_based.manipulation.lift.config.franka.FrankaCubeLiftEnvCfg` |
| `ur3e_lift` | UR3e | 6 | Robotiq Hand-E (~40 mm stroke) | This project's `isaac_robots.tasks.manager_based.lift` |

Each task is wrapped by `velgate_bench/envs/*.py` to swap **only** the
`object_goal_tracking_fine_grained` reward term while leaving all other env
parameters, action spaces, observation spaces, and PPO hyperparameters
identical to the baseline.

## Conditions (independent variable)

Five reward kernels, all replacing the same reward term
(see `velgate_bench/kernels.py` for exact code):

1. **`tanh`** — `R(d) = 1 - tanh(d / σ)` (upstream baseline; nonzero
   gradient at d=0 → hypothesized limit cycle).
2. **`gaussian`** — `R(d) = exp(-(d / σ)²)` (zero gradient at d=0 →
   hypothesized late-stage drift).
3. **`velocity_gated_tanh`** — `R(d, q̇) = (1 - tanh(d / σ)) · G(d, q̇)`
   (proposed), where `G` multiplies the reward by
   `clip(1 - ‖q̇‖_arm / v_thresh, 0, 1)` inside the goal neighborhood.
4. **`tanh_additive_velpen`** — `R(d, q̇) = (1 - tanh(d / σ)) -
   scale · 1[d < r] · clip(‖q̇‖_arm / v_thresh, 0, 1)` (competing
   baseline). The *additive* counterpart of the multiplicative gate with
   identical parameters; answers "would a plain distance-gated velocity
   penalty do just as well?". `penalty_scale = 1.0` matches the penalty
   range to the tracking term's [0, 1] so neither formulation is
   magnitude-handicapped (a reasoned default, not tuned).
5. **`velocity_gated_tanh_smooth`** — same as (3) but the hard indicator
   `1[d < r]` is replaced by `sigmoid((r − d)/τ)`, `τ = 0.02 m`
   (ablation). Removes the reward discontinuity at the neighborhood
   boundary, which in principle admits a boundary-orbiting local optimum.
   Franka-only (compute budget).

All kernels use σ = 0.05, weight = 5.0 (matching upstream lift's
`fine_grained` term); velocity-aware kernels share `v_thresh = 0.5 rad/s`,
`r = 0.10 m`. The coarse tracking term (σ = 0.3, weight = 16) is left on
vanilla tanh in all conditions, since it shapes approach behavior and is
not the limit-cycle source.

### Condition matrix

| Task | Kernels | Seeds | Cells |
|---|---|---|---|
| `franka_lift` | all 5 | 0–4 | 25 |
| `ur3e_lift` | 1–4 (no smooth ablation) | 0–4 | 20 |
| **Total** | | | **45** |

## Seeds

5 seeds per (task, kernel) combination: 0, 1, 2, 3, 4 (fixed and
reproducible). Eval rollouts reuse the training seed.

## Training budget

Budgets are expressed in **skrl trainer timesteps** (vector-env steps) and
set by the sweep via the hydra override `++agent.trainer.timesteps`
(NOT via train.py's `--max_iterations`, which silently multiplies by
`rollouts` before writing `trainer.timesteps` — a 24–32× inflation).

| Task | Trainer timesteps | Rationale |
|---|---|---|
| `franka_lift` | 500 000 | Upstream YAML default (36 000) is far short of convergence at 4096 envs. Pilot seed-0 curve peaks at ~240–360 k; 500 k covers the peak plus margin to expose late-training drift (which `drift_ratio` measures). |
| `ur3e_lift` | 300 000 | The tuned `ur3e_hande` skrl YAML default. |

## PPO hyperparameters

Sourced from each task's existing `agents/skrl_ppo_cfg.yaml` and
**identical across all kernel conditions within a task**:

| Param | Franka lift | UR3e lift |
|---|---|---|
| Network | [256, 128, 64] elu | [256, 128, 64] elu |
| `min_log_std` / `max_log_std` | -20 / 2 (upstream; effectively no σ floor) | -2.0 / 0.0 (σ floor 0.135) |
| `entropy_loss_scale` | 0.001 | 0.005 |
| `learning_rate` | 1e-4 | 3e-4 |
| `learning_rate_scheduler` | KLAdaptive (kl 0.01) | KLAdaptive (kl 0.008, min_lr 1e-5) |
| `rollouts` / `epochs` / `minibatches` | 24 / 8 / 4 | 32 / 8 / 4 |
| `random_timesteps` | 0 | 5 000 (pure-exploration warmup; applies equally to all kernels) |
| trainer `timesteps` | 500 000 (sweep override) | 300 000 |
| `num_envs` | 4 096 (default) | 8 192 (default) |
| `episode_length_s` | 5.0 | 8.0 |

Per-task hyperparameters differ between Franka and UR3e, but **within a
task all kernels see identical hyperparameters** — this isolates the kernel
as the independent variable. Cross-task comparisons conflate task
difficulty with hyperparameter choices and are not made.

**Exploration-noise control.** A competing explanation for the limit cycle
is forced exploration noise: UR3e's `min_log_std = -2` keeps action σ ≥
0.135 throughout training, so the learned mean policy must be
noise-robust. Franka's upstream bounds (-20/2) impose effectively no σ
floor. Since eval uses `mean_actions` (no policy noise), the Franka results
double as a control: if the tanh kernel still shows degraded hold metrics
on Franka, forced exploration noise alone cannot explain the limit cycle.

## Metrics

### From TB scalars

skrl's TB write interval is set to 500 trainer timesteps
(`++agent.agent.experiment.write_interval=500` → 1000 points per Franka
run, 600 per UR3e run; the "auto" default of timesteps/100 was too sparse).

1. **`peak_reward`** — max of the reward curve after smoothing with a
   centered rolling mean over ~5 % of points. (Max of the *raw* curve is
   biased upward by noise, inflating drift even for flat curves.)
2. **`final_reward`** — mean of smoothed values within the last 10 % of
   training **timesteps** (not points).
3. **`drift_ratio`** — `(peak_reward − final_reward) / peak_reward`.
   Reported as NaN for curves with < 20 points.

Training reward is **kernel-specific** (each kernel changes the reward
function), so `peak_reward` is never compared *across* kernels; it is
reported descriptively, and `drift_ratio` (a within-kernel ratio) remains
comparable.

### From deterministic eval rollouts (post-training, 100 episodes / seed)

Eval rollouts use `mean_actions` (no policy noise) on `best_agent.pt`.
These metrics are kernel-independent and form the **primary** comparison
axis:

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

- **Aggregate**: mean ± std across the 5 seeds, plus **IQM
  (interquartile mean) with 95 % percentile-bootstrap CI** (B = 10 000,
  seeded RNG) for the headline metrics (`success_rate`,
  `hold_joint_vel_l2`, `drift_ratio`) — robust to outlier seeds at small n
  (Agarwal et al., NeurIPS 2021).
- **Compare**: Welch's t-test (two-sided) per (comparison, task).
- **Multiple comparisons**: all primary pairwise p-values form one family
  corrected with **Holm–Bonferroni**; verdicts use adjusted p < 0.05.
- **Effect size**: Cohen's d, interpreted per Cohen 1988 thresholds
  (0.2 / 0.5 / 0.8).

The test family (see `COMPARISONS` in `velgate_bench/analyze.py`):

| Test | Metric | Candidate | Baseline | Direction |
|---|---|---|---|---|
| H1a performance | `success_rate`, `hold_obj_goal_dist_mean`, `time_at_goal_s` | velgated | tanh (+ gaussian for success) | not worse |
| H1b hold stability | `hold_joint_vel_l2` | velgated | tanh | lower, d > 0.8 |
| H1c drift | `drift_ratio` | velgated | gaussian | lower, velgated ≤ 5 % |
| H2 vs additive | `success_rate`, `hold_joint_vel_l2` | velgated | tanh_additive_velpen | better trade-off |
| Gate ablation | `success_rate`, `hold_joint_vel_l2` | smooth gate | hard gate | no regression |

Note on H1b framing: the velocity gate *directly* optimizes hold joint
velocity, so a reduction is expected nearly by construction. The
substantive claim is the **trade-off**: lower hold velocity *without*
losing success rate or hold precision (H1a + H1b jointly), and better than
the equally-parameterized additive penalty (H2).

## Related approaches (not implemented here)

- **CAPS** (Mysore et al. 2021, arXiv:2012.06644) regularizes the *policy
  loss* for temporal/spatial action smoothness — a loss-level alternative
  to reward-level gating. Orthogonal and composable; out of scope.
- **Potential-based reward shaping** (Ng et al. 1999) is policy-invariant
  by construction and therefore *cannot* change the optimal policy's hold
  behavior — it is not applicable as a fix for the limit cycle.
- The env's existing global `joint_vel` / `action_rate` penalties are
  present (identically) in all conditions; the additive baseline tests the
  *distance-gated* strengthening of that idea.

## Computational budget

| Sweep | # cells | Per-cell wall clock | Total (single RTX 4070 Ti) |
|---|---|---|---|
| `sweep_smoke.yaml` (gate) | 5 | ~10–15 min | ~1.25 h |
| `sweep_franka.yaml` | 25 | ~3 h | ~3 days |
| `sweep_ur3e.yaml` | 20 | ~3–6 h | ~3–5 days |
| **Total benchmark** | **45 (+5 smoke)** | | **~7–9 days** |

Eval rollouts add ~2–10 minutes per cell (100 episodes × 64 envs,
deterministic). Training-video recording is enabled (one ~12 s clip every
10 % of the budget, ~11 clips/run) and adds ~10–20 % to per-cell wall
clock; the estimates above include this overhead margin. Each training
cell has an 8 h wall-clock ceiling; cells killed by the ceiling or by the
kit-crash-dialog watchdog are labeled in `results/manifest.json`
(`fail(timeout)` / `fail(kit_crash_dialog)`).

## Software / hardware versions (pinned)

| Component | Version |
|---|---|
| Isaac Sim | 5.1.0.0 |
| isaaclab / isaaclab_tasks / isaaclab_rl | 0.54.3 / 0.11.14 / 0.5.0 |
| skrl | 1.4.3 (editable install) |
| torch | 2.7.0+cu128 |
| numpy / scipy | 1.26.0 / 1.15.3 |
| Python | 3.11 |
| GPU | NVIDIA RTX 4070 Ti (12 GB), single GPU |
| OS | Ubuntu (Linux 6.8) |

## Reproducibility checklist

- [x] Fixed seeds (0..4); eval seed == train seed
- [x] All hyperparameters declared in checked-in YAML / Python config
- [x] Single command runs the entire sweep (`scripts/run_sweep.sh`)
- [x] Single command regenerates all plots/tables (`scripts/make_figures.sh`)
- [x] Hardware + software versions pinned (table above)
- [x] Resumable sweep (skip already-trained cells)
- [x] Per-cell metadata (`cell_info.json`, `KERNEL.txt`, TB HParams)
