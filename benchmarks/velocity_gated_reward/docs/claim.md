# Claim & Hypotheses

## Primary claim

**H1**: For PPO continuous-control "reach-and-hold" manipulation tasks, a
**velocity-gated tanh** reward kernel for fine-grained goal tracking
achieves a strictly better trade-off between task performance, goal-hold
stability, and late-stage training stability than either the vanilla
**tanh** kernel (the upstream Isaac Lab default) or a **Gaussian**
kernel — across multiple robot platforms.

All performance comparisons use **kernel-independent eval-rollout metrics**
(success rate, hold distance, time at goal, hold joint velocity), because
training reward is in kernel-specific units: each condition *changes* the
reward function, so training-reward magnitudes are not comparable across
kernels and are reported descriptively only.

## Sub-claims (each testable, each with a falsification criterion)

### H1a — Performance is not sacrificed

`success_rate(velocity_gated_tanh) ≥ success_rate(tanh) − ε` for ε = 0.05,
AND `hold_obj_goal_dist_mean(velocity_gated_tanh)` is not significantly
worse than tanh's, on every task in the benchmark.

**Falsified if**: velocity-gated tanh's success rate or hold distance is
significantly worse than the tanh baseline (Welch's t-test, Holm-adjusted
p < 0.05) by more than ε.

### H1b — Limit cycle is eliminated

`hold_joint_vel_l2(velocity_gated_tanh) < hold_joint_vel_l2(tanh)`, with
Cohen's d > 0.8 (large effect size).

Note: the gate directly optimizes this quantity, so the reduction alone is
weak evidence; H1b is only meaningful **jointly with H1a** (stability gained
without performance lost).

**Falsified if**: velocity-gated tanh's hold joint velocity is statistically
indistinguishable from tanh, or larger.

### H1c — No late-stage drift

`drift_ratio(velocity_gated_tanh) ≤ 5%` on every task and seed, AND
`drift_ratio(velocity_gated_tanh) < drift_ratio(gaussian)`.

Where `drift_ratio = (peak − final) / peak` computed on the smoothed
training-reward curve; `final` is the mean over the last 10% of training
**timesteps** (see methodology.md for the exact estimator). Drift is a
within-kernel ratio, so it remains comparable across kernels.

**Falsified if**: velocity-gated tanh's mean drift ratio exceeds 5%, OR
exceeds the Gaussian baseline.

### H1d — Generalization across platforms

Sub-claims H1a–c hold for both Franka Panda (7-DoF, parallel jaws) and UR3e
+ Robotiq Hand-E (6-DoF, parallel jaws).

**Falsified if**: any of H1a–c fails on either platform.

### H2 — Multiplicative gating beats the additive penalty (secondary)

Against the equally-parameterized additive baseline
(`tanh_additive_velpen`, same `v_thresh`, `neighborhood`, and a
penalty range matched to the tracking term), the multiplicative gate
achieves at least as high a success rate at equal-or-lower hold joint
velocity.

**Falsified if**: the additive baseline matches or beats velocity-gated
tanh on both success rate and hold joint velocity. (This would mean the
contribution reduces to a known technique — distance-gated velocity
penalties — and the paper's framing must change.)

### Gate-boundary ablation (supporting, Franka only)

`velocity_gated_tanh_smooth` (sigmoid-blended gate, τ = 0.02 m) performs
within noise of the hard-gate variant. This defends against the objection
that the hard gate's reward discontinuity at the neighborhood boundary
drives the results or admits a boundary-orbiting optimum.

## What is *not* claimed

To avoid overclaiming:

1. **We do not claim** the velocity-gated tanh is the unique optimum. Other
   kernels (smoothed step, learned reward, hybrid) may match or exceed it.
   Loss-level smoothness regularization (CAPS, Mysore et al. 2021) is an
   orthogonal, composable alternative that we do not benchmark.
2. **We do not claim** the result transfers to off-policy algorithms (SAC,
   TD3, TQC) without further verification. The benchmark is PPO-specific.
   We partially control for exploration-noise effects via the Franka task,
   whose PPO config has effectively no policy-σ floor (`min_log_std=-20`)
   in contrast to UR3e's forced σ ≥ 0.135 (see methodology.md).
3. **We do not claim** the result transfers to non-PD-controlled robots, to
   bimanual / non-prehensile tasks, or to image-based observations without
   further verification.
4. **We do not claim** the kernel needs no hyperparameter tuning per task.
   The defaults (`velocity_thresh=0.5 rad/s`, `neighborhood=0.10 m`) work
   for the benchmark tasks but are not derived theoretically.

## Significance threshold

For all statistical tests:
- Two-sided Welch's t-test per (comparison, task); the full set of primary
  comparisons forms one family corrected with **Holm–Bonferroni**;
  verdicts require adjusted p < 0.05.
- n = 5 seeds per (task, kernel) condition.
- Cohen's d reported alongside p-values; effect sizes interpreted as
  small (0.2), medium (0.5), large (0.8) per Cohen 1988.
- IQM with 95% bootstrap CI (B = 10 000, seeded) reported for headline
  metrics alongside mean ± std.
