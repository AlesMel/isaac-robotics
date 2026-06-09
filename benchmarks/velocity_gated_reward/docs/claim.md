# Claim & Hypotheses

## Primary claim

**H1**: For PPO continuous-control "reach-and-hold" manipulation tasks, a
**velocity-gated tanh** reward kernel for fine-grained goal tracking
achieves a strictly better trade-off between final policy performance,
goal-hold stability, and late-stage training stability than either the
vanilla **tanh** kernel (the upstream Isaac Lab default) or a **Gaussian**
kernel — across multiple robot platforms.

## Sub-claims (each testable, each with a falsification criterion)

### H1a — Performance is not sacrificed

`peak_reward(velocity_gated_tanh) ≥ peak_reward(tanh) - ε` for ε = 5 reward
points (or 3% of mean, whichever is larger), on every task in the benchmark.

**Falsified if**: velocity-gated tanh's peak reward is significantly lower
than the tanh baseline by > 5 points / 3% (Welch's t-test, α=0.05).

### H1b — Limit cycle is eliminated

`hold_joint_vel_l2(velocity_gated_tanh) < hold_joint_vel_l2(tanh)`, with
Cohen's d > 0.8 (large effect size).

**Falsified if**: velocity-gated tanh's hold joint velocity is statistically
indistinguishable from tanh, or larger.

### H1c — No late-stage drift

`drift_ratio(velocity_gated_tanh) ≤ 5%` on every task and seed, AND
`drift_ratio(velocity_gated_tanh) < drift_ratio(gaussian)`.

Where `drift_ratio = (peak_reward − final_reward) / peak_reward` and
`final_reward` is the mean reward over the last 10% of training timesteps.

**Falsified if**: velocity-gated tanh's mean drift ratio exceeds 5%, OR
exceeds the Gaussian baseline.

### H1d — Generalization across platforms

Sub-claims H1a–c hold for both Franka Panda (7-DoF, parallel jaws) and UR3e
+ Robotiq Hand-E (6-DoF, parallel jaws). Ideally also for at least one
additional task (e.g. Franka reach) in the publication-grade extension.

**Falsified if**: any of H1a–c fails on either platform.

## What is *not* claimed

To avoid overclaiming:

1. **We do not claim** the velocity-gated tanh is the unique optimum. Other
   kernels (smoothed step, learned reward, hybrid) may match or exceed it.
2. **We do not claim** the result transfers to off-policy algorithms (SAC,
   TD3, TQC) without further verification. The benchmark is PPO-specific.
3. **We do not claim** the result transfers to non-PD-controlled robots, to
   bimanual / non-prehensile tasks, or to image-based observations without
   further verification.
4. **We do not claim** the kernel needs no hyperparameter tuning per task.
   The defaults (`velocity_thresh=0.5 rad/s`, `neighborhood=0.10 m`) work
   for the benchmark tasks but are not derived theoretically.

## Significance threshold

For all statistical tests:
- α = 0.05 (two-sided Welch's t-test)
- n ≥ 5 seeds per (task, kernel) condition
- Cohen's d reported alongside p-values; effect sizes interpreted as
  small (0.2), medium (0.5), large (0.8) per Cohen 1988.
