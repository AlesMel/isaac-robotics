# UR3e + Robotiq Hand-E — Manager-Based Lift Cube

> Isaac Lab manager-based RL environment for training a Universal Robots UR3e
> arm equipped with a Robotiq Hand-E parallel-jaw gripper to grasp and lift a
> cube to a randomized target pose. PPO via skrl.

This document is both a technical reference for the environment and a candid
write-up of the design decisions, debugging journey, and current open issues.

---

## 1. Project Overview

### 1.1 What it does

The environment instantiates `N` parallel scenes (typically 4 096 or 8 192),
each containing:

| Asset | Source | Configured at |
|---|---|---|
| **UR3e arm + Robotiq Hand-E** (one articulation) | Local USD at [`source/isaac_robots/data/ur3e/ur3e_robotiq_hande.usd`](../../../../data/ur3e/ur3e_robotiq_hande.usd) wrapping the upstream Nucleus Hand-E reference | `UR3E_ROBOTIQ_HANDE_CFG` in [`tasks/direct/_shared/assets.py`](../../direct/_shared/assets.py) |
| **DexCube** (40 mm cube) | Isaac Nucleus instanceable asset (scale 0.5) | Inline in env config |
| **Seattle Lab Table** | Isaac Nucleus | Inherited from upstream `ObjectTableSceneCfg` |

The agent observes joint positions/velocities + object position + goal command,
and outputs 6 arm-joint relative-position commands + 1 binary gripper command.
Reward shaping is the upstream Isaac Lab lift recipe (reaching → lifting →
goal tracking) with a custom Gaussian-kernel fine-grained tracking term.

### 1.2 Why manager-based (rather than direct)

The project already shipped a direct-style version of this task at
[`tasks/direct/isaac_ur3e_lift_cube/`](../../direct/isaac_ur3e_lift_cube/).
The manager-based version was built to unlock:

- Modular reward/observation/event composition (managers can be swapped per
  variant — joint-pos vs. IK, demos vs. RL, etc.)
- The Isaac Lab `CommandManager` for randomized goal sampling
- Compatibility with upstream Isaac Lab manipulation tooling (teleop, robomimic
  demos, etc.)
- IK-relative control variant for human demonstration capture

### 1.3 Gym IDs

| ID | Variant | Control mode | num_envs default |
|---|---|---|---|
| `Isaac-Robots-Lift-Cube-UR3e-HandE-v0` | Training | Relative joint position | 4096 |
| `Isaac-Robots-Lift-Cube-UR3e-HandE-Play-v0` | Playback / interactive | Relative joint position | 50 |
| `Isaac-Robots-Lift-Cube-UR3e-HandE-IK-Rel-v0` | Teleop / demo capture | Differential IK (relative pose) | 4096 |

Registered in [`config/ur3e_hande/__init__.py`](config/ur3e_hande/__init__.py).

---

## 2. Architecture

### 2.1 Directory layout

```
source/isaac_robots/isaac_robots/tasks/manager_based/lift/
├── __init__.py                       # package marker
├── lift_env_cfg.py                   # re-exports upstream LiftEnvCfg & sub-configs
├── mdp/
│   ├── __init__.py                   # wildcard re-export of upstream mdp + local overrides
│   └── rewards.py                    # custom: object_goal_distance_gaussian
└── config/
    ├── __init__.py
    └── ur3e_hande/
        ├── __init__.py               # gym.register calls (3 variants)
        ├── joint_pos_env_cfg.py      # UR3eHandECubeLiftEnvCfg + _PLAY (primary)
        ├── ik_rel_env_cfg.py         # IK-relative variant for teleop
        └── agents/
            ├── __init__.py
            └── skrl_ppo_cfg.yaml     # PPO hyperparameters
```

### 2.2 Design pattern: re-export upstream, override locally

The strategy mirrors the existing `tasks/manager_based/stack/` package:

1. **`lift/lift_env_cfg.py`** does `from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import (...)` — no local copy of upstream code.
2. **`lift/mdp/__init__.py`** does `from isaaclab_tasks.manager_based.manipulation.lift.mdp import *` (which itself wildcards both `isaaclab.envs.mdp` and the lift-specific extras like `object_goal_distance`), then locally adds `object_goal_distance_gaussian`.
3. **`config/ur3e_hande/joint_pos_env_cfg.py`** subclasses upstream `LiftEnvCfg` and overrides robot wiring + reward kernel + sim params in `__post_init__`.

Benefit: upstream Isaac Lab bug-fixes flow automatically. Only the robot- and
gripper-specific bits are local code.

---

## 3. Key Technical Decisions

### 3.1 Action representation — *Relative* joint position, not absolute

Upstream Franka uses `JointPositionActionCfg(use_default_offset=True)`, which
maps the policy's continuous action to a **target joint position**:

```
target = default_joint_pos + scale * action
```

This couples per-step velocity to workspace coverage. With a small `scale`,
the policy must output **large action magnitudes** (e.g. mean ≈ 5) just to
reach the cube. Then the transition from "reaching down" (action ≈ 5) to
"lifting up" (action ≈ 0) is a 5-unit drop in one timestep — the PD controller
snaps the joint targets so fast that the cube is **torn out of the gripper**.

We switched to `RelativeJointPositionActionCfg`:

```
target = current_joint_pos + scale * action
```

Now `scale` is a true per-step delta cap. With `scale=0.05 rad` and 50 Hz
control, **max joint speed ≈ 143°/s** regardless of action magnitude. Workspace
coverage is unlimited (policy accumulates deltas over time). No snap on pose
transitions.

Episode length was bumped from upstream's 5.0 s to **8.0 s** to give the
slower arm enough time for reach → grasp → lift → hold.

### 3.2 Cube sizing — must match Hand-E stroke

DexCube native size is ~80 mm. Upstream Franka uses `scale=0.8` (64 mm cube),
appropriate for Franka's ~80 mm gripper stroke. The Hand-E has a much shorter
stroke (the local USD limits sliders to `[-0.02, 0.0]` = **40 mm total**), so
the 64 mm cube **physically cannot fit between the Hand-E pads**.

What we observed in playback: the policy reached reward 178 with `lifting_object`
near maximum, but visual playback showed the cube **clipping into the side of
the gripper body**. The policy had learned a *lateral cheat* — pinning the cube
against the convex-hull collision shapes of the defeature parts (screws,
washers) on the Hand-E base, which acted as a "shelf" that friction lifted.

Fix: `scale=0.5` (~40 mm cube), `pos.z=0.036` (smaller cube sits lower on
table). After this change, the policy is forced to actually grasp between the
pads. See [`hande-cube-sizing` auto-memory](../../../../../../.claude/projects/-home-urkui-3-Documents-isaac-robotics/memory/hande-cube-sizing.md).

### 3.3 PPO hyperparameter strategy

Two failure modes shaped the final hyperparameters:

**Failure mode A: Entropy collapse**
- Symptom: policy std drops below ~0.05, exploration dies, value function
  collapses to a near-constant prediction, gradient signal vanishes,
  catastrophic reward drop with no recovery.
- Observed: run `2026-06-04_15-41-03` collapsed at step 128k.
- Fix: `min_log_std: -2.0` (hard floor at std ≈ 0.135) + `entropy_loss_scale:
  0.005`.

**Failure mode B: Entropy runaway**
- Symptom: policy std grows to extreme values (observed 1.62), action noise
  becomes huge, training rewards look high but policy is chaotic — collecting
  *entropy bonus* rather than learning a precise policy.
- Observed: run `2026-06-06_11-44-01` had std climb to 1.62 with reward 258
  (false high).
- Fix: `max_log_std: 0.0` (hard ceiling at std = 1.0).

The combination — **floor + ceiling + moderate entropy bonus** — gives a
self-stabilizing PPO that neither collapses nor runs away. Standard deviation
settled at exactly `exp(-2) = 0.135` for the bulk of training in the
subsequent run.

### 3.4 Curriculum — disabled

Upstream's `LiftEnvCfg` ramps `action_rate` and `joint_vel` penalty weights
from `-1e-4` to `-1e-1` (1000×) at env step 10 000. For UR3e + Hand-E this
caused two failures:

- At step 10k, the ramp fired before the grasp was learned → policy frozen
  near home pose because every motion was now heavily penalized.
- Delayed to step 100k with reduced ramp (100×) → still produced post-ramp
  shock collapse (lifting_object 13.9 → 1.7 within 10k steps).

We disabled the ramp entirely — `action_rate` and `joint_vel` weights stay at
`-1e-4` for the whole training run. The penalties are small enough not to
collapse the policy and large enough to keep the action_rate value visible in
TB (≈ -1e-4 per episode under normal conditions).

### 3.5 Reward kernel — Gaussian for fine-grained tracking

Upstream uses two tracking terms:

| Term | Kernel | std | weight |
|---|---|---|---|
| `object_goal_tracking` (coarse) | `1 - tanh(d/0.3)` | 0.3 | 16.0 |
| `object_goal_tracking_fine_grained` | `1 - tanh(d/0.05)` | 0.05 | 5.0 |

The **tanh kernel has nonzero gradient at d=0**: `∂/∂d (1 - tanh(d/std)) = -1/std`
at d=0. For `std=0.05`, that's a gradient of **-20**. Even when the cube is
exactly at the goal, the policy "feels" a corrective pull. Combined with the
cube weight perturbing the arm by ~1 mm/step, this produces a visible
**up-down limit cycle** in the playback at the goal pose.

We replaced the fine-grained kernel with a Gaussian:

```python
def object_goal_distance_gaussian(env, std, minimal_height, command_name, ...):
    # ... distance computation identical to upstream ...
    return lifted.float() * torch.exp(-(distance / std) ** 2)
```

The Gaussian has the **same maximum** (1.0 at d=0) but **zero gradient at d=0**
— smooth peak. The policy gets full reward when at goal and no penalty signal
for sub-cm drift → no correction → no limit cycle.

**Open issue**: see [§6.1](#61-gaussian-kernel-causes-late-stage-policy-drift).

### 3.6 GPU sim buffer sizing for large num_envs

At `num_envs ≥ 8192`, PhysX overflows two default buffers. The env config
bumps both:

```python
self.sim.physx.gpu_max_rigid_patch_count = 5 * 2**17          # 4× default (was 163_840)
self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024 # 4× default (was 16_384)
```

This headroom scales to ~16–32k envs comfortably.

---

## 4. Implementation Details

### 4.1 Robot articulation wiring

```python
self.scene.robot = UR3E_ROBOTIQ_HANDE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
self.scene.robot.spawn.activate_contact_sensors = False  # ~10-20% sim speedup
```

The shared `UR3E_ROBOTIQ_HANDE_CFG` comes from
[`tasks/direct/_shared/assets.py`](../../direct/_shared/assets.py) and provides:

- Implicit PD actuators for the 6 arm joints (shoulder/elbow/wrist groupings)
- Implicit PD actuator for the gripper sliders (effort_limit=20 N, stiffness=2000)
- USD path env-var override (`UR3E_ROBOTIQ_HANDE_USD_PATH`) for asset experimentation

### 4.2 Hand-E gripper specifics

The local USD has the slider limits re-defined as `[-0.02, 0.0]` (NOT the
upstream Nucleus `[0.0, 0.025]`). Convention:

- `0.0` (upper limit) = jaws **open** (spawn pose)
- `-0.02` (lower limit) = jaws **closed**

```python
self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
    asset_name="robot",
    joint_names=["Slider_.*"],
    open_command_expr={"Slider_.*": 0.0},
    close_command_expr={"Slider_.*": -0.02},
)
```

`BinaryJointPositionActionCfg` is a **switch**, not a magnitude-scaled action.
Policy output > 0 → close, ≤ 0 → open. Grip force is set by the actuator's
`effort_limit_sim=20.0 N` in the shared cfg, not by the action class.

### 4.3 End-effector frame

```python
self.scene.ee_frame = FrameTransformerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base_link",
    target_frames=[FrameTransformerCfg.FrameCfg(
        prim_path="{ENV_REGEX_NS}/Robot/tool0",
        name="end_effector",
        offset=OffsetCfg(pos=(0.0, 0.0, 0.119)),  # ~grasp point between closed pads
    )],
)
```

The EE frame is the FrameTransformer-tracked grasp point used by all
distance-based rewards. The `tool0` body is the UR3e wrist-3 flange; the
Hand-E grasp point sits along tool0's local +Z direction. (Note: under the
hood, the Hand-E base is attached to tool0 with a 120° rotation around the
(1,1,1) axis, so the slider axis actually maps to tool0's local +X. The
0.119 m offset along tool0's +Z is a workspace approximation that empirically
sites the marker close enough to the actual pad faces for the policy to align
the cube there correctly.)

### 4.4 Goal command sampling

```python
self.commands.object_pose.body_name = "tool0"
self.commands.object_pose.ranges.pos_x = (0.3, 0.45)
self.commands.object_pose.ranges.pos_y = (-0.2, 0.2)
self.commands.object_pose.ranges.pos_z = (0.15, 0.35)
```

Tightened from upstream Franka's `(0.4, 0.6) × (-0.25, 0.25) × (0.25, 0.5)`
because UR3e has ~0.5 m reach vs. Franka's ~0.8 m. Resampling every 5 s
(upstream default).

### 4.5 Network architecture

```yaml
models:
  separate: False              # shared backbone, two heads
  policy:                       # actor head
    layers: [256, 128, 64]      # pyramid (matches upstream Franka shape)
    activations: elu
  value:                        # critic head
    layers: [256, 128, 64]
    activations: elu
```

Shared backbone (fewer params, value benefits from policy features). Pyramid
shape mirrors upstream Franka lift's `rsl_rl_ppo_cfg.py`. ELU activations are
the standard for Isaac Lab manipulation; tanh was tested early but produced
marginally slower convergence with no qualitative difference.

---

## 5. Training Journey (Chronological)

This section is the candid debugging story. Each TB run name is preserved for
traceability.

### Run 1: Initial creation (peak 154, then collapse)

- First training with upstream-derived config — Franka-sized cube (0.8 scale),
  absolute `JointPositionActionCfg`, full upstream curriculum.
- TB run `2026-06-03_09-55`: clean ramp to reward 154 at step 50k, then
  collapse at step 210k as upstream curriculum fired.

### Run 2: Curriculum softening — collapsed again

- Delayed curriculum step from 10k → 100k, reduced ramp from 1000× to 100×.
- TB run `2026-06-03_15-41`: post-ramp shock collapsed lifting reward from 13.9
  → 1.7 in a single 10k-step window. KL-adaptive LR decayed to ~0 trying to
  recover. Never returned to peak.

### Run 3: Curriculum disabled — collapse-free but with action-coupling issue

- Curriculum entirely disabled (weights stay at -1e-4 throughout). Added
  `min_lr: 1e-5` floor on KL-adaptive scheduler.
- Training stable to reward 154, but playback revealed the **lateral cheat**:
  cube wasn't actually grasped between Hand-E pads.
- Diagnosis: cube too big (64 mm) for Hand-E stroke (40 mm).

### Run 4: Cube scaled 0.8 → 0.5 — grasp works but oscillation appears

- 40 mm cube fits between pads. Policy learns proper grasp.
- TB run `2026-06-06_22-24`: clean training to **reward 249 stable**. Best
  numerical result.
- Playback: hover oscillation at the goal pose. Functional but visually poor.

### Run 5: Widened fine_grained std (Fix 1) — std runaway disaster

- Hypothesis: sharp fine_grained reward (std=0.05) drives the limit cycle;
  widen to 0.08.
- TB run `2026-06-06_11-44`: reward climbed to **258 false high**, but policy
  std exploded from 0.18 → **1.62**. Action_rate penalty went from -0.0001 to
  -0.012 (100×). Visually catastrophic — chaotic noisy motion. Wider reward
  let entropy bonus drive a runaway.

### Run 6: Trio fix (max_log_std cap + entropy down + revert) — back to stable

- `max_log_std: 0.0` (std ceiling = 1.0), `entropy_loss_scale: 0.005`,
  reverted fine_grained std to 0.05.
- TB run `2026-06-06_22-24-24` (sister run): policy std stabilized at floor
  `0.135`, action_rate at clean -0.0001, reward climbed monotonically to **249**.
- Playback: same hover oscillation as Run 4. The trio fix prevented std
  runaway but didn't address the underlying limit-cycle cause.

### Run 7: Gaussian fine_grained kernel — fixes oscillation, introduces drift

- Replaced tanh fine_grained kernel with Gaussian `exp(-(d/std)²)`. Zero
  gradient at d=0 = no centering pull at goal.
- TB run `2026-06-07_23-03`: peak reward **226 at step 153k**, then degradation
  to **181 by step 288k** (-20%). fine_grained reward dropped from peak 1.76
  to 0.35 (-80%); reaching_object dropped 0.84 → 0.74.
- Diagnosis: Gaussian's zero gradient at d=0 also means no "stickiness" at
  optimum. Policy slowly drifts under noise without the gradient pull-back
  that tanh provides.

### Where we are now

Final state of the env is **Gaussian kernel active**, but with the known
trade-off above. See [§6.1](#61-gaussian-kernel-causes-late-stage-policy-drift).

---

## 6. Open Issues

### 6.1 Goal-hold oscillation — RESOLVED (2026-06-08)

**Resolution:** Velocity-gated tanh kernel + bumped action_rate/joint_vel
weights to -1e-3 (constant, no curriculum).

The full evolution:

| Run | Kernel | action_rate weight | Peak | End | Limit cycle? | Drift? |
|---|---|---|---|---|---|---|
| `22-24` | tanh (upstream) | -1e-4 | 249 | 249 stable | Yes (visible) | No |
| `23-03` | Gaussian | -1e-4 | 226 | 181 (-20%) | No | **Yes** |
| **Final** | **velocity-gated tanh** | **-1e-3** | TBD | **Stable** | **No** | **No** |

**Key insight:** Limit cycle and drift are mirror failure modes of "reward
gradient at d=0":
- **Nonzero gradient at d=0 (tanh)** → policy always tries to "improve" the
  position → noise amplification → cycle.
- **Zero gradient at d=0 (Gaussian)** → policy has no anchor at optimum →
  noise dispersion → drift.

The velocity-gated tanh changes **what** is rewarded inside the goal
neighborhood — from "be at goal" (sensitive to micro-perturbations from
cube weight) to "be at goal AND still" (the cube-weight perturbation can no
longer dent the reward because the policy isn't penalized for being at goal
with non-zero velocity; the gate only penalizes ITS OWN corrective
oscillation). Tanh's centering pull is preserved, so the policy stays at
optimum.

See `mdp/rewards.py:object_goal_distance_velocity_gated`. Default parameters:
`velocity_thresh=0.5 rad/s`, `neighborhood=0.10 m`, gate applied only to arm
joints (gripper sliders excluded via `SceneEntityCfg(joint_names=...)`).

### 6.2 Snappy reach when cube spawns close to base

When the cube spawns at the low-x end of its range (close to the robot base),
the policy executes a fast descent that risks brushing the table. The policy
is calibrated for "average reach distance"; close-cube cases are out-of-
distribution for the trained reach trajectory.

Candidate fixes:
- Tighten cube spawn range to exclude x < 0.35.
- Lower action scale further (0.05 → 0.04). Sacrifices some max joint speed.
- Add a curriculum that includes near-base spawn positions early.

---

## 7. Final Configuration Reference

### 7.1 Environment ([`joint_pos_env_cfg.py`](config/ur3e_hande/joint_pos_env_cfg.py))

| Parameter | Value | Notes |
|---|---|---|
| `episode_length_s` | 8.0 s | Longer than upstream 5.0 — relative action needs more steps |
| `decimation` | 2 | Upstream default → policy at 50 Hz |
| `sim.dt` | 0.01 s | 100 Hz physics |
| `sim.physx.gpu_max_rigid_patch_count` | 5 × 2¹⁷ | 4× default, headroom for 8k+ envs |
| `sim.physx.gpu_total_aggregate_pairs_capacity` | 64 × 1024 | 4× default |
| **Robot** | `UR3E_ROBOTIQ_HANDE_CFG` | `activate_contact_sensors=False` |
| **Arm action** | `RelativeJointPositionActionCfg`, scale=0.05 | ~143°/s max joint speed |
| **Gripper action** | `BinaryJointPositionActionCfg`, open=0.0, close=-0.02 | Local USD slider convention |
| **EE frame** | base_link → tool0, offset (0,0,0.119) | Empirical grasp-point approximation |
| **Object** | DexCube, scale 0.5 (~40 mm) | Fits Hand-E 40 mm stroke |
| **Object spawn** | pos (0.4, 0, 0.036), randomized via upstream events | |
| **Goal range** | x ∈ [0.3, 0.45], y ∈ [-0.2, 0.2], z ∈ [0.15, 0.35] | Tightened for UR3e reach |
| **Curriculum** | Disabled (weights stay -1e-4) | Upstream ramp collapsed UR3e |
| **fine_grained kernel** | Gaussian `exp(-(d/std)²)` | Zero gradient at d=0 (open issue 6.1) |

### 7.2 PPO ([`skrl_ppo_cfg.yaml`](config/ur3e_hande/agents/skrl_ppo_cfg.yaml))

| Parameter | Value | Notes |
|---|---|---|
| **Network** | layers [256, 128, 64], elu | Mirrors upstream Franka |
| **Shared backbone** | True (`separate: False`) | |
| **min_log_std** | -2.0 | Floor: std ≥ exp(-2) ≈ 0.135 (collapse prevention) |
| **max_log_std** | 0.0 | Ceiling: std ≤ exp(0) = 1.0 (runaway prevention) |
| **initial_log_std** | -1.0 | std ≈ 0.37 at start |
| **rollouts** | 32 | |
| **learning_epochs** | 8 | |
| **mini_batches** | 4 | |
| **learning_rate** | 3e-4 | |
| **LR scheduler** | KLAdaptiveLR, kl_threshold=0.008, min_lr=1e-5 | |
| **entropy_loss_scale** | 0.005 | Higher (0.01) drove std runaway |
| **value_loss_scale** | 2.0 | |
| **grad_norm_clip** | 1.0 | |
| **ratio_clip** | 0.2 | |
| **timesteps** | 300_000 | |

---

## 8. How to Train and Play

```bash
# Activate the Isaac Lab conda environment
conda activate env_isaaclab

# Train (recommended: 8192 envs for fastest wall-clock)
python scripts/skrl/train.py \
    --task=Isaac-Robots-Lift-Cube-UR3e-HandE-v0 \
    --headless \
    --num_envs 8192

# Watch live progression
tensorboard --logdir logs/skrl/ur3e_hande_lift_cube

# Playback the best checkpoint
python scripts/skrl/play.py \
    --task=Isaac-Robots-Lift-Cube-UR3e-HandE-Play-v0 \
    --checkpoint logs/skrl/ur3e_hande_lift_cube/<RUN_DIR>/checkpoints/best_agent.pt
```

Expected training time on RTX 4070 Ti: ~25 min for 300 000 timesteps with
8 192 envs. Peak reward typically reached between steps 150k and 250k.

---

## 9. Lessons Learned

A list of takeaways that generalize beyond this specific task:

1. **Action representation matters more than action scale.** Switching from
   absolute `JointPositionAction` to `RelativeJointPositionAction` solved
   problems (chatter, lift snap) that no scale tuning could.

2. **Asset compatibility checks are cheap; physics debugging is expensive.**
   The 64 mm-vs-40 mm cube/stroke mismatch was a 30-second check that we
   missed and spent a day debugging as a "reward shape" or "policy collapse"
   problem.

3. **PPO collapse and runaway are mirror images.** Both need explicit
   safeguards: `min_log_std` floor to prevent collapse, `max_log_std` ceiling
   to prevent runaway. Default skrl/Isaac Lab configs only guard one side.

4. **Curriculum ramps are dangerous in PPO.** A 100× or 1000× sudden weight
   change to penalty terms shocks the value function and triggers LR collapse
   via KL-adaptive scheduler. Constant penalties are safer; if you need
   stronger penalties later, train from scratch with them.

5. **Reward kernels are physical objects.** tanh and Gaussian have the same
   maximum but different gradients at d=0, producing physically different
   policies (limit cycle vs. drift). The choice has consequences far beyond
   "looks similar".

6. **Late-stage drift is not a collapse.** It's a separate failure mode where
   the policy degrades without the loss-trace signature of a true collapse.
   Detected by reward decline despite stable std and stable LR.

7. **Visual playback is non-negotiable.** TB metrics can show "all-green" runs
   that are silently exploiting reward bugs (the lateral-cheat grasp gave
   reward 178 while the cube wasn't actually grasped).

---

## 10. Future Work

- Resolve open issue 6.1 (Gaussian drift) — try hybrid kernel or early-stopping.
- Resolve open issue 6.2 (snappy close-cube reach).
- Add a teleop demo capture pipeline using the IK-relative variant + skrl
  recorder for robomimic-compatible datasets.
- Once teleop demos exist, evaluate behavior cloning + PPO fine-tune as a
  way to avoid the from-scratch RL training cost.
- Real-robot transfer evaluation: sim2real gap will primarily come from (a)
  the Hand-E grip force / friction model and (b) the cube physics material.
  Both are explicit in this config and tunable.

---

## References

- Upstream Isaac Lab Franka lift: `/home/urkui-3/Documents/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/`
- Direct-style UR3e Hand-E lift (sibling task): [`tasks/direct/isaac_ur3e_lift_cube/`](../../direct/isaac_ur3e_lift_cube/)
- Manager-based UR3e stack (similar pattern): [`tasks/manager_based/stack/`](../stack/)
- skrl PPO documentation: https://skrl.readthedocs.io
- Isaac Lab manager-based environments: https://isaac-sim.github.io/IsaacLab/main/source/setup/tutorials/

---

*Last updated: 2026-06-08*
