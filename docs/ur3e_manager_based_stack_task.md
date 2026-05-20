# UR3e Manager-Based Cube Stacking Task

## Motivation

The UR3e manager-based stack task is intended to train a simulated Universal Robots UR3e arm to stack three cubes on a table using a suction-style end effector. The task mirrors Isaac Lab's manager-based manipulation stack examples, but adapts the robot, TCP frame, and reward design to the local UR3e asset used in this repository.

The original Isaac Lab stack examples are well suited for imitation learning with robomimic when demonstration data is available. In this project, no demonstration dataset is assumed. For that reason, the task also exposes a `skrl` PPO entry point and includes shaped rewards so that the policy can be trained from scratch.

The practical goal is to create a reusable sim training task for UR3e manipulation:

- validate the local UR3e USD asset in a contact-rich tabletop scenario;
- train a policy without manually collecting robomimic demonstrations;
- keep compatibility with Isaac Lab's manager-based environment structure;
- preserve a future path for behavior cloning if demonstrations are collected later.

## Environment Description

The environment is registered as two Gym tasks:

- `Isaac-Robots-Stack-Cube-UR3e-Long-Suction-IK-Rel-v0`
- `Isaac-Robots-Stack-Cube-UR3e-Short-Suction-IK-Rel-v0`

Both tasks use `isaaclab.envs:ManagerBasedRLEnv` and the local manager-based UR3e stack configuration. The difference between the two tasks is the suction offset used for the end effector:

| Variant | TCP body | Suction offset |
| --- | --- | --- |
| Long suction | `tool0` | `0.22 m` |
| Short suction | `tool0` | `0.1585 m` / `0.159 m` |

The scene contains:

- one UR3e articulation loaded from the local `UR3E_CFG`;
- one Seattle lab table;
- three rigid cube objects: `cube_1`, `cube_2`, and `cube_3`;
- a ground plane and dome light;
- a `FrameTransformer` tracking the UR3e `tool0` TCP frame;
- a surface gripper attached under `{ENV_REGEX_NS}/Robot/tool0/SurfaceGripper`.

Isaac Lab's surface gripper backend currently requires CPU simulation, so the environment sets both `self.device` and `self.sim.device` to `cpu`.

## Action And Observation Space

The task uses relative differential inverse kinematics for arm motion:

- arm action: relative end-effector pose command through `DifferentialInverseKinematicsActionCfg`;
- gripper action: binary surface-gripper command through `SurfaceGripperBinaryActionCfg`;
- IK body: `tool0`;
- IK method: damped least squares (`dls`).

For PPO, the policy observation group is configured with `concatenate_terms = True`. This gives `skrl` a flat observation tensor instead of the dictionary-style observation layout used by robomimic.

The policy observation is intentionally low-dimensional and camera-free. The `object` term is used as a pose-provider contract: during RL training it reads privileged cube positions from sim; later it can be replaced by a CNN or other perception module that outputs the same coordinates.

The policy observation contains:

- previous action;
- relative joint positions;
- relative joint velocities;
- cube positions in the environment-local frame: `[cube_1_xyz, cube_2_xyz, cube_3_xyz]`;
- end-effector position;
- end-effector quaternion;
- gripper position.

Two environment variables can inject perception-like imperfections while keeping the same observation contract:

- `UR3E_CUBE_POSITION_OBS_NOISE_STD`: Gaussian noise in meters, default `0.0`;
- `UR3E_CUBE_POSITION_OBS_DROPOUT_PROB`: per-cube missing-detection probability, default `0.0`.

## Reward Design

The PPO reward is intentionally shaped. Sparse success-only reward is unlikely to train efficiently for a three-cube contact task from scratch, especially with a suction gripper and no demonstrations. The reward therefore breaks the task into staged behaviors:

1. approach `cube_2`;
2. lift `cube_2`;
3. align `cube_2` over `cube_1`;
4. stack `cube_2` on `cube_1`;
5. approach `cube_3`;
6. lift `cube_3`;
7. align `cube_3` over `cube_2`;
8. stack `cube_3` on `cube_2`;
9. award a full-stack success bonus.

The current reward terms are:

| Reward term | Weight | Purpose |
| --- | ---: | --- |
| `reach_cube_2` | `2.0` | Move the TCP close to `cube_2`. |
| `lift_cube_2` | `4.0` | Lift `cube_2` above table height. |
| `align_cube_2_on_1` | `4.0` | Align `cube_2` horizontally above `cube_1`. |
| `stack_cube_2_on_1` | `10.0` | Reward a valid two-cube stack. |
| `reach_cube_3` | `1.0` | Move the TCP close to `cube_3`. |
| `lift_cube_3` | `3.0` | Lift `cube_3` above table height. |
| `align_cube_3_on_2` | `5.0` | Align `cube_3` horizontally above `cube_2`. |
| `stack_cube_3_on_2` | `15.0` | Reward placing the third cube on the stack. |
| `full_stack` | `30.0` | Reward the complete three-cube stack. |
| `action_rate` | `-0.005` | Penalize abrupt action changes. |
| `joint_vel` | `-0.0005` | Penalize excessive joint velocity. |

### Reward Formulas

The reaching reward uses a Gaussian distance shaping term:

```text
reward = exp(-(distance / std)^2)
```

The lifting reward is clipped between `0` and `1`:

```text
reward = clip((object_z - table_height) / lift_height, 0, 1)
```

The alignment reward combines horizontal alignment with a lifted gate:

```text
xy_reward = exp(-(xy_distance / xy_std)^2)
lifted_gate = clip((upper_z - lower_z) / height_diff, 0, 1)
reward = xy_reward * lifted_gate
```

The stack checks use cube-relative geometry instead of Franka finger-joint state. A pair is considered stacked when:

```text
xy_distance(upper, lower) < 0.05 m
abs((upper_z - lower_z) - 0.0468 m) < 0.01 m
```

The full success check requires both:

```text
cube_2 stacked on cube_1
cube_3 stacked on cube_2
```

## Termination And Success

The task keeps the upstream timeout and cube-dropping terminations. The original stack success condition is replaced with a local success check that does not assume a Franka gripper or finger joints. This is important because the UR3e task uses a surface gripper attached to `tool0`.

An episode is successful when:

- `cube_2` is geometrically stacked on `cube_1`;
- `cube_3` is geometrically stacked on `cube_2`;
- the pairwise stack checks pass their horizontal and vertical thresholds.

## Training Path

The task supports two learning paths:

| Method | Status | Use case |
| --- | --- | --- |
| `skrl` PPO | Active | Train from scratch without demonstrations. |
| robomimic BC | Registered for future use | Train from demonstrations once an HDF5 dataset exists. |

Recommended smoke test:

```bash
python scripts/zero_agent.py --task=Isaac-Robots-Stack-Cube-UR3e-Long-Suction-IK-Rel-v0 --device cpu --num_envs 1
```

Recommended first PPO run:

```bash
python scripts/skrl/train.py --task=Isaac-Robots-Stack-Cube-UR3e-Long-Suction-IK-Rel-v0 --device cpu --num_envs 16
```

The task is expected to be difficult for PPO. If learning is unstable, the recommended progression is:

1. train only the first two-cube stack behavior;
2. tune lift and alignment rewards;
3. re-enable the third cube;
4. increase parallel environments if CPU performance allows;
5. consider scripted or teleoperated demonstrations for robomimic if PPO plateaus.

## Implementation References

Key local files:

- `source/isaac_robots/isaac_robots/tasks/manager_based/stack/config/ur3e_gripper/__init__.py`
- `source/isaac_robots/isaac_robots/tasks/manager_based/stack/config/ur3e_gripper/stack_joint_pos_env_cfg.py`
- `source/isaac_robots/isaac_robots/tasks/manager_based/stack/config/ur3e_gripper/stack_ik_rel_env_cfg.py`
- `source/isaac_robots/isaac_robots/tasks/manager_based/stack/mdp/rewards.py`
- `source/isaac_robots/isaac_robots/tasks/manager_based/stack/config/ur3e_gripper/agents/skrl_ppo_cfg.yaml`

External design reference:

- Isaac Lab manager-based manipulation stack task family.
- Isaac Lab surface gripper tutorial and CPU simulation requirement.
