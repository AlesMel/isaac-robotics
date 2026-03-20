# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Isaac Lab extension for training Crazyflie quadrotor navigation policies using PPO (via skrl). Four task variants of increasing complexity share the same robot dynamics but differ in obstacles, waypoint generation, and reward shaping.

## Key Commands

All commands require an Isaac Lab conda environment (`conda activate isaaclab`) and the extension installed in editable mode:

```bash
python -m pip install -e source/isaac_robots
```

**Training** (skrl PPO):
```bash
python scripts/skrl/train.py --task=Isaac-Robots-ObstacleNav-Direct-v0
python scripts/skrl/train.py --task=Isaac-Robots-Eights-Direct-v0
python scripts/skrl/train.py --task=Isaac-Robots-CrazyFlie-Direct-v0
```

**Evaluation/Playback**:
```bash
python scripts/skrl/play.py --task=Isaac-Robots-ObstacleNav-Direct-v0 --checkpoint <path>
```

**Labyrinth training** (CNN asymmetric actor-critic):
```bash
python scripts/labyrinth/train_cnn.py --task=Isaac-Robots-Labyrinth-Direct-v0 --num_envs 512 --headless
python scripts/labyrinth/play_cnn.py --task=Isaac-Robots-Labyrinth-Direct-v0 --checkpoint <path>
python scripts/labyrinth/train_curriculum.py --challenge corridor --num_envs 4096
python scripts/labyrinth/view.py --challenge corridor --difficulty 0.5
python scripts/labyrinth/evaluate.py --checkpoint <path>
```

**Voxelizer** (required before training isaac_obstacle tasks):
```bash
python scripts/voxelizer.py --usd /path/to/scene.usd --resolution 0.1 --goal -4.5 3.5 7.0
python scripts/voxelizer.py --from-config --resolution 0.1 --goal -4.5 3.5 7.0
```

**Other scripts**:
```bash
python scripts/list_envs.py          # List registered Gymnasium task IDs
python scripts/zero_agent.py --task=<TASK>    # Zero-action sanity check
python scripts/random_agent.py --task=<TASK>  # Random-action sanity check
```

**Code formatting**: `pre-commit run --all-files`

## Architecture

### Task Variants

All environments extend Isaac Lab's `DirectRLEnv`. The robot is always a Crazyflie 2.x with 4D action space (thrust + 3-axis moment).

| Task | Gym ID | Envs | Key Feature |
|---|---|---|---|
| **isaac_robots** | `Isaac-Robots-CrazyFlie-Direct-v0` | 64 | Random goal reaching, no obstacles, 12D obs |
| **isaac_eights** | `Isaac-Robots-Eights-Direct-v0` | 4096 | Figure-8 around 2 randomized cuboid obstacles, 16 waypoints, ToF sensors |
| **isaac_obstacle** | `Isaac-Robots-ObstacleNav-Direct-v0` | 4096 | Warehouse navigation using A* on voxelized occupancy grid, ToF sensors |
| **isaac_labyrinth** | `Isaac-Robots-Labyrinth-Direct-v0` | 4096 | Procedural maze with camera sensor, CNN+MLP asymmetric actor-critic |

There is also `Isaac-Robots-No-Warehouse-Direct-v0` — an obstacle variant without the warehouse mesh.

### Environment Code Pattern

Each task variant under `source/isaac_robots/isaac_robots/tasks/direct/` follows the same structure:
- `*_env.py` — Environment class (`_setup_scene`, `_pre_physics_step`, `_compute_rewards`, `_get_observations`, `_reset_idx`)
- `*_env_cfg.py` — Dataclass config (sim params, robot cfg, sensor cfg, scene cfg)
- `cfg/` — Modular configs: `sensors.py` (task-specific sensor setup), plus task-specific files like `camera.py` (labyrinth only). Shared configs (`assets.py`, `tof_pattern.py`) live in `tasks/direct/_shared/`.
- `agents/` — `__init__.py` registers Gymnasium entry points; `skrl_ppo_cfg.yaml` defines PPO hyperparameters

### Observation Space

Base observations (all tasks): `[lin_vel(3), ang_vel(3), projected_gravity(3), goal_pos_body(3)]` = 12D. Tasks with ToF sensors append 6 range values (front/back/left/right/up/down, max 4m).

### Voxelizer → A* Pipeline (isaac_obstacle only)

1. `voxelizer.py` converts a USD scene to `voxel_output/distance_field.npz` (occupancy grid + per-goal distance fields via Dijkstra)
2. The environment loads this at init and runs A* to compute multi-waypoint paths (spawn→goal0→goal1→goal0 cycle)
3. Path progress reward = change in remaining A*-geodesic distance, providing a dense navigation signal through complex geometry
4. `--goal` flags in voxelizer **must match** `_goal_offsets` order in `obstacle_nav_env.py`; mismatch triggers an assertion error

### Training Stack

- **RL library**: skrl (PPO with GaussianMixin policy, DeterministicMixin value, both [256,256] ELU)
- **Config**: YAML files in each task's `agents/` directory
- **Sim**: Isaac Sim physics at 100Hz, policy at 50Hz (decimation=2)
- **Logging**: TensorBoard under `logs/` directory
