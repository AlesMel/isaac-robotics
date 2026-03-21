# Isaac Labyrinth Environment

Crazyflie 2.x quadrotor navigating procedurally generated mazes with cylinder pillars and torus rings as waypoints. Supports two sensor modes: camera (asymmetric actor-critic with CNN) and lidar-only (flat MLP).

**Gym ID:** `Isaac-Robots-Labyrinth-Direct-v0`

## Action Space

| Index | Description | Range |
|-------|-------------|-------|
| 0 | Thrust (normalized) | [-1, 1] &rarr; [0, 2&times;weight] |
| 1 | Roll moment | [-1, 1] &times; moment_scale |
| 2 | Pitch moment | [-1, 1] &times; moment_scale |
| 3 | Yaw moment | [-1, 1] &times; moment_scale |

- `thrust_to_weight = 2.0` &mdash; action 0 maps to 0&ndash;2&times; body weight via `(a+1)/2`
- `moment_scale = 0.01`
- Physics at 100 Hz, policy at 50 Hz (decimation = 2)

## Observation Space

### Camera mode (asymmetric actor-critic)

**Actor observation** &mdash; `12 + frame_stack * H * W` (default: 12 + 4&times;64&times;64 = 16396)

| Slice | Dim | Description |
|-------|-----|-------------|
| `[0:3]` | 3 | Linear velocity (body frame) |
| `[3:6]` | 3 | Angular velocity (body frame) |
| `[6:9]` | 3 | Projected gravity (body frame) |
| `[9:12]` | 3 | Goal position relative to body |
| `[12:]` | frame_stack &times; H &times; W | Stacked grayscale camera frames, normalized to [0, 1] |

Camera: Himax HM01B0 monochrome (AI Deck 1.1), downscaled to 64&times;64 by default. Grayscale conversion: `0.299R + 0.587G + 0.114B`. Frame buffer shifts left each step (oldest dropped, newest appended).

**Critic observation (privileged)** &mdash; 11D

| Slice | Dim | Description |
|-------|-----|-------------|
| `[0:3]` | 3 | Goal position relative to body |
| `[3:6]` | 3 | Linear velocity (body frame) |
| `[6:9]` | 3 | Angular velocity (body frame) |
| `[9]` | 1 | Geodesic distance to current waypoint (metres, from 2D Dijkstra field) |
| `[10]` | 1 | Nearest obstacle distance (min lidar range, metres) |

### Lidar-only mode (flat MLP)

**Policy observation** &mdash; 18D

| Slice | Dim | Description |
|-------|-----|-------------|
| `[0:12]` | 12 | Same proprio as camera mode |
| `[12:18]` | 6 | ToF range readings: front, back, left, right, up, down (normalized by max range 4m, clamped to [0, 1]) |

Lidar-only mode is active when `cfg.camera = None`. Sensor noise (`domain_rand.sensor_noise_std`) is added to lidar readings during training.

## Reward Function

| Component | Scale | Formula | dt-scaled |
|-----------|-------|---------|-----------|
| **path_progress** | +16.0 | Change in geodesic distance to current waypoint, clamped to [-1, 1] | No |
| **goal_reached** | +10.0 | Binary: 1 when within 0.25m of waypoint center | No |
| **alive** | +0.5 | Binary: 1 when speed > 0.1 m/s | Yes |
| **proximity** | -5.0 | `clamp(1 - min_range / 0.3, min=0)` wall danger penalty | Yes |
| **tilt** | -0.1 | `1 + projected_gravity_z` (0 when level, 2 when inverted) | Yes |
| **lin_vel** | -0.01 | Squared linear velocity magnitude | Yes |
| **ang_vel** | -0.01 | Squared angular velocity magnitude | Yes |
| **action_smoothness** | -0.03 | Squared difference between consecutive actions | Yes |

- **path_progress** is the primary navigation signal. It uses a 2D Dijkstra distance field precomputed per waypoint, providing dense reward through maze corridors.
- **dt-scaled** rewards are multiplied by `step_dt` for framerate independence.
- Waypoints are ring centers visited in sequence; after all rings, the index wraps for continuous training.

## Termination Conditions

| Condition | Type |
|-----------|------|
| Episode timeout (20s default) | Truncation |
| Height < 0.1m or > wall_height - 0.05m | Termination |
| XY out of arena bounds | Termination |
| Contact force > 0.5 N (after 0.3s grace period) | Termination |
| NaN / Inf in robot state | Termination |

## Network Architectures

### NatureCnnPolicy (actor)

Used in camera mode. Processes stacked grayscale frames through a NatureCNN encoder, concatenates with proprioceptive features, then outputs actions via an MLP.

```
Input: proprio(12) + frames(4 x 64 x 64)
                    |
         +----------+----------+
         |                     |
    proprio(12)         frames(4, 64, 64)
         |                     |
         |              Conv2d(4 -> 32, 8x8, stride=4) + ReLU
         |              Conv2d(32 -> 64, 4x4, stride=2) + ReLU
         |              Conv2d(64 -> 64, 3x3, stride=1) + ReLU
         |              Flatten
         |              Linear(cnn_out -> 512) + ReLU
         |                     |
         +-------concat--------+
                   |
            Linear(524 -> 256) + ReLU
            Linear(256 -> 128) + ReLU
            Linear(128 -> 4)  [action mean]
                   +
            Learnable log_std(4)
```

Output: Gaussian distribution over 4D action space.

### MlpCritic (critic)

Operates on the 11D privileged state (goal, velocities, geodesic distance, obstacle distance).

```
Input: privileged_state(11)
         |
    Linear(11 -> 256) + ELU
    Linear(256 -> 256) + ELU
    Linear(256 -> 1)   [value estimate]
```

### Flat MLP mode (lidar-only)

When no camera is configured, the generic skrl PPO runner uses symmetric MLP models defined in `skrl_ppo_cfg.yaml`:

- **Policy:** Linear(18 &rarr; 256 &rarr; 256 &rarr; 128, Tanh) &rarr; Linear(128 &rarr; 4)
- **Value:** Linear(11 &rarr; 256 &rarr; 256, ELU) &rarr; Linear(256 &rarr; 1), reading from `STATES`

## Domain Randomization

All disabled by default (`DomainRandomizationCfg`):

| Parameter | Description |
|-----------|-------------|
| `thrust_noise_std` | Multiplicative Gaussian noise on thrust: `thrust *= (1 + N(0, std))` |
| `sensor_noise_std` | Additive Gaussian noise on lidar readings |
| `mass_randomization_pct` | Uniform &plusmn;pct scaling of effective body weight for thrust computation |

## Maze Generation

Procedural layouts via `LabyrinthBuilder`:

- **Arena:** 3.5m &times; 3.5m with 1.2m walls
- **Pillars:** Gaussian-smoothed noise density map, 6&ndash;16 cylinders (scales with difficulty)
- **Rings:** Torus meshes at pillar-midpoint chokepoints, 2&ndash;6 per layout (scales with difficulty)
- **Multi-layout:** `n_layouts` distinct mazes are pre-generated; envs are assigned layouts round-robin
- **Geodesic fields:** Per-waypoint 2D Dijkstra distance fields on a 0.12m resolution occupancy grid
