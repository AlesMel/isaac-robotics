# Warehouse Navigation Task (`Isaac-Robots-ObstacleNav-Direct-v0`)

Crazyflie quadrotor navigating between waypoints inside a warehouse environment.

## Robot

- **Platform:** Crazyflie 2.x
- **Default spawn:** `(1.0, -0.5, 0.3)` relative to env origin
- **Control:** Direct wrench (thrust + moment) applied to the body frame

## Action Space

Continuous `Box(-1, 1)` with 4 dimensions:

| Index | Meaning | Mapping |
|-------|---------|---------|
| 0 | Collective thrust | Mapped to `[0, thrust_to_weight * weight]` via `(a+1)/2` |
| 1 | Roll moment | Scaled by `moment_scale` (0.01) |
| 2 | Pitch moment | Scaled by `moment_scale` (0.01) |
| 3 | Yaw moment | Scaled by `moment_scale` (0.01) |

## Observation Space

Continuous vector of size **12 + lidar_flat_dim** (18 with default multi-ranger):

| Indices | Dim | Description |
|---------|-----|-------------|
| 0-2 | 3 | Linear velocity in body frame |
| 3-5 | 3 | Angular velocity in body frame |
| 6-8 | 3 | Projected gravity in body frame |
| 9-11 | 3 | Goal position in body frame (relative to drone) |
| 12-17 | 6 | Multi-ranger ToF distances (normalized to `[0, 1]`) |

### Multi-ranger sensor (6 rays)

Simulates the Crazyflie Multi-ranger + Flow deck:

| Ray | Direction | Purpose |
|-----|-----------|---------|
| 0 | +X (front) | Front obstacle |
| 1 | -X (back) | Rear obstacle |
| 2 | +Y (left) | Left obstacle |
| 3 | -Y (right) | Right obstacle |
| 4 | +Z (up) | Ceiling |
| 5 | -Z (down) | Ground / altitude (Flow deck) |

- Max range: 4.0 m
- Sensor offset: 2 cm above body origin
- Ranges are clamped and normalized: `range / max_distance`, clipped to `[0, 1]`

## Reward

All terms are summed per step. Velocity terms are scaled by `step_dt`.

| Term | Scale | Formula | Purpose |
|------|-------|---------|---------|
| `lin_vel` | -0.05 | `sum(v_b^2) * scale * dt` | Penalize fast linear motion |
| `ang_vel` | -0.01 | `sum(w_b^2) * scale * dt` | Penalize fast rotation |
| `distance_to_goal` | +15.0 | `(1 - tanh(d / 0.8)) * scale * dt` | Attract toward current waypoint |
| `goal_reached` | +15.0 | `1.0 * bonus` (when `d < 0.2 m`) | Sparse bonus on waypoint arrival |

When a waypoint is reached (`distance < 0.2 m`), the target advances to the next waypoint (cycling through 2 goals).

## Waypoints

Two fixed goals relative to each env origin:

| Goal | X | Y | Z |
|------|---|---|---|
| GOAL0 | 0.0 | 0.6 | 0.8 |
| GOAL1 | -2.0 | 1.6 | 0.12 |

The drone cycles GOAL0 -> GOAL1 -> GOAL0 -> ... indefinitely within an episode.

## Termination

| Condition | Type |
|-----------|------|
| `z < 0.1 m` | Death (too low / crashed) |
| `z > 2.5 m` | Death (too high) |
| `t >= 20 s` | Timeout |

## Environment Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| `episode_length_s` | 20.0 | Max episode duration |
| `decimation` | 2 | Physics steps per policy step |
| `sim.dt` | 1/100 s | Physics timestep |
| `num_envs` | 4096 | Parallel environments |
| `env_spacing` | 5.0 m | Distance between env origins |
| `thrust_to_weight` | 1.9 | Max thrust as multiple of weight |
| `terrain` | `warehouse.usd` | USD scene loaded as terrain |

## Training

Uses **PPO** via skrl with:
- Network: shared 2x256 MLP (ELU activations)
- Rollout length: 32 steps
- Learning rate: 5e-4 (KL-adaptive scheduler)
- Discount: 0.99, GAE lambda: 0.95
- Timesteps: 100,000

```bash
python scripts/skrl/train.py --task Isaac-Robots-ObstacleNav-Direct-v0
```
