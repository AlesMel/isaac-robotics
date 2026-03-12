# Warehouse Navigation Task (`Isaac-Robots-ObstacleNav-Direct-v0`)

Crazyflie quadrotor navigating between waypoints inside a warehouse environment.

## Robot

- **Platform:** Crazyflie 2.x
- **Default spawn:** `(3.5, 4.0, 5.0)` relative to env origin
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

## Geodesic Distance Field

Reward and termination are driven by a **precomputed 3D geodesic distance field** rather than straight-line distance, so the drone is guided around obstacles rather than through them.

### How it is computed (`voxelizer.py`)

1. **Obstacle extraction** — The warehouse USD (`primhouse2.usd`) is opened with OpenUSD (`pxr`). Every `Mesh`, `Cube`, and `Cylinder` prim is traversed and its world-space axis-aligned bounding box is recorded.
2. **Voxelization** — The scene is discretized into a 3D boolean occupancy grid at a configurable resolution (default 0.05 m). Each obstacle bounding box is rasterized into the grid. Obstacles are then **inflated** by the drone's radius (0.08 m, ~2 voxels) using binary dilation so the drone body is treated as a point.
3. **BFS distance field** — A standard 3D breadth-first search using 6-connected (face-adjacent) neighbors propagates outward from each goal voxel through free space. The result is a `float32` array of voxel-step counts. Multiplying by the grid resolution converts to meters.
4. **Output** — `voxel_output/distance_field.npz` stores:
   - `distance_fields`: shape `(num_goals, X, Y, Z)` — BFS steps from each goal
   - `goals_world`: shape `(num_goals, 3)` — goal positions in env-local world frame
   - `origin`: 3D world-frame origin of the grid
   - `resolution`: voxel edge length in meters
   - `grid_metadata.json`: includes `max_geodesic_dist_m` used to normalize the tanh reward

### Runtime lookup

At each policy step, the drone's world position is converted to a grid index (clamped to bounds). The stored BFS step count for the current waypoint is multiplied by `resolution` to get meters:

```python
local = pos_w - env_origin
idx = ((local - grid_origin) / resolution).long()
geo_dist_m = dist_fields[waypoint_idx, ix, iy, iz] * resolution
```

The reward term is the **reduction** in geodesic distance since the previous step (`geo_dist_improvement = prev - current`), scaled by `distance_to_goal_reward_scale * step_dt`.

### Regenerating the distance field

```bash
# From warehouse USD (requires OpenUSD / IsaacLab conda env)
python voxelizer.py \
    --usd source/isaac_robots/isaac_robots/tasks/direct/isaac_obstacle/primhouse2.usd \
    --resolution 0.05 \
    --goal -4.5 3.5 7.0 \
    --output-dir voxel_output

# Without USD (uses built-in sample obstacle config)
python voxelizer.py --from-config --goal -4.5 3.5 7.0
```

Goal positions must match `_goal_offsets` in `obstacle_nav_env.py` exactly (within 0.01 m); the env asserts this on startup.

## Reward

All terms are summed per step. Velocity terms are scaled by `step_dt`.

| Term | Scale | Formula | Purpose |
|------|-------|---------|---------|
| `lin_vel` | -0.05 | `sum(v_b^2) * scale * dt` | Penalize fast linear motion |
| `ang_vel` | -0.01 | `sum(w_b^2) * scale * dt` | Penalize fast rotation |
| `distance_to_goal` | +15.0 | `geodesic_progress_m * scale * dt` | Reward geodesic distance improvement toward goal |
| `goal_reached` | +15.0 | `1.0 * bonus` (when `d < 0.2 m`) | Sparse bonus on waypoint arrival |

`geodesic_progress_m` is the reduction in meters of geodesic distance compared to the previous step. When a waypoint is reached (`Euclidean distance < 0.2 m`), the target advances to the next waypoint (cycling).

## Waypoints

One fixed goal offset relative to each env origin:

| Goal | X | Y | Z |
|------|---|---|---|
| GOAL0 | -4.5 | 3.5 | 7.0 |

Goal offsets are in the env-local frame (added to the env origin). The same distance field is used for all parallel environments. The target resets to GOAL0 on each episode reset.

## Termination

| Condition | Type |
|-----------|------|
| No Euclidean progress toward goal for 2 s (`stuck`) | Death |
| Contact force on body > 0.1 N (`collided`) | Death |
| `t >= episode_length_s` | Timeout |

The stuck check tracks the best (minimum) Euclidean distance to goal seen so far. Progress is defined as improving that best by more than 0.05 m. If no improvement occurs for `2 s / (decimation × dt) = 100` policy steps, the episode terminates.

## Environment Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| `episode_length_s` | 10.0 | Max episode duration |
| `decimation` | 2 | Physics steps per policy step |
| `sim.dt` | 1/100 s | Physics timestep |
| `num_envs` | 4096 | Parallel environments |
| `env_spacing` | 11.0 m | Distance between env origins |
| `thrust_to_weight` | 1.9 | Max thrust as multiple of weight |
| `collision_force_threshold` | 0.1 N | Contact force on body to trigger termination |
| `terrain` | `primhouse2.usd` | USD scene loaded as warehouse |

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
