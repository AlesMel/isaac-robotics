# Voxelizer — Quick Start Guide

Converts a USD scene into a 3D occupancy grid and geodesic distance fields used by the training environment.

---

## Basic Usage

```bash
# From the repo root, activate the IsaacLab conda env first
conda activate isaaclab

# Run against a USD file
python voxelizer.py --usd /path/to/scene.usd --resolution 0.1 --goal -4.5 3.5 7.0

# Run with the built-in sample obstacles (no USD needed)
python voxelizer.py --from-config --resolution 0.1 --goal -4.5 3.5 7.0
```

Output is written to `./voxel_output/` by default.

---

## Adding Goals / Waypoints

Each `--goal x y z` flag adds one waypoint. The order matters — it must match the order in `_goal_offsets` inside `obstacle_nav_env.py`.

**Example: two waypoints**

```bash
python voxelizer.py --usd assets/primhouse2.usd \
    --resolution 0.1 \
    --goal 0.0 -2.0 3.0 \
    --goal -4.5 3.5 7.0
```

The corresponding env config (already set):

```python
self._goal_offsets = torch.tensor(
    [
        [0.0, -2.0, 3.0],   # GOAL0  ← must be first
        [-4.5, 3.5, 7.0],   # GOAL1  ← must be second
    ],
    device=self.device,
)
```

> The environment asserts that the number of goals in `distance_field.npz` matches `_num_waypoints`. If they differ, training will crash with a clear error message asking you to re-run the voxelizer.

---

## All Options

| Flag | Default | Description |
|---|---|---|
| `--usd PATH` | — | USD scene file to voxelize |
| `--from-config` | — | Use built-in sample obstacles instead of a USD |
| `--resolution M` | `0.05` | Voxel size in metres (smaller = finer, slower) |
| `--goal X Y Z` | `[-4.5, 3.5, 7.0]` | Goal position in world coords; repeat for multiple |
| `--z-min Z` | auto | Override minimum Z bound |
| `--z-max Z` | auto | Override maximum Z bound |
| `--output-dir DIR` | `./voxel_output` | Where to write output files |
| `--mode-2d` | off | Also compute a 2D (height-collapsed) distance field |
| `--flight-height M` | `1.0` | Z height used for 2D slice visualisation |

---

## Output Files

| File | Description |
|---|---|
| `distance_field.npz` | Main training file — occupancy + one distance field per goal |
| `occupancy_grid.npy` | Raw 3D boolean occupancy array |
| `distance_field.npy` | Single distance field (metres) |
| `grid_metadata.json` | Resolution, origin, shape, goal positions |
| `maze_visualization.png` | 3D scene render |
| `maze_slices.png` | 2D cross-section slices at several Z heights |

---

## Workflow When Changing Goals

1. Edit `_goal_offsets` in `obstacle_nav_env.py`
2. Re-run the voxelizer with `--goal` flags in the **same order**
3. Start training — the env will load the new `distance_field.npz` automatically

---

## Validation

Use the Jupyter notebook to visually verify the output before training:

```bash
jupyter notebook notebooks/test_visual_voxelizer_and_reward.ipynb
```

It shows occupancy slices, BFS paths from spawn to each goal, and a full episode reward simulation so you can catch unreachable goals early.
