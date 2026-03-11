#!/usr/bin/env python3
"""
Standalone maze voxelizer & BFS distance field generator.

Reads a USD stage, extracts obstacle meshes, voxelizes the scene into a 3D
occupancy grid, computes geodesic (BFS) distance fields from goal positions,
and saves everything for visualization.

Outputs:
  - occupancy_grid.npy         : 3D boolean array (True = occupied)
  - distance_field.npy         : 3D float array (geodesic dist in meters, inf = unreachable)
  - grid_metadata.json         : grid bounds, resolution, goal position
  - maze_visualization.png     : 3D visualization (pyvista or matplotlib)
  - maze_slices.png            : 2D slice plots at various heights

Usage (inside IsaacLab conda env with USD available):
  python voxelize_maze.py --usd /path/to/your/warehouse.usd \
                          --resolution 0.1 \
                          --goal 3.0 4.0 1.0 \
                          --z-min 0.0 --z-max 2.5

Usage (without USD -- from obstacle config dict):
  python voxelize_maze.py --from-config \
                          --resolution 0.1 \
                          --goal 3.0 4.0 1.0
"""

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path
from scipy.ndimage import binary_dilation

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# 1. OBSTACLE EXTRACTION FROM USD
# ──────────────────────────────────────────────────────────────────────────────

def extract_meshes_from_usd(usd_path: str):
    """
    Extract mesh bounding boxes from a USD stage.
    Returns list of dicts with 'min' and 'max' corners (world space).
    """
    try:
        from pxr import Usd, UsdGeom, Gf
    except ImportError:
        print("ERROR: pxr (OpenUSD) not available. Install via IsaacLab conda env")
        print("       or use --from-config mode with manually defined obstacles.")
        sys.exit(1)

    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise FileNotFoundError(f"Cannot open USD stage: {usd_path}")

    obstacles = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cube) or prim.IsA(UsdGeom.Cylinder):
            imageable = UsdGeom.Imageable(prim)
            bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
            bbox = bbox_cache.ComputeWorldBound(prim)
            box = bbox.ComputeAlignedRange()

            bmin = box.GetMin()
            bmax = box.GetMax()

            obstacles.append({
                "prim_path": str(prim.GetPath()),
                "min": [bmin[0], bmin[1], bmin[2]],
                "max": [bmax[0], bmax[1], bmax[2]],
            })
            print(f"  Found: {prim.GetPath()} "
                  f"bbox=[{bmin[0]:.2f},{bmin[1]:.2f},{bmin[2]:.2f}] -> "
                  f"[{bmax[0]:.2f},{bmax[1]:.2f},{bmax[2]:.2f}]")

    print(f"\nExtracted {len(obstacles)} obstacle primitives from USD.")
    return obstacles


def get_sample_obstacles():
    """
    Sample warehouse obstacle config for testing without USD.
    Adapt this to match your crazyflie_env_cfg.py obstacle definitions.
    """
    obstacles = []

    # ── Warehouse boundary walls ──
    walls = [
        {"min": [-0.1, -5.0, 0.0], "max": [0.1, 5.0, 2.5]},   # west wall
        {"min": [9.9,  -5.0, 0.0], "max": [10.1, 5.0, 2.5]},   # east wall
        {"min": [-0.1, -5.1, 0.0], "max": [10.1, -4.9, 2.5]},  # south wall
        {"min": [-0.1,  4.9, 0.0], "max": [10.1, 5.1, 2.5]},   # north wall
    ]

    # ── Interior walls / shelving units ──
    interior = [
        # Long shelf row 1
        {"min": [2.0, -3.0, 0.0], "max": [2.3,  1.0, 2.0]},
        # Long shelf row 2
        {"min": [4.5, -1.0, 0.0], "max": [4.8,  3.5, 2.0]},
        # Cross barrier
        {"min": [6.0, -4.0, 0.0], "max": [8.0, -3.7, 1.5]},
        # Box stack
        {"min": [7.0,  1.0, 0.0], "max": [8.0,  2.0, 1.8]},
        # Low barrier (drone can fly over)
        {"min": [3.0,  2.0, 0.0], "max": [5.0,  2.3, 0.8]},
        # Hanging obstacle (drone must fly under)
        {"min": [1.0, -2.0, 1.5], "max": [3.5, -1.7, 2.5]},
    ]

    for w in walls:
        w["prim_path"] = "/World/Walls"
        obstacles.append(w)
    for o in interior:
        o["prim_path"] = "/World/Obstacles"
        obstacles.append(o)

    return obstacles


# ──────────────────────────────────────────────────────────────────────────────
# 2. VOXELIZATION
# ──────────────────────────────────────────────────────────────────────────────

def voxelize(obstacles, resolution, bounds=None, padding=0.5):
    """
    Convert obstacle bounding boxes into a 3D occupancy grid.

    Args:
        obstacles: list of dicts with 'min' and 'max' keys (3D coords)
        resolution: voxel edge length in meters
        bounds: optional dict with 'min' and 'max' (auto-computed if None)
        padding: extra meters around the scene

    Returns:
        grid: np.ndarray[bool] shape (nx, ny, nz), True = occupied
        origin: np.ndarray[3] world coords of grid[0,0,0] corner
        resolution: float
    """
    if bounds is None:
        all_mins = np.array([o["min"] for o in obstacles])
        all_maxs = np.array([o["max"] for o in obstacles])
        scene_min = all_mins.min(axis=0) - padding
        scene_max = all_maxs.max(axis=0) + padding
    else:
        scene_min = np.array(bounds["min"])
        scene_max = np.array(bounds["max"])

    origin = scene_min
    dims = np.ceil((scene_max - scene_min) / resolution).astype(int)
    print(f"\nVoxel grid: {dims[0]} x {dims[1]} x {dims[2]} = "
          f"{dims[0]*dims[1]*dims[2]:,} voxels")
    print(f"  Origin: [{origin[0]:.2f}, {origin[1]:.2f}, {origin[2]:.2f}]")
    print(f"  Extent: [{scene_max[0]:.2f}, {scene_max[1]:.2f}, {scene_max[2]:.2f}]")
    print(f"  Resolution: {resolution}m")

    grid = np.zeros(dims, dtype=bool)

    for obs in obstacles:
        # Convert world coords to grid indices
        imin = np.floor((np.array(obs["min"]) - origin) / resolution).astype(int)
        imax = np.ceil((np.array(obs["max"]) - origin) / resolution).astype(int)

        # Clamp to grid bounds
        imin = np.clip(imin, 0, dims - 1)
        imax = np.clip(imax, 0, dims)

        grid[imin[0]:imax[0], imin[1]:imax[1], imin[2]:imax[2]] = True

    occupied = grid.sum()
    total = grid.size
    print(f"  Occupied: {occupied:,} / {total:,} ({100*occupied/total:.1f}%)")

    return grid, origin, resolution


# ──────────────────────────────────────────────────────────────────────────────
# 3. BFS DISTANCE FIELD (3D)
# ──────────────────────────────────────────────────────────────────────────────

def bfs_distance_field_3d(grid, goal_idx):
    """
    Compute geodesic distance from goal through free space using 3D BFS.
    Uses 6-connected neighbors (face-adjacent).

    Args:
        grid: 3D boolean occupancy (True = blocked)
        goal_idx: (gx, gy, gz) grid indices of goal

    Returns:
        dist: 3D float array, distances in voxel units (multiply by resolution for meters)
    """
    nx, ny, nz = grid.shape
    gx, gy, gz = goal_idx

    if grid[gx, gy, gz]:
        print(f"  WARNING: Goal voxel [{gx},{gy},{gz}] is inside an obstacle!")
        print("  Searching for nearest free voxel...")
        # Find nearest free voxel
        free = np.argwhere(~grid)
        dists = np.abs(free - np.array([gx, gy, gz])).sum(axis=1)
        nearest = free[dists.argmin()]
        gx, gy, gz = nearest
        print(f"  Using [{gx},{gy},{gz}] instead.")

    dist = np.full((nx, ny, nz), np.inf, dtype=np.float32)
    dist[gx, gy, gz] = 0

    queue = deque([(gx, gy, gz)])
    neighbors = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]

    visited = 0
    while queue:
        x, y, z = queue.popleft()
        d = dist[x, y, z]
        for dx, dy, dz in neighbors:
            nx2, ny2, nz2 = x+dx, y+dy, z+dz
            if 0 <= nx2 < nx and 0 <= ny2 < ny and 0 <= nz2 < nz:
                if not grid[nx2, ny2, nz2] and dist[nx2, ny2, nz2] == np.inf:
                    dist[nx2, ny2, nz2] = d + 1
                    queue.append((nx2, ny2, nz2))
                    visited += 1

    reachable = np.isfinite(dist).sum() - 1  # exclude goal itself
    free_total = (~grid).sum()
    print(f"  BFS visited {visited:,} voxels")
    print(f"  Reachable from goal: {reachable:,} / {free_total:,} free voxels "
          f"({100*reachable/max(free_total,1):.1f}%)")

    return dist


def bfs_distance_field_2d(grid_2d, goal_idx_2d):
    """
    2D variant for faster iteration when z-axis is simple.
    Collapses 3D grid to 2D by taking any-occupied along z.
    """
    nx, ny = grid_2d.shape
    gx, gy = goal_idx_2d

    if grid_2d[gx, gy]:
        free = np.argwhere(~grid_2d)
        dists = np.abs(free - np.array([gx, gy])).sum(axis=1)
        nearest = free[dists.argmin()]
        gx, gy = nearest
        print(f"  2D goal adjusted to [{gx},{gy}]")

    dist = np.full((nx, ny), np.inf, dtype=np.float32)
    dist[gx, gy] = 0
    queue = deque([(gx, gy)])
    neighbors = [(1,0),(-1,0),(0,1),(0,-1)]

    while queue:
        x, y = queue.popleft()
        d = dist[x, y]
        for dx, dy in neighbors:
            nx2, ny2 = x+dx, y+dy
            if 0 <= nx2 < nx and 0 <= ny2 < ny:
                if not grid_2d[nx2, ny2] and dist[nx2, ny2] == np.inf:
                    dist[nx2, ny2] = d + 1
                    queue.append((nx2, ny2))

    return dist


# ──────────────────────────────────────────────────────────────────────────────
# 4. VISUALIZATION OUTPUTS
# ──────────────────────────────────────────────────────────────────────────────

def save_slice_plots(grid, dist_field, origin, resolution, goal_world,
                     output_path, n_slices=6):
    """Save 2D slice images at multiple z-heights using matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except ImportError:
        print("  matplotlib not available, skipping slice plots")
        return

    nz = grid.shape[2]
    slice_indices = np.linspace(0, nz - 1, n_slices, dtype=int)

    fig, axes = plt.subplots(2, n_slices, figsize=(4 * n_slices, 8))

    for col, zi in enumerate(slice_indices):
        z_world = origin[2] + zi * resolution

        # Occupancy slice
        ax = axes[0, col]
        ax.imshow(grid[:, :, zi].T, origin="lower", cmap="Greys",
                  extent=[origin[0], origin[0] + grid.shape[0]*resolution,
                          origin[1], origin[1] + grid.shape[1]*resolution])
        ax.set_title(f"Occupancy z={z_world:.2f}m")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        # Distance field slice
        ax = axes[1, col]
        d_slice = dist_field[:, :, zi].T * resolution
        d_slice_masked = np.ma.masked_where(np.isinf(d_slice) | grid[:, :, zi].T,
                                             d_slice)
        im = ax.imshow(d_slice_masked, origin="lower", cmap="viridis_r",
                       extent=[origin[0], origin[0] + grid.shape[0]*resolution,
                               origin[1], origin[1] + grid.shape[1]*resolution])
        ax.plot(goal_world[0], goal_world[1], "r*", markersize=12)
        ax.set_title(f"Distance z={z_world:.2f}m")
        ax.set_xlabel("x (m)")
        plt.colorbar(im, ax=ax, label="geodesic dist (m)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved slice plots: {output_path}")


def save_3d_visualization(grid, dist_field, origin, resolution, goal_world,
                          output_path, max_display_voxels=20000, flight_height=1.0):
    """
    Save a 3D visualization of the voxel grid and distance field.
    Tries pyvista first; falls back to matplotlib 3D scatter.
    """
    occupied_coords = np.argwhere(grid)
    if len(occupied_coords) > max_display_voxels:
        step = max(1, len(occupied_coords) // max_display_voxels)
        occupied_coords = occupied_coords[::step]
        print(f"  Downsampled to {len(occupied_coords)} voxels (step={step})")
    voxels_world = occupied_coords * resolution + origin

    flight_z_idx = min(int((flight_height - origin[2]) / resolution), grid.shape[2] - 1)
    floor_dist = dist_field[:, :, flight_z_idx] * resolution
    floor_occ = grid[:, :, flight_z_idx]

    # Build heatmap point arrays for the floor slice
    xs, ys, ds = [], [], []
    heat_step = max(1, max(floor_dist.shape) // 150)
    for xi in range(0, floor_dist.shape[0], heat_step):
        for yi in range(0, floor_dist.shape[1], heat_step):
            if not floor_occ[xi, yi] and np.isfinite(floor_dist[xi, yi]):
                xs.append(origin[0] + xi * resolution)
                ys.append(origin[1] + yi * resolution)
                ds.append(floor_dist[xi, yi])
    xs, ys, ds = np.array(xs), np.array(ys), np.array(ds)
    zs = np.full_like(xs, origin[2] + flight_z_idx * resolution)

    # ── Try pyvista ──
    try:
        import pyvista as pv

        pl = pv.Plotter(off_screen=True, window_size=(1600, 1000))
        pl.set_background("black")

        # Obstacle voxels
        if len(voxels_world) > 0:
            cloud = pv.PolyData(voxels_world.astype(np.float32))
            pl.add_mesh(cloud, color="#546e7a", point_size=resolution * 200,
                        render_points_as_spheres=False, opacity=0.85,
                        label="Obstacles")

        # Distance field heatmap on floor slice
        if len(xs) > 0:
            pts = np.column_stack([xs, ys, zs]).astype(np.float32)
            heat_cloud = pv.PolyData(pts)
            heat_cloud["distance_m"] = ds.astype(np.float32)
            pl.add_mesh(heat_cloud, scalars="distance_m", cmap="viridis_r",
                        point_size=heat_step * resolution * 200,
                        render_points_as_spheres=False, opacity=0.6,
                        scalar_bar_args={"title": "Geodesic dist (m)"})

        # Goal marker
        goal_sphere = pv.Sphere(radius=max(resolution * 3, 0.15),
                                center=goal_world.tolist())
        pl.add_mesh(goal_sphere, color="red", label="Goal")

        pl.add_text(
            f"Grid: {grid.shape[0]}x{grid.shape[1]}x{grid.shape[2]}  "
            f"res={resolution}m  "
            f"goal=[{goal_world[0]:.1f},{goal_world[1]:.1f},{goal_world[2]:.1f}]",
            position="upper_left", font_size=10, color="white",
        )

        scene_center = (origin + np.array(grid.shape) * resolution / 2).tolist()
        scene_size = max(np.array(grid.shape) * resolution)
        pl.camera.focal_point = scene_center
        pl.camera.position = [
            scene_center[0] + scene_size,
            scene_center[1] - scene_size,
            scene_center[2] + scene_size * 0.8,
        ]
        pl.camera.up = (0, 0, 1)

        pl.screenshot(str(output_path))
        print(f"  Saved pyvista 3D view: {output_path}")
        return

    except ImportError:
        print("  pyvista not available, falling back to matplotlib 3D")
    except Exception as e:
        print(f"  pyvista failed ({e}), falling back to matplotlib 3D")

    # ── Fallback: matplotlib 3D ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        import matplotlib.cm as cm
    except ImportError:
        print("  matplotlib not available, skipping 3D visualization")
        return

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    # Obstacle voxels
    if len(voxels_world) > 0:
        ax.scatter(
            voxels_world[:, 0], voxels_world[:, 1], voxels_world[:, 2],
            c="#546e7a", s=2, alpha=0.5, depthshade=True, label="Obstacles",
        )

    # Distance heatmap (floor slice as colored scatter)
    if len(xs) > 0:
        norm_ds = (ds - ds.min()) / (ds.max() - ds.min() + 1e-9)
        colors = cm.viridis_r(norm_ds)
        ax.scatter(xs, ys, zs, c=colors, s=6, alpha=0.4, depthshade=False)

    # Goal
    ax.scatter(*goal_world, c="red", s=200, marker="*", zorder=5, label="Goal")

    ax.set_xlabel("x (m)", color="white")
    ax.set_ylabel("y (m)", color="white")
    ax.set_zlabel("z (m)", color="white")
    ax.tick_params(colors="white")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
    ax.legend(facecolor="#333355", labelcolor="white", loc="upper left")
    ax.set_title(
        f"Voxel grid {grid.shape[0]}x{grid.shape[1]}x{grid.shape[2]}  "
        f"res={resolution}m",
        color="white",
    )

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Saved matplotlib 3D view: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 5. INTEGRATION HELPER: Export for IsaacLab env
# ──────────────────────────────────────────────────────────────────────────────

def save_for_training(grid, dist_field, origin, resolution, output_dir):
    """
    Save data in a format easy to load in your DirectRLEnv.

    In your env's __init__ or _setup_scene:
        data = np.load("distance_field.npz")
        self._dist_field = torch.tensor(data["distance_field"], device=self.device)
        self._grid_origin = torch.tensor(data["origin"], device=self.device)
        self._grid_res = float(data["resolution"])

    Then in _get_rewards:
        pos = self.drone_pos[:, :3]  # (num_envs, 3)
        idx = ((pos - self._grid_origin) / self._grid_res).long()
        idx = idx.clamp(min=0, max=torch.tensor(self._dist_field.shape)-1)
        potential = self._dist_field[idx[:,0], idx[:,1], idx[:,2]] * self._grid_res
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out / "distance_field.npz",
        occupancy=grid,
        distance_field=dist_field,
        origin=origin,
        resolution=np.array([resolution]),
    )
    print(f"  Saved training data: {out / 'distance_field.npz'}")

    # Also save a 2D collapsed version (faster for 2D navigation)
    grid_2d = grid.any(axis=2)
    goal_z_idx = 0  # will be recomputed
    dist_2d = bfs_distance_field_2d(grid_2d, (grid.shape[0]//2, grid.shape[1]//2))

    np.savez_compressed(
        out / "distance_field_2d.npz",
        occupancy_2d=grid_2d,
        distance_field_2d=dist_2d,
        origin=origin[:2],
        resolution=np.array([resolution]),
    )
    print(f"  Saved 2D training data: {out / 'distance_field_2d.npz'}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Voxelize maze from USD and compute BFS distance field")
    parser.add_argument("--usd", type=str, help="Path to USD file")
    parser.add_argument("--from-config", action="store_true",
                        help="Use built-in sample obstacles (no USD needed)")
    parser.add_argument("--resolution", type=float, default=0.05,
                        help="Voxel size in meters (default: 0.1)")
    parser.add_argument("--goal", type=float, nargs=3, default= [-4.5, 3.5, 7.0],
                        help="Goal position x y z in world coords")
    parser.add_argument("--z-min", type=float, default=None,
                        help="Override z-min bound")
    parser.add_argument("--z-max", type=float, default=None,
                        help="Override z-max bound")
    parser.add_argument("--output-dir", type=str, default="./voxel_output",
                        help="Output directory")
    parser.add_argument("--mode-2d", action="store_true",
                        help="Also compute 2D collapsed distance field")
    parser.add_argument("--flight-height", type=float, default=1.0,
                        help="Drone flight height for 2D slice visualization")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Extract obstacles ──
    print("=" * 60)
    print("STEP 1: Extracting obstacles")
    print("=" * 60)
    if args.usd:
        obstacles = extract_meshes_from_usd(args.usd)
    elif args.from_config:
        obstacles = get_sample_obstacles()
        print(f"  Using {len(obstacles)} sample obstacles")
    else:
        print("ERROR: Specify --usd <path> or --from-config")
        sys.exit(1)

    if not obstacles:
        print("ERROR: No obstacles found!")
        sys.exit(1)

    # ── Voxelize ──
    print("\n" + "=" * 60)
    print("STEP 2: Voxelizing scene")
    print("=" * 60)

    bounds = None
    if args.z_min is not None or args.z_max is not None:
        all_mins = np.array([o["min"] for o in obstacles])
        all_maxs = np.array([o["max"] for o in obstacles])
        bmin = all_mins.min(axis=0) - 0.5
        bmax = all_maxs.max(axis=0) + 0.5
        if args.z_min is not None:
            bmin[2] = args.z_min
        if args.z_max is not None:
            bmax[2] = args.z_max
        bounds = {"min": bmin.tolist(), "max": bmax.tolist()}

    grid, origin, res = voxelize(obstacles, args.resolution, bounds)
    
    drone_radius = 0.08  # Crazyflie half-span + margin
    r_voxels = int(np.ceil(drone_radius / res))
    structure = np.ones((2*r_voxels+1,)*3, dtype=bool)
    grid = binary_dilation(grid, structure=structure)
    print(f"  Inflated obstacles by {r_voxels} voxels ({drone_radius}m drone radius)")

    # ── BFS distance field ──
    print("\n" + "=" * 60)
    print("STEP 3: Computing BFS distance field")
    print("=" * 60)

    goal_world = np.array(args.goal)
    goal_idx = np.round((goal_world - origin) / res).astype(int)
    goal_idx = np.clip(goal_idx, 0, np.array(grid.shape) - 1)
    print(f"  Goal world: {goal_world}")
    print(f"  Goal voxel: {goal_idx}")

    dist_field = bfs_distance_field_3d(grid, tuple(goal_idx))

    # ── Save outputs ──
    print("\n" + "=" * 60)
    print("STEP 4: Saving outputs")
    print("=" * 60)

    # Metadata
    meta = {
        "resolution": res,
        "origin": origin.tolist(),
        "grid_shape": list(grid.shape),
        "goal_world": goal_world.tolist(),
        "goal_voxel": goal_idx.tolist(),
        "occupied_voxels": int(grid.sum()),
        "total_voxels": int(grid.size),
        "max_geodesic_dist_m": float(dist_field[np.isfinite(dist_field)].max() * res)
            if np.any(np.isfinite(dist_field) & (dist_field > 0)) else 0,
    }
    with open(output_dir / "grid_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved metadata: {output_dir / 'grid_metadata.json'}")

    # Numpy arrays
    np.save(output_dir / "occupancy_grid.npy", grid)
    np.save(output_dir / "distance_field.npy", dist_field)
    print(f"  Saved occupancy_grid.npy and distance_field.npy")

    # Training-ready format
    save_for_training(grid, dist_field, origin, res, output_dir)

    # Visualizations
    print("\n" + "=" * 60)
    print("STEP 5: Generating visualizations")
    print("=" * 60)

    save_slice_plots(grid, dist_field, origin, res, goal_world,
                     output_dir / "maze_slices.png")

    save_3d_visualization(grid, dist_field, origin, res, goal_world,
                          output_dir / "maze_visualization.png",
                          flight_height=args.flight_height)

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nView {output_dir / 'maze_visualization.png'} for 3D overview.")
    print(f"View {output_dir / 'maze_slices.png'} for 2D cross-sections.")
    print(f"\nTo use in IsaacLab env, load {output_dir / 'distance_field.npz'}")
    print(f"See save_for_training() docstring for integration code.\n")


if __name__ == "__main__":
    main()