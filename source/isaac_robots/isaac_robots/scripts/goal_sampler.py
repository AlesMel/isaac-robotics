"""
Free-space occupancy grid + BFS reachability + goal sampler.

Works entirely in env-local 2-D (XY slice, ignoring Z for the BFS)
but stores free Z ranges per cell so goal Z is sampled inside real
open space in 3-D.

Usage:
    grid = OccupancyGrid(size=6.0, resolution=0.15)
    grid.mark_wall_specs(wall_specs)          # from LabyrinthBuilder
    grid.mark_rings(ring_list)                # RingCfg list
    sampler = GoalSampler(grid, spawn_local=(0,0,1))
    goals = sampler.sample(n_envs=4096, z_min=0.3, z_max=1.4)
    # goals: np.ndarray shape (n_envs, 3), env-local coords
"""

from __future__ import annotations
import heapq
import math
from collections import deque

import numpy as np
from scipy.ndimage import binary_dilation

from .ring_obstacle import RingCfg


# ---------------------------------------------------------------------------
# Occupancy grid (2-D XY, env-local)
# ---------------------------------------------------------------------------

class OccupancyGrid:
    """
    Binary 2-D grid. Cell = 1 if any obstacle occupies it at any Z.
    Resolution of ~0.15 m is fast and sufficient for rejection sampling.
    """

    def __init__(self, size: float = 6.0, resolution: float = 0.15):
        self.size = size
        self.res = resolution
        n = int(math.ceil(size / resolution))
        self.n = n
        self.grid = np.zeros((n, n), dtype=bool)  # True = occupied
        self._half = size / 2.0

    # --- coordinate helpers ---

    def _to_cell(self, x: float, y: float) -> tuple[int, int]:
        """Convert world XY coordinates (env-local) to grid cell indices (i, j)."""
        ci = int((x + self._half) / self.res)
        cj = int((y + self._half) / self.res)
        return np.clip(ci, 0, self.n - 1), np.clip(cj, 0, self.n - 1)

    def _cell_center(self, ci: int, cj: int) -> tuple[float, float]:
        """Convert grid cell indices back to world XY coordinates (env-local center of cell)."""
        x = -self._half + (ci + 0.5) * self.res
        y = -self._half + (cj + 0.5) * self.res
        return x, y

    # --- obstacle marking ---

    def mark_aabb(
        self,
        x_min: float, x_max: float,
        y_min: float, y_max: float,
        padding: float = 0.05,
    ) -> None:
        """Mark all cells overlapping with an XY AABB as occupied."""
        x_min -= padding
        x_max += padding
        y_min -= padding
        y_max += padding
        ci0, cj0 = self._to_cell(x_min, y_min)
        ci1, cj1 = self._to_cell(x_max, y_max)
        self.grid[ci0 : ci1 + 1, cj0 : cj1 + 1] = True

    def mark_wall_specs(self, specs: list) -> None:
        """
        Mark wall specs from LabyrinthBuilder.
        spec = ((cx, cy, cz), (lx, ly, lz), rot_deg)
        Only uses XY footprint; rotation approximated as AABB for speed.
        """
        for (cx, cy, _cz), (lx, ly, _lz), rot_deg in specs:
            # rotate corners, take AABB
            rot = math.radians(rot_deg)
            cos_r, sin_r = math.cos(rot), math.sin(rot)
            hx, hy = lx / 2, ly / 2
            corners = [
                ( hx * cos_r - hy * sin_r,  hx * sin_r + hy * cos_r),
                (-hx * cos_r - hy * sin_r, -hx * sin_r + hy * cos_r),
                ( hx * cos_r + hy * sin_r,  hx * sin_r - hy * cos_r),
                (-hx * cos_r + hy * sin_r, -hx * sin_r - hy * cos_r),
            ]
            xs = [cx + c[0] for c in corners]
            ys = [cy + c[1] for c in corners]
            self.mark_aabb(min(xs), max(xs), min(ys), max(ys))

    def mark_rings(self, rings: list[RingCfg], padding: float = 0.08) -> None:
        """
        Mark the tube of each ring as occupied in XY.
        We mark an annulus: cells whose distance from ring centre is
        between (radius - tube) and (radius + tube).
        """
        for ring in rings:
            cx, cy, _cz = ring.pos
            r_outer = ring.radius + ring.tube_radius + padding
            r_inner = max(0.0, ring.radius - ring.tube_radius - padding)

            # only scan a bounding square
            ci0, cj0 = self._to_cell(cx - r_outer, cy - r_outer)
            ci1, cj1 = self._to_cell(cx + r_outer, cy + r_outer)
            for ci in range(ci0, ci1 + 1):
                for cj in range(cj0, cj1 + 1):
                    px, py = self._cell_center(ci, cj)
                    dist = math.hypot(px - cx, py - cy)
                    if r_inner <= dist <= r_outer:
                        self.grid[ci, cj] = True

    # --- distance field (Dijkstra, 8-connected) ---

    _NEIGHBORS_8 = (
        (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
        (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
    )

    def compute_distance_field(self, target_xy: tuple[float, float]) -> np.ndarray:
        """Dijkstra distance field from *target_xy* outward.

        Returns (n, n) float32 array in **metres**.  Unreachable / occupied
        cells keep ``np.inf``.
        """
        dist = np.full((self.n, self.n), np.inf, dtype=np.float32)
        ti, tj = self._to_cell(*target_xy)

        # snap target to nearest free cell if it lands inside an obstacle
        if self.grid[ti, tj]:
            free = np.argwhere(~self.grid)
            if len(free) == 0:
                return dist  # entire grid is occupied
            dists_to_free = np.linalg.norm(free - np.array([ti, tj]), axis=1)
            ti, tj = tuple(free[dists_to_free.argmin()])

        dist[ti, tj] = 0.0
        heap: list[tuple[float, int, int]] = [(0.0, ti, tj)]

        while heap:
            d, ci, cj = heapq.heappop(heap)
            if d > dist[ci, cj]:
                continue
            for dci, dcj, step_cost in self._NEIGHBORS_8:
                ni, nj = ci + dci, cj + dcj
                if 0 <= ni < self.n and 0 <= nj < self.n and not self.grid[ni, nj]:
                    new_d = d + step_cost * self.res
                    if new_d < dist[ni, nj]:
                        dist[ni, nj] = new_d
                        heapq.heappush(heap, (new_d, ni, nj))

        return dist

    # --- BFS reachability from drone spawn ---

    def clear_spawn_zone(
        self,
        spawn_xy: tuple[float, float],
        radius_cells: int = 3,
    ) -> None:
        """Force-clear a small zone around spawn so BFS can start."""
        si, sj = self._to_cell(*spawn_xy)
        r = radius_cells
        i0, i1 = max(0, si - r), min(self.n, si + r + 1)
        j0, j1 = max(0, sj - r), min(self.n, sj + r + 1)
        self.grid[i0:i1, j0:j1] = False

    def reachable_mask(
        self,
        spawn_xy: tuple[float, float],
        min_clearance_cells: int = 1,
    ) -> np.ndarray:
        """
        BFS from spawn cell. Returns bool mask of reachable free cells.
        min_clearance_cells: erode obstacles slightly so goals aren't
        placed right against a wall.
        """
        # inflate obstacles slightly for clearance
        inflated = binary_dilation(
            self.grid,
            iterations=min_clearance_cells,
        ) if min_clearance_cells > 0 else self.grid.copy()

        visited = np.zeros((self.n, self.n), dtype=bool)
        si, sj = self._to_cell(*spawn_xy)

        if inflated[si, sj]:
            # spawn is inside obstacle — this should not happen
            return visited

        queue = deque()
        queue.append((si, sj))
        visited[si, sj] = True

        while queue:
            ci, cj = queue.popleft()
            for dci, dcj in ((1,0),(-1,0),(0,1),(0,-1)):
                ni, nj = ci + dci, cj + dcj
                if 0 <= ni < self.n and 0 <= nj < self.n:
                    if not inflated[ni, nj] and not visited[ni, nj]:
                        visited[ni, nj] = True
                        queue.append((ni, nj))

        return visited


# ---------------------------------------------------------------------------
# Goal sampler
# ---------------------------------------------------------------------------

class GoalSampler:
    """
    Samples valid goal positions in env-local space.

    After building the grid and computing reachability, call sample()
    to get a batch of goal positions for all envs simultaneously.
    """

    def __init__(
        self,
        grid: OccupancyGrid,
        spawn_local: tuple[float, float, float] = (0.0, 0.0, 0.3),
        min_dist_from_spawn: float = 1.0,
        seed: int = 0,
    ):
        """Precompute the set of valid goal cells reachable from spawn.

        Tries clearances of 2, 1, and 0 grid cells (in that order) to find
        reachable free space — looser clearance is used automatically when the
        layout is very dense. Goals closer than min_dist_from_spawn to spawn
        are excluded; if nothing remains, the full reachable set is used.
        """
        self.grid = grid
        self.spawn_xy = (spawn_local[0], spawn_local[1])
        self.min_dist = min_dist_from_spawn
        self.rng = np.random.default_rng(seed)

        # ensure spawn cell is free (walls may overlap spawn in some layouts)
        grid.clear_spawn_zone(self.spawn_xy, radius_cells=3)

        # precompute reachable cell list, reducing clearance if needed
        cells = np.empty((0, 2), dtype=np.intp)
        for clearance in (2, 1, 0):
            mask = grid.reachable_mask(self.spawn_xy, min_clearance_cells=clearance)
            cells = np.argwhere(mask)   # (K, 2)
            if len(cells) > 0:
                if clearance < 2:
                    print(
                        f"[WARN] GoalSampler: reduced clearance to {clearance} cell(s) "
                        f"to find reachable space from spawn."
                    )
                break

        if len(cells) == 0:
            raise RuntimeError(
                "GoalSampler: no reachable free cells found! "
                "Check that spawn is not inside an obstacle."
            )

        centers = np.array([
            grid._cell_center(ci, cj) for ci, cj in cells
        ])                              # (K, 2)
        dists = np.linalg.norm(centers - np.array(self.spawn_xy), axis=1)
        far_mask = dists >= min_dist_from_spawn

        self._free_cells = cells[far_mask]
        self._free_centers = centers[far_mask]

        if len(self._free_cells) == 0:
            # fall back to all reachable if nothing is far enough
            self._free_cells = cells
            self._free_centers = centers

    def sample(
        self,
        n: int,
        z_min: float = 0.3,
        z_max: float = 1.4,
        min_spacing: float = 1.0,
    ) -> np.ndarray:
        """
        Returns (n, 3) array of env-local goal positions.
        z is sampled uniformly in [z_min, z_max].
        Goals are spaced at least min_spacing apart (rejection sampling).
        """
        placed: list[np.ndarray] = []
        max_attempts = n * 50

        for _ in range(max_attempts):
            if len(placed) >= n:
                break
            idx = self.rng.integers(0, len(self._free_cells))
            xy = self._free_centers[idx]
            z = self.rng.uniform(z_min, z_max)
            candidate = np.array([xy[0], xy[1], z])

            # enforce minimum spacing between goals
            if placed and min_spacing > 0:
                dists = np.linalg.norm(
                    np.array(placed) - candidate, axis=1
                )
                if np.any(dists < min_spacing):
                    continue

            placed.append(candidate)

        # if we couldn't place enough with spacing, fill randomly
        while len(placed) < n:
            idx = self.rng.integers(0, len(self._free_cells))
            xy = self._free_centers[idx]
            z = self.rng.uniform(z_min, z_max)
            placed.append(np.array([xy[0], xy[1], z]))

        return np.array(placed)

    def sample_ring_waypoints(
        self,
        rings: list[RingCfg],
    ) -> np.ndarray:
        """
        Generate one waypoint per ring: the ring center.

        Returns (n_rings, 3) array of env-local positions.
        """
        goals = [(ring.pos[0], ring.pos[1], ring.pos[2]) for ring in rings]
        if not goals:
            return np.zeros((0, 3), dtype=np.float32)
        return np.array(goals, dtype=np.float32)

    def compute_waypoint_distance_fields(
        self, rings: list[RingCfg],
    ) -> np.ndarray:
        """Compute a 2-D Dijkstra distance field for each ring center.

        Returns ``(n_rings, grid_n, grid_n)`` float32 array in metres.
        """
        waypoints = self.sample_ring_waypoints(rings)
        fields = np.stack([
            self.grid.compute_distance_field((wp[0], wp[1]))
            for wp in waypoints
        ])
        return fields