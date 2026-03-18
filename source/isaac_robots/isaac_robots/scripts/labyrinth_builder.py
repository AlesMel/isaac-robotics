"""
Procedural labyrinth generator for IsaacLab / Crazyflie RL.
Usage inside _setup_scene():
    cfg = LabyrinthCfg(challenge="corridor", size=6.0, seed=42)
    self._labyrinth = LabyrinthBuilder(cfg)
    self._labyrinth.build(self.scene, self._env_origins)
"""

from __future__ import annotations
import dataclasses
import math
import random
from typing import Literal

import numpy as np
import torch
import isaaclab.sim as sim_utils

from .ring_obstacle import RingCfg, RingChallengeCfg, spawn_ring
from .goal_sampler import OccupancyGrid, GoalSampler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class LabyrinthCfg:
    challenge: Literal[
        "corridor", "gate_slalom", "pillar_forest", "vertical_layers", "room_maze"
    ] = "corridor"
    size: float = 6.0          # bounding box (meters) — env-local square
    wall_height: float = 1.5   # metres above floor
    wall_thickness: float = 0.1
    seed: int = 0
    difficulty: float = 0.5    # 0.0 = easy, 1.0 = hard  (controls gap/density)


# ---------------------------------------------------------------------------
# Primitive helpers
# ---------------------------------------------------------------------------

_WALL_CFG = sim_utils.CuboidCfg(
    size=(1.0, 1.0, 1.0),  # overridden per call
    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.45, 0.5)),
)


def _spawn_wall(
    prim_path: str,
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
    rot_deg: float = 0.0,
) -> None:
    """Spawn a single cuboid wall at env-local position."""
    cfg = dataclasses.replace(_WALL_CFG, size=size)
    angle = math.radians(rot_deg)
    quat = (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))  # yaw only
    cfg.func(prim_path, cfg, translation=pos, orientation=quat)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

WallSpec = tuple[
    tuple[float, float, float],  # (x, y, z) center, env-local
    tuple[float, float, float],  # (lx, ly, lz) size
    float,                       # yaw rotation degrees
]

# (x, y, z) env-local chokepoint position where a ring should be placed.
RingHint = tuple[float, float, float]

# Return type for all challenge functions.
ChallengeResult = tuple[list[WallSpec], list[RingHint]]


# ---------------------------------------------------------------------------
# Challenge modules — each returns (wall_specs, ring_hints)
# ring_hints are the natural chokepoints derived from the wall layout.
# ---------------------------------------------------------------------------

def _challenge_corridor(cfg: LabyrinthCfg, rng: random.Random) -> ChallengeResult:
    """
    Zig-zag corridor: alternating walls from left and right with a gap.
    Ring hints are placed at each gap center so the drone must thread every gap.
    """
    specs: list[WallSpec] = []
    hints: list[RingHint] = []
    s = cfg.size
    n_walls = int(3 + cfg.difficulty * 4)         # 3 to 7 cross-walls
    gap = s * (0.45 - cfg.difficulty * 0.25)       # gap width: 0.45s → 0.20s
    spacing = s / (n_walls + 1)
    t = cfg.wall_thickness
    h = cfg.wall_height
    ring_z_min, ring_z_max = 0.4, h - 0.3         # safe Z range: above floor, below wall top

    for i in range(n_walls):
        y = -s / 2 + spacing * (i + 1)
        from_left = (i % 2 == 0)
        wall_len = s - gap
        cx = (-s / 2 + wall_len / 2) if from_left else (s / 2 - wall_len / 2)
        specs.append(((cx, y, h / 2), (wall_len, t, h), 0.0))

        # Gap is on the opposite side from the wall.
        gap_cx = (s / 2 - gap / 2) if from_left else (-s / 2 + gap / 2)
        ring_z = rng.uniform(ring_z_min, ring_z_max)
        hints.append((gap_cx, y, ring_z))

    return specs, hints


def _challenge_gate_slalom(cfg: LabyrinthCfg, rng: random.Random) -> ChallengeResult:
    """
    Full cross-walls with an offset rectangular gate cut out.
    Ring hints are placed at each gate opening center.
    """
    specs: list[WallSpec] = []
    hints: list[RingHint] = []
    s = cfg.size
    n_gates = int(3 + cfg.difficulty * 3)
    spacing = s / (n_gates + 1)
    gate_w = 0.6 - cfg.difficulty * 0.25          # 0.6 m → 0.35 m
    gate_h = 0.5 - cfg.difficulty * 0.15          # 0.5 m → 0.35 m
    t = cfg.wall_thickness
    full_h = cfg.wall_height

    for i in range(n_gates):
        y = -s / 2 + spacing * (i + 1)
        gate_x = rng.uniform(-s / 2 + 0.4, s / 2 - 0.4)
        gate_z_bot = rng.uniform(0.2, full_h - gate_h - 0.1)

        # Left segment
        left_len = (gate_x - gate_w / 2) - (-s / 2)
        if left_len > 0.05:
            cx = -s / 2 + left_len / 2
            specs.append(((cx, y, full_h / 2), (left_len, t, full_h), 0.0))

        # Right segment
        right_start = gate_x + gate_w / 2
        right_len = s / 2 - right_start
        if right_len > 0.05:
            cx = right_start + right_len / 2
            specs.append(((cx, y, full_h / 2), (right_len, t, full_h), 0.0))

        # Bottom cap (below gate)
        if gate_z_bot > 0.05:
            specs.append(((gate_x, y, gate_z_bot / 2), (gate_w, t, gate_z_bot), 0.0))

        # Top cap (above gate)
        top_z_start = gate_z_bot + gate_h
        top_len = full_h - top_z_start
        if top_len > 0.05:
            cz = top_z_start + top_len / 2
            specs.append(((gate_x, y, cz), (gate_w, t, top_len), 0.0))

        # Ring hint: center of the gate opening
        hints.append((gate_x, y, gate_z_bot + gate_h / 2))

    return specs, hints


def _challenge_pillar_forest(cfg: LabyrinthCfg, rng: random.Random) -> ChallengeResult:
    """
    Random square pillars.
    No structural chokepoints — ring hints are midpoints between pillar pairs.
    """
    specs: list[WallSpec] = []
    s = cfg.size
    n = int(8 + cfg.difficulty * 20)
    pillar_r = 0.12 + cfg.difficulty * 0.08       # 0.12 → 0.20 m radius
    min_dist = pillar_r * 2 + 0.3                  # clearance
    h = cfg.wall_height
    ring_z_min, ring_z_max = 0.4, h - 0.3         # safe Z range: above floor, below wall top
    placed: list[tuple[float, float]] = []

    for _ in range(n * 10):                        # attempts
        if len(placed) >= n:
            break
        x = rng.uniform(-s / 2 + 0.3, s / 2 - 0.3)
        y = rng.uniform(-s / 2 + 0.3, s / 2 - 0.3)
        # Keep centre clear (drone spawn)
        if abs(x) < 0.5 and abs(y) < 0.5:
            continue
        if all(math.hypot(x - px, y - py) >= min_dist for px, py in placed):
            placed.append((x, y))
            side = pillar_r * 2
            specs.append(((x, y, h / 2), (side, side, h), 0.0))

    # Hints: midpoints between neighbouring pillar pairs (up to n_rings).
    n_rings = int(2 + cfg.difficulty * 4)
    hints: list[RingHint] = []
    rng.shuffle(placed)
    for i in range(min(len(placed) - 1, n_rings)):
        ax, ay = placed[i]
        bx, by = placed[i + 1]
        mx, my = (ax + bx) / 2, (ay + by) / 2
        # Nudge slightly toward the center if midpoint is inside a pillar.
        mx = mx * 0.9
        my = my * 0.9
        ring_z = rng.uniform(ring_z_min, ring_z_max)
        hints.append((mx, my, ring_z))

    return specs, hints


def _challenge_vertical_layers(cfg: LabyrinthCfg, rng: random.Random) -> ChallengeResult:
    """
    Horizontal floor/ceiling slabs with alternating gaps — pure altitude challenge.
    Ring hints are placed inside each gap so the drone must thread each one.
    """
    specs: list[WallSpec] = []
    hints: list[RingHint] = []
    s = cfg.size
    n_layers = int(2 + cfg.difficulty * 3)         # 2 to 5 layers
    gap_h = 0.4 - cfg.difficulty * 0.12            # vertical gap: 0.4 → 0.28 m
    layer_t = 0.08
    full_h = cfg.wall_height
    # Note: ring Z is placed inside each gap (z + gap_h/2), so no separate ring_z_min/max needed here.

    layer_zs = sorted(rng.uniform(0.4, full_h - 0.4) for _ in range(n_layers))

    for i, z in enumerate(layer_zs):
        from_left = (i % 2 == 0)
        gap_x = (-s / 4) if from_left else (s / 4)
        gap_w = s * (0.4 - cfg.difficulty * 0.1)

        # Left slab
        left_end = gap_x - gap_w / 2
        left_len = left_end - (-s / 2)
        if left_len > 0.05:
            specs.append(((-s / 2 + left_len / 2, 0.0, z), (left_len, s, layer_t), 0.0))

        # Right slab
        right_start = gap_x + gap_w / 2
        right_len = s / 2 - right_start
        if right_len > 0.05:
            specs.append(((right_start + right_len / 2, 0.0, z), (right_len, s, layer_t), 0.0))

        # Ring hint: inside the gap, at slab height with vertical clearance
        hint_y = rng.uniform(-s / 4, s / 4)
        hints.append((gap_x, hint_y, z + gap_h / 2))

    return specs, hints


def _challenge_room_maze(cfg: LabyrinthCfg, rng: random.Random) -> ChallengeResult:
    """
    BSP binary-space-partition room maze.
    Ring hints are placed at each door opening.
    """
    specs: list[WallSpec] = []
    hints: list[RingHint] = []
    s = cfg.size
    h = cfg.wall_height
    t = cfg.wall_thickness
    min_room = 1.2 + (1.0 - cfg.difficulty) * 0.8  # min room side
    ring_z_min, ring_z_max = 0.4, h - 0.3           # safe Z range: above floor, below wall top

    def split(x0, y0, x1, y1, depth):
        w, d = x1 - x0, y1 - y0
        if depth == 0 or (w < min_room * 2 and d < min_room * 2):
            return

        if w > d:
            if w < min_room * 2:
                return
            sx = rng.uniform(x0 + min_room, x1 - min_room)
            door_y = rng.uniform(y0 + 0.3, y1 - 0.3)
            door_h_size = 0.5 - cfg.difficulty * 0.15
            bot_len = door_y - 0.5 * door_h_size - y0
            if bot_len > 0.05:
                specs.append(((sx, y0 + bot_len / 2, h / 2), (t, bot_len, h), 0.0))
            top_start = door_y + 0.5 * door_h_size
            top_len = y1 - top_start
            if top_len > 0.05:
                specs.append(((sx, top_start + top_len / 2, h / 2), (t, top_len, h), 0.0))
            # Ring hint: at the door opening, approaching along X
            ring_z = rng.uniform(ring_z_min, ring_z_max)
            hints.append((sx, door_y, ring_z))
            split(x0, y0, sx, y1, depth - 1)
            split(sx, y0, x1, y1, depth - 1)
        else:
            if d < min_room * 2:
                return
            sy = rng.uniform(y0 + min_room, y1 - min_room)
            door_x = rng.uniform(x0 + 0.3, x1 - 0.3)
            door_h_size = 0.5 - cfg.difficulty * 0.15
            left_len = door_x - 0.5 * door_h_size - x0
            if left_len > 0.05:
                specs.append(((x0 + left_len / 2, sy, h / 2), (left_len, t, h), 0.0))
            right_start = door_x + 0.5 * door_h_size
            right_len = x1 - right_start
            if right_len > 0.05:
                specs.append(((right_start + right_len / 2, sy, h / 2), (right_len, t, h), 0.0))
            # Ring hint: at the door opening, approaching along Y
            ring_z = rng.uniform(0.4, h - 0.3)
            hints.append((door_x, sy, ring_z))
            split(x0, y0, x1, sy, depth - 1)
            split(x0, sy, x1, y1, depth - 1)

    depth = int(1 + cfg.difficulty * 2)
    split(-s / 2, -s / 2, s / 2, s / 2, depth)
    return specs, hints


# ---------------------------------------------------------------------------
# Spawn-zone wall filter
# ---------------------------------------------------------------------------

def _filter_spawn_zone(
    specs: list[WallSpec],
    spawn_xy: tuple[float, float] = (0.0, 0.0),
    clear_radius: float = 0.5,
) -> list[WallSpec]:
    """Remove wall specs whose XY footprint contains the spawn point."""
    sx, sy = spawn_xy
    result = []
    for (cx, cy, cz), (lx, ly, lz), rot_deg in specs:
        if (cx - lx / 2 <= sx <= cx + lx / 2) and (cy - ly / 2 <= sy <= cy + ly / 2):
            continue
        result.append(((cx, cy, cz), (lx, ly, lz), rot_deg))
    return result


# ---------------------------------------------------------------------------
# Perimeter walls
# ---------------------------------------------------------------------------

def _build_perimeter_walls(cfg: LabyrinthCfg) -> list[WallSpec]:
    """Four boundary walls enclosing the labyrinth arena."""
    s = cfg.size
    h = cfg.wall_height
    t = cfg.wall_thickness
    half = s / 2
    return [
        ((0.0, -half, h / 2), (s + 2 * t, t, h), 0.0),
        ((0.0,  half, h / 2), (s + 2 * t, t, h), 0.0),
        ((-half, 0.0, h / 2), (t, s, h), 0.0),
        (( half, 0.0, h / 2), (t, s, h), 0.0),
    ]


# ---------------------------------------------------------------------------
# Challenge registry
# ---------------------------------------------------------------------------

_CHALLENGES: dict[str, ...] = {
    "corridor":        _challenge_corridor,
    "gate_slalom":     _challenge_gate_slalom,
    "pillar_forest":   _challenge_pillar_forest,
    "vertical_layers": _challenge_vertical_layers,
    "room_maze":       _challenge_room_maze,
}


# ---------------------------------------------------------------------------
# Ring placement driven by wall-derived hints
# ---------------------------------------------------------------------------

def _place_rings(
    cfg: LabyrinthCfg,
    wall_specs: list[WallSpec],
    ring_hints: list[RingHint],
    rng: random.Random,
) -> list[RingCfg]:
    """
    Place rings at positions derived from the maze structure (ring_hints).
    Each hint is a chokepoint (gap, gate, door) generated by the challenge.
    Falls back to random placement only when no hints are available.
    """
    ring_cfg = RingChallengeCfg(
        n_rings=int(2 + cfg.difficulty * 4),
        min_radius=0.30,
        max_radius=0.55,
        max_tilt_deg=cfg.difficulty * 35.0,
        ring_color=(0.95, 0.55, 0.05),
    )
    s = cfg.size
    h = cfg.wall_height

    # Precompute wall AABBs for fast rejection.
    wall_aabbs = []
    for (cx, cy, cz), (lx, ly, lz), _rot in wall_specs:
        wall_aabbs.append((
            cx - lx / 2 - 0.3, cx + lx / 2 + 0.3,
            cy - ly / 2 - 0.3, cy + ly / 2 + 0.3,
            cz - lz / 2,       cz + lz / 2,
        ))

    # Shuffle hints so we don't always use the same subset when there are more
    # hints than n_rings.
    shuffled_hints = list(ring_hints)
    rng.shuffle(shuffled_hints)

    # Pad with random candidates if there are fewer hints than desired rings.
    n_random_extra = max(0, ring_cfg.n_rings - len(shuffled_hints))
    random_candidates = [
        (
            rng.uniform(-s / 2 + 0.8, s / 2 - 0.8),
            rng.uniform(-s / 2 + 0.8, s / 2 - 0.8),
            rng.uniform(0.5, h - 0.2),
        )
        for _ in range(n_random_extra * 20)
    ]
    candidate_positions = shuffled_hints + random_candidates

    rings: list[RingCfg] = []
    placed_centers: list[tuple[float, float, float]] = []

    for rx, ry, rz in candidate_positions:
        if len(rings) >= ring_cfg.n_rings:
            break

        # Skip if too close to drone spawn.
        if abs(rx) < 0.6 and abs(ry) < 0.6:
            continue

        radius = rng.uniform(ring_cfg.min_radius, ring_cfg.max_radius)

        # Reject if center overlaps a wall AABB.
        inside_wall = False
        for x0, x1, y0, y1, z0, z1 in wall_aabbs:
            closest_x = max(x0, min(rx, x1))
            closest_y = max(y0, min(ry, y1))
            if math.hypot(rx - closest_x, ry - closest_y) < radius + 0.15:
                inside_wall = True
                break
        if inside_wall:
            continue

        # Reject if too close to an already-placed ring.
        if any(math.hypot(rx - px, ry - py) < radius + 0.6 for px, py, pz in placed_centers):
            continue

        # Find a yaw where approach and exit points are clear of walls.
        best_yaw = None
        approach_dist = 0.8
        for yaw_cand in [rng.uniform(0, 360) for _ in range(12)]:
            yaw_rad = math.radians(yaw_cand)
            nx = -math.sin(yaw_rad)
            ny = math.cos(yaw_rad)
            ax, ay = rx + nx * approach_dist, ry + ny * approach_dist
            ex, ey = rx - nx * approach_dist, ry - ny * approach_dist
            blocked = any(
                (x0 < ax < x1 and y0 < ay < y1) or (x0 < ex < x1 and y0 < ey < y1)
                for x0, x1, y0, y1, z0, z1 in wall_aabbs
            )
            if not blocked:
                best_yaw = yaw_cand
                break

        if best_yaw is None:
            continue

        tilt = rng.uniform(0, min(ring_cfg.max_tilt_deg, 15.0)) * rng.choice([-1, 1])
        rings.append(RingCfg(
            radius=radius,
            tube_radius=0.045,
            n_segments=24,
            pos=(rx, ry, rz),
            yaw_deg=best_yaw,
            tilt_deg=tilt,
            color=ring_cfg.ring_color,
        ))
        placed_centers.append((rx, ry, rz))

    return rings


# ---------------------------------------------------------------------------
# LabyrinthBuilder
# ---------------------------------------------------------------------------

class LabyrinthBuilder:
    def __init__(self, cfg: LabyrinthCfg):
        self.cfg = cfg
        self.goal_sampler: GoalSampler | None = None
        self.rings_env0: list[RingCfg] = []

    def build(self, scene, env_origins) -> None:
        """Build the labyrinth for env_0 and prepare the goal sampler.

        Generates wall specs and ring hints from the selected challenge, filters
        walls that overlap the spawn zone, places rings at maze chokepoints, builds
        a 2-D occupancy grid, and spawns all USD prims into env_0. After this call
        sample_goals() is available.
        """
        cfg = self.cfg
        fn = _CHALLENGES[cfg.challenge]

        rng0 = random.Random(cfg.seed)
        spawn_xy = (0.0, 0.0)

        # Each challenge now returns (wall_specs, ring_hints).
        wall_specs_raw, ring_hints = fn(cfg, rng0)
        wall_specs_0 = _filter_spawn_zone(
            wall_specs_raw + _build_perimeter_walls(cfg),
            spawn_xy=spawn_xy,
        )

        # Rings are placed at hint positions (maze chokepoints), not randomly.
        rings_0 = _place_rings(cfg, wall_specs_0, ring_hints, rng0)
        self.rings_env0 = rings_0

        grid = OccupancyGrid(size=cfg.size, resolution=0.12)
        grid.mark_wall_specs(wall_specs_0)
        grid.mark_rings(rings_0)

        self.goal_sampler = GoalSampler(
            grid,
            spawn_local=(0.0, 0.0, 0.3),
            min_dist_from_spawn=1.0,
            seed=cfg.seed,
        )

        for wall_idx, (pos_local, size, rot_deg) in enumerate(wall_specs_0):
            prim_path = f"/World/envs/env_0/labyrinth/wall_{wall_idx:03d}"
            _spawn_wall(prim_path, pos_local, size, rot_deg)

        for ring_idx, ring in enumerate(rings_0):
            prim_path = f"/World/envs/env_0/labyrinth/ring_{ring_idx:02d}"
            spawn_ring(prim_path, ring, (0.0, 0.0, 0.0))

    def sample_goals(
        self,
        n_envs: int,
        z_min: float = 0.3,
        z_max: float = 1.4,
    ) -> np.ndarray:
        """Returns (n_envs, 3) array of env-local goal positions."""
        if self.goal_sampler is None:
            raise RuntimeError("Call build() before sample_goals()")
        return self.goal_sampler.sample(n_envs, z_min=z_min, z_max=z_max)
