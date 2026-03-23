"""
Procedural labyrinth generator for IsaacLab / Crazyflie RL.

Layout: 4 perimeter cuboid walls  +  Gaussian-noise-scattered cylinder
pillars  +  torus-mesh rings placed at pillar-midpoint chokepoints.

Total colliders per env: ~20 (vs ~336 with the old sliced-wall + cuboid-ring approach).

Usage inside _setup_scene():
    cfg = LabyrinthCfg(size=3.5, seed=None, difficulty=0.5)
    self._labyrinth = LabyrinthBuilder(cfg)
    self._labyrinth.build(self.scene, self._env_origins)
"""
from __future__ import annotations
import dataclasses
import math
import random

import numpy as np
from scipy.ndimage import gaussian_filter

import isaaclab.sim as sim_utils

from .ring_obstacle import RingCfg, RingChallengeCfg, spawn_ring, ring_yaw_tilt_to_quat
from .goal_sampler import OccupancyGrid, GoalSampler


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class LabyrinthCfg:
    size: float = 3.5          # arena bounding box (metres, square)
    wall_height: float = 1.2   # metres above floor
    wall_thickness: float = 0.1
    seed: int | None = None    # None = random layout each run
    difficulty: float = 0.5    # 0.0 = easy (fewer/smaller pillars), 1.0 = hard
    spawn_walls: bool = True   # set False for open arena (rings only)
    n_pillars: int | None = None  # override pillar count (None = derived from difficulty)
    n_rings: int | None = None    # override ring/goal count (None = derived from difficulty)
    pillar_radius_min: float | None = None  # per-pillar radius lower bound (None = difficulty-derived)
    pillar_radius_max: float | None = None  # per-pillar radius upper bound (None = difficulty-derived)
    n_layouts: int = 8       # number of distinct procedural layouts to pre-generate


# ---------------------------------------------------------------------------
# Wall spawning (perimeter cuboids)
# ---------------------------------------------------------------------------

_WALL_CFG = sim_utils.MeshCuboidCfg(
    size=(1.0, 1.0, 1.0),  # overridden per call via dataclasses.replace
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
    cfg = dataclasses.replace(_WALL_CFG, size=size)
    angle = math.radians(rot_deg)
    quat = (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))
    cfg.func(prim_path, cfg, translation=pos, orientation=quat)


def _build_perimeter_walls(
    cfg: LabyrinthCfg,
) -> list[tuple[tuple, tuple, float]]:
    """4 axis-aligned boundary cuboids as (pos, size, rot_deg) tuples."""
    s, h, t = cfg.size, cfg.wall_height, cfg.wall_thickness
    hw = s / 2 + t / 2
    zc = h / 2
    return [
        ((hw,   0.0, zc), (t, s + 2 * t, h), 0.0),
        ((-hw,  0.0, zc), (t, s + 2 * t, h), 0.0),
        ((0.0,  hw,  zc), (s + 2 * t, t, h), 0.0),
        ((0.0, -hw,  zc), (s + 2 * t, t, h), 0.0),
    ]


# ---------------------------------------------------------------------------
# Cylinder pillar spawning
# ---------------------------------------------------------------------------

_PILLAR_CFG = sim_utils.CylinderCfg(
    radius=0.1,   # overridden per call
    height=1.0,   # overridden per call
    axis="Z",
    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.40, 0.45)),
)


def _spawn_pillar(
    prim_path: str,
    pos: tuple[float, float, float],
    radius: float,
    height: float,
) -> None:
    cfg = dataclasses.replace(_PILLAR_CFG, radius=radius, height=height)
    cfg.func(prim_path, cfg, translation=pos)


# ---------------------------------------------------------------------------
# Noise-based pillar placement
# ---------------------------------------------------------------------------

def _place_pillars(
    cfg: LabyrinthCfg,
    rng: random.Random,
) -> list[tuple[float, float, float, float]]:
    """Scatter cylinder pillars using a Gaussian-smoothed noise density map.

    Returns list of (x, y, z_center, radius) in env-local coords.
    """
    s = cfg.size
    h = cfg.wall_height
    n_pillars = cfg.n_pillars if cfg.n_pillars is not None else int(6 + cfg.difficulty * 10)
    r_min = cfg.pillar_radius_min if cfg.pillar_radius_min is not None else 0.08 + cfg.difficulty * 0.03
    r_max = cfg.pillar_radius_max if cfg.pillar_radius_max is not None else 0.08 + cfg.difficulty * 0.09
    r_min, r_max = min(r_min, r_max), max(r_min, r_max)
    spawn_clearance = 0.6                              # keep spawn zone open

    # Generate Gaussian-smoothed noise density map (32×32 grid over arena)
    np_rng = np.random.default_rng(rng.randint(0, 2**31))
    noise = np_rng.random((32, 32))
    density = gaussian_filter(noise, sigma=3.0)
    density /= density.max()

    # Candidate positions: cells in the top 40% of density
    threshold = np.percentile(density, 60)
    candidates = np.argwhere(density >= threshold)  # (K, 2)

    res = s / 32
    half = s / 2

    def cell_to_xy(ci: int, cj: int) -> tuple[float, float]:
        return -half + (ci + 0.5) * res, -half + (cj + 0.5) * res

    order = np_rng.permutation(len(candidates))
    pillars: list[tuple[float, float, float, float]] = []

    for idx in order:
        if len(pillars) >= n_pillars:
            break
        ci, cj = candidates[idx]
        x, y = cell_to_xy(int(ci), int(cj))

        radius = rng.uniform(r_min, r_max)

        # Reject if too close to spawn or arena boundary
        if math.hypot(x, y) < spawn_clearance + radius:
            continue
        if abs(x) > half - radius - 0.1 or abs(y) > half - radius - 0.1:
            continue
        # Reject if too close to another pillar (per-pillar radii)
        if any(math.hypot(x - px, y - py) < (radius + pr + 0.3) for px, py, _, pr in pillars):
            continue

        pillars.append((x, y, h / 2, radius))

    return pillars


# ---------------------------------------------------------------------------
# Ring placement at pillar-midpoint chokepoints
# ---------------------------------------------------------------------------

def _place_rings(
    cfg: LabyrinthCfg,
    pillars: list[tuple[float, float, float, float]],
    perimeter: list[tuple],
    rng: random.Random,
    target: int | None = None,
) -> list[RingCfg]:
    """Place rings at midpoints between neighboring pillar pairs.

    Falls back to random positions if not enough pillar pairs exist.
    """
    n_rings = target if target is not None else int(2 + cfg.difficulty * 4)
    ring_cfg = RingChallengeCfg(
        n_rings=n_rings,
        min_radius=0.25,
        max_radius=0.40,
        max_tilt_deg=cfg.difficulty * 35.0,
        ring_color=(0.95, 0.55, 0.05),
    )
    s = cfg.size
    h = cfg.wall_height
    approach_dist = 0.5

    # Perimeter AABBs for yaw-clearance check (x0, x1, y0, y1)
    wall_aabbs = [
        (cx - lx / 2 - 0.2, cx + lx / 2 + 0.2,
         cy - ly / 2 - 0.2, cy + ly / 2 + 0.2)
        for (cx, cy, _), (lx, ly, _), _ in perimeter
    ]

    # Build ring hints: midpoints between close pillar pairs
    hints: list[tuple[float, float, float]] = []
    pillar_xy = [(px, py) for px, py, *_ in pillars]
    for i, (px, py) in enumerate(pillar_xy):
        for j in range(i + 1, len(pillar_xy)):
            qx, qy = pillar_xy[j]
            if 0.4 < math.hypot(px - qx, py - qy) < 1.5:
                hints.append(((px + qx) / 2, (py + qy) / 2, rng.uniform(0.4, h - 0.3)))

    # Random fallback hints
    for _ in range(ring_cfg.n_rings * 8):
        hints.append((
            rng.uniform(-s / 2 + 0.5, s / 2 - 0.5),
            rng.uniform(-s / 2 + 0.5, s / 2 - 0.5),
            rng.uniform(0.4, h - 0.3),
        ))

    rng.shuffle(hints)

    rings: list[RingCfg] = []
    placed_centers: list[tuple[float, float, float]] = []

    for rx, ry, rz in hints:
        if len(rings) >= ring_cfg.n_rings:
            break
        if math.hypot(rx, ry) < 0.6:
            continue

        # Reject if ring center overlaps a pillar
        if any(
            math.hypot(rx - px, ry - py) < pr + ring_cfg.min_radius / 2 + 0.1
            for px, py, _, pr in pillars
        ):
            continue

        radius = rng.uniform(ring_cfg.min_radius, ring_cfg.max_radius)

        if any(math.hypot(rx - px, ry - py) < radius + 0.5 for px, py, _ in placed_centers):
            continue

        # Find yaw where approach/exit are clear of perimeter walls
        best_yaw = None
        for yaw_cand in [rng.uniform(0, 360) for _ in range(12)]:
            yaw_rad = math.radians(yaw_cand)
            nx = -math.sin(yaw_rad)
            ny = math.cos(yaw_rad)
            ax, ay = rx + nx * approach_dist, ry + ny * approach_dist
            ex, ey = rx - nx * approach_dist, ry - ny * approach_dist
            blocked = any(
                (x0 < ax < x1 and y0 < ay < y1) or (x0 < ex < x1 and y0 < ey < y1)
                for x0, x1, y0, y1 in wall_aabbs
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
            pos=(rx, ry, rz),
            yaw_deg=best_yaw,
            tilt_deg=tilt,
            color=ring_cfg.ring_color,
        ))
        placed_centers.append((rx, ry, rz))

    # Pad to target count with below-floor sentinels (never reachable by drone).
    while len(rings) < ring_cfg.n_rings:
        rings.append(RingCfg(
            radius=0.25, tube_radius=0.045,
            pos=(0.0, 0.0, -1.0),
            yaw_deg=0.0, tilt_deg=0.0,
            color=(0.0, 0.0, 0.0),
        ))

    return rings


# ---------------------------------------------------------------------------
# LabyrinthBuilder
# ---------------------------------------------------------------------------

def _set_translate_op(prim, wx: float, wy: float, wz: float) -> None:
    from pxr import UsdGeom, Gf
    xform = UsdGeom.Xformable(prim)
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(wx, wy, wz))
            return


def _set_orient_op(prim, quat: tuple[float, float, float, float]) -> None:
    """Update the OrientOp on a cloned prim with a new quaternion (w,x,y,z)."""
    from pxr import UsdGeom, Gf
    xform = UsdGeom.Xformable(prim)
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            op.Set(Gf.Quatf(quat[0], quat[1], quat[2], quat[3]))
            return


class LabyrinthBuilder:
    def __init__(self, cfg: LabyrinthCfg):
        self.cfg = cfg
        # Per-layout data (index 0 = canonical env_0 layout)
        self._pillars_per_layout: list[list[tuple]] = []
        self._rings_per_layout:   list[list[RingCfg]] = []
        self._goal_samplers:      list[GoalSampler] = []
        # Backward-compat aliases pointing at layout 0
        self.goal_sampler: GoalSampler | None = None
        self.rings_env0: list[RingCfg] = []

    def build(self, scene, env_origins) -> None:
        """Generate n_layouts distinct mazes, spawn layout-0 into env_0 for cloning."""
        cfg = self.cfg
        n_layouts = cfg.n_layouts
        base_seed = cfg.seed if cfg.seed is not None else random.randint(0, 2**30)

        perimeter = _build_perimeter_walls(cfg)
        # Canonical ring count: same across all layouts for uniform tensor shapes.
        canonical_n_rings = cfg.n_rings if cfg.n_rings is not None else int(2 + cfg.difficulty * 4)

        for layout_id in range(n_layouts):
            rng = random.Random(base_seed + layout_id)
            pillars = _place_pillars(cfg, rng) if cfg.spawn_walls else []
            rings = _place_rings(cfg, pillars, perimeter, rng, target=canonical_n_rings)

            grid = OccupancyGrid(size=cfg.size, resolution=0.12)
            grid.mark_wall_specs(perimeter)
            for px, py, _, pr in pillars:
                grid.mark_aabb(px - pr, px + pr, py - pr, py + pr)
            grid.mark_rings(rings)

            sampler = GoalSampler(
                grid,
                spawn_local=(0.0, 0.0, 0.3),
                min_dist_from_spawn=1.0,
                seed=base_seed + layout_id,
            )
            self._pillars_per_layout.append(pillars)
            self._rings_per_layout.append(rings)
            self._goal_samplers.append(sampler)

        # Backward-compat aliases
        self.rings_env0 = self._rings_per_layout[0]
        self.goal_sampler = self._goal_samplers[0]

        # Pillar slots in env_0 determine canonical count (other layouts pad/hide extras)
        pillars_0 = self._pillars_per_layout[0]

        # Spawn perimeter walls (identical across all layouts)
        if cfg.spawn_walls:
            for wall_idx, (pos, size, rot_deg) in enumerate(perimeter):
                _spawn_wall(
                    f"/World/envs/env_0/labyrinth/wall_{wall_idx:03d}",
                    pos, size, rot_deg,
                )
            # Spawn layout-0 pillars into env_0
            for pillar_idx, (px, py, pz, pr) in enumerate(pillars_0):
                _spawn_pillar(
                    f"/World/envs/env_0/labyrinth/pillar_{pillar_idx:03d}",
                    (px, py, pz),
                    radius=pr,
                    height=cfg.wall_height,
                )

        # Spawn layout-0 ring meshes into env_0
        for ring_idx, ring in enumerate(self._rings_per_layout[0]):
            spawn_ring(
                f"/World/envs/env_0/labyrinth/ring_{ring_idx:02d}",
                ring,
                (0.0, 0.0, 0.0),
            )

    def apply_layouts_to_envs(self, num_envs: int, env_origins) -> None:
        """Reposition each env's pillars and rings to its assigned layout.

        Must be called AFTER scene.clone_environments().  Layout-0 envs are
        already correct (they received env_0's geometry during cloning).
        """
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        n_layouts = self.cfg.n_layouts
        canonical_n_pillars = len(self._pillars_per_layout[0])
        canonical_n_rings = len(self._rings_per_layout[0])

        pillars_missing = rings_missing = 0

        for env_id in range(num_envs):
            layout_id = env_id % n_layouts
            if layout_id == 0:
                continue  # already correct from cloning

            pillars = self._pillars_per_layout[layout_id]
            rings = self._rings_per_layout[layout_id]

            # Reposition pillars — translateOp is LOCAL (parent xform carries env offset)
            for i in range(canonical_n_pillars):
                prim = stage.GetPrimAtPath(
                    f"/World/envs/env_{env_id}/labyrinth/pillar_{i:03d}"
                )
                if not prim.IsValid():
                    pillars_missing += 1
                    continue
                if i < len(pillars):
                    px, py, pz, _ = pillars[i]
                    _set_translate_op(prim, px, py, pz)
                else:
                    # Hide surplus pillar below the floor (local coords)
                    _set_translate_op(prim, 0.0, 0.0, -10.0)

            # Reposition and reorient rings — translateOp is LOCAL
            for j in range(canonical_n_rings):
                prim = stage.GetPrimAtPath(
                    f"/World/envs/env_{env_id}/labyrinth/ring_{j:02d}"
                )
                if not prim.IsValid():
                    rings_missing += 1
                    continue
                ring = rings[j]
                _set_translate_op(prim, ring.pos[0], ring.pos[1], ring.pos[2])
                quat = ring_yaw_tilt_to_quat(ring.yaw_deg, ring.tilt_deg)
                _set_orient_op(prim, quat)

        if pillars_missing + rings_missing > 0:
            print(
                f"[LabyrinthBuilder] WARNING apply_layouts_to_envs: "
                f"pillars missing={pillars_missing}, rings missing={rings_missing}"
            )

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
