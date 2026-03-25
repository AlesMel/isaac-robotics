"""
Procedural racing track generator for IsaacLab / Crazyflie RL.

Layout: 4 perimeter cuboid walls + gate post pairs at each checkpoint + optional
random scatter obstacles (cylinders and boxes) scattered inside the arena.

N control points are placed on a perturbed circle and smoothed with a
Catmull-Rom spline.  At each control point, two gate posts are placed
perpendicular to the local track tangent, spaced track_width apart — just like
real FPV racing circuits.  Between gates the arena is open (plus scatter).

Usage inside _setup_scene():
    cfg = RacingTrackCfg(size=5.0, n_control_points=6, track_width=0.9)
    self._track = RacingTrackBuilder(cfg)
    self._track.build(self.scene, self._env_origins)
"""
from __future__ import annotations

import dataclasses
import math
import random

import numpy as np

import isaaclab.sim as sim_utils


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class RacingTrackCfg:
    size: float = 5.0                # arena side length (m)
    wall_height: float = 1.2         # gate post height (m)
    wall_thickness: float = 0.1      # perimeter wall thickness (m)
    n_control_points: int = 6        # number of gates / checkpoints per lap
    track_width: float = 0.9         # gate opening width (m)
    pillar_radius_min: float = 0.06  # gate post radius lower bound (m)
    pillar_radius_max: float = 0.10  # gate post radius upper bound (m)
    radial_noise: float = 0.25       # fractional perturbation of control point radius
    checkpoint_height: float = 0.5   # z of checkpoint targets (m above floor); used when vary_checkpoint_heights=False
    track_topology: str = "circular"  # "circular" | "figure8"
    seed: int | None = None
    n_layouts: int = 8
    spawn_walls: bool = True
    # Scatter obstacles (random cylinders + boxes placed inside arena between gates)
    n_scatter_obstacles: int = 0
    scatter_radius_min: float = 0.05
    scatter_radius_max: float = 0.12
    scatter_box_prob: float = 0.4
    # Height-varied checkpoints — each gate target at a different altitude
    vary_checkpoint_heights: bool = False
    checkpoint_height_min: float = 0.25
    checkpoint_height_max: float = 0.95
    # Gate crossbars — horizontal cylinder at a random height on each gate
    add_crossbars: bool = False
    crossbar_radius: float = 0.04
    crossbar_height_min: float = 0.35   # random height range for crossbars
    crossbar_height_max: float = 1.05
    # Hanging bars — standalone horizontal obstacles between gates at varying heights
    n_hanging_bars: int = 0
    hanging_bar_radius: float = 0.04
    hanging_bar_length_min: float = 0.4  # as fraction of track_width
    hanging_bar_length_max: float = 0.8
    # Oscillating hanging bars (sinusoidal up/down motion)
    oscillate_hanging_bars: bool = False
    osc_amplitude_min: float = 0.10
    osc_amplitude_max: float = 0.25
    osc_frequency_min: float = 0.3
    osc_frequency_max: float = 0.8


# ---------------------------------------------------------------------------
# Wall spawning (perimeter cuboids)
# ---------------------------------------------------------------------------

_WALL_CFG = sim_utils.MeshCuboidCfg(
    size=(1.0, 1.0, 1.0),
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


def _build_perimeter_walls(cfg: RacingTrackCfg) -> list[tuple[tuple, tuple, float]]:
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
# Cylinder / box spawning
# ---------------------------------------------------------------------------

_PILLAR_CFG = sim_utils.CylinderCfg(
    radius=0.1, height=1.0, axis="Z",
    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.60, 0.85)),
)

_SCATTER_BOX_CFG = sim_utils.MeshCuboidCfg(
    size=(1.0, 1.0, 1.0),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.35, 0.25)),
)

_SCATTER_CYL_CFG = sim_utils.CylinderCfg(
    radius=0.1, height=1.0, axis="Z",
    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.65, 0.30, 0.20)),
)


def _spawn_gate_post(prim_path, pos, radius, height):
    cfg = dataclasses.replace(_PILLAR_CFG, radius=radius, height=height)
    cfg.func(prim_path, cfg, translation=pos)


def _spawn_scatter_cylinder(prim_path, pos, radius, height):
    cfg = dataclasses.replace(_SCATTER_CYL_CFG, radius=radius, height=height)
    cfg.func(prim_path, cfg, translation=pos)


_CROSSBAR_CFG = sim_utils.CylinderCfg(
    radius=0.04, height=1.0, axis="X",
    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.80, 0.40)),
)

_HANGING_BAR_CFG = sim_utils.CylinderCfg(
    radius=0.04, height=1.0, axis="X",
    # Kinematic so set_transforms works reliably on the GPU pipeline across resets.
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=True,
        disable_gravity=True,
        linear_damping=100.0,
        angular_damping=100.0,
    ),
    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.40, 0.10)),
)


def _spawn_scatter_box(prim_path, pos, half_size, height, rot_deg=0.0):
    cfg = dataclasses.replace(_SCATTER_BOX_CFG, size=(half_size * 2, half_size * 2, height))
    angle = math.radians(rot_deg)
    quat = (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))
    cfg.func(prim_path, cfg, translation=pos, orientation=quat)


def _spawn_crossbar(prim_path: str, pos: tuple, length: float, radius: float, tang: np.ndarray) -> None:
    cfg = dataclasses.replace(_CROSSBAR_CFG, radius=radius, height=length)
    quat = _tang_to_perp_quat(tang)
    cfg.func(prim_path, cfg, translation=pos, orientation=quat)


def _spawn_hanging_bar(prim_path: str, pos: tuple, length: float, radius: float, tang: np.ndarray) -> None:
    cfg = dataclasses.replace(_HANGING_BAR_CFG, radius=radius, height=length)
    quat = _tang_to_perp_quat(tang)
    cfg.func(prim_path, cfg, translation=pos, orientation=quat)


# ---------------------------------------------------------------------------
# USD transform helper
# ---------------------------------------------------------------------------

def _set_translate_op(prim, wx: float, wy: float, wz: float) -> None:
    from pxr import UsdGeom, Gf
    xform = UsdGeom.Xformable(prim)
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(wx, wy, wz))
            return


def _set_orient_op(prim, quat_wxyz: tuple[float, float, float, float]) -> None:
    from pxr import UsdGeom, Gf
    xform = UsdGeom.Xformable(prim)
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            w, x, y, z = quat_wxyz
            op.Set(Gf.Quatd(w, x, y, z))
            return


def _tang_to_perp_quat(tang: np.ndarray) -> tuple[float, float, float, float]:
    """Quaternion (w,x,y,z) rotating the X-axis to align with the gate perpendicular direction."""
    perp = np.array([-tang[1], tang[0]], dtype=np.float64)
    theta = math.atan2(float(perp[1]), float(perp[0]))
    return (math.cos(theta / 2), 0.0, 0.0, math.sin(theta / 2))


# ---------------------------------------------------------------------------
# Centerline generation
# ---------------------------------------------------------------------------

def _catmull_rom_spline(points: np.ndarray, samples_per_segment: int = 20) -> np.ndarray:
    n = len(points)
    result = []
    for i in range(n):
        p0 = points[(i - 1) % n]
        p1 = points[i]
        p2 = points[(i + 1) % n]
        p3 = points[(i + 2) % n]
        for j in range(samples_per_segment):
            t = j / samples_per_segment
            q = 0.5 * (
                (2 * p1)
                + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3
            )
            result.append(q)
    return np.array(result, dtype=np.float32)


def _generate_circular_centerline(
    cfg: RacingTrackCfg,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (spline_pts (M,2), ctrl_arr (N,2)) for a perturbed closed loop."""
    n = cfg.n_control_points
    base_r = cfg.size * 0.35
    jitter = math.pi / n * 0.4
    ctrl_pts = []
    for i in range(n):
        a = 2 * math.pi * i / n + rng.uniform(-jitter, jitter)
        r = base_r * (1.0 + rng.uniform(-cfg.radial_noise, cfg.radial_noise))
        r = max(r, base_r * 0.3)
        r = min(r, cfg.size * 0.48)
        ctrl_pts.append([r * math.cos(a), r * math.sin(a)])

    ctrl_arr = np.array(ctrl_pts, dtype=np.float32)
    centroid = ctrl_arr.mean(axis=0)
    order = np.argsort(np.arctan2(ctrl_arr[:, 1] - centroid[1], ctrl_arr[:, 0] - centroid[0]))
    ctrl_arr = ctrl_arr[order]

    spline_pts = _catmull_rom_spline(ctrl_arr, samples_per_segment=20)
    return spline_pts, ctrl_arr


def _generate_figure8_centerline(
    cfg: RacingTrackCfg,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (spline_pts (M,2), ctrl_arr (N,2)) for a figure-8 track.

    The left loop is traversed counterclockwise, the right loop clockwise.
    This forces the agent to make both left and right turns — it can never
    converge to a policy of simply spinning in one direction.
    """
    n = max(cfg.n_control_points, 6)
    n_left = n // 2
    n_right = n - n_left

    r = cfg.size * 0.22 * (1.0 + rng.uniform(-0.1, 0.1))
    sep = r * 1.9 * (1.0 + rng.uniform(-0.15, 0.15))   # center-to-center separation
    half = cfg.size / 2 - 0.3

    ctrl_pts = []
    # Left loop — counterclockwise (positive angle direction)
    for i in range(n_left):
        a = 2 * math.pi * i / n_left + rng.uniform(-0.2, 0.2)
        dr = rng.uniform(-r * cfg.radial_noise, r * cfg.radial_noise)
        x = -sep / 2 + (r + dr) * math.cos(a)
        y = (r + dr) * math.sin(a)
        ctrl_pts.append([max(-half, min(half, x)), max(-half, min(half, y))])

    # Right loop — clockwise (negative angle direction)
    for i in range(n_right):
        a = -(2 * math.pi * i / n_right) + rng.uniform(-0.2, 0.2)
        dr = rng.uniform(-r * cfg.radial_noise, r * cfg.radial_noise)
        x = sep / 2 + (r + dr) * math.cos(a)
        y = (r + dr) * math.sin(a)
        ctrl_pts.append([max(-half, min(half, x)), max(-half, min(half, y))])

    ctrl_arr = np.array(ctrl_pts, dtype=np.float32)
    spline_pts = _catmull_rom_spline(ctrl_arr, samples_per_segment=20)
    return spline_pts, ctrl_arr


def _generate_centerline(
    cfg: RacingTrackCfg,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to the appropriate centerline generator based on cfg.track_topology."""
    if cfg.track_topology == "figure8":
        return _generate_figure8_centerline(cfg, rng)
    return _generate_circular_centerline(cfg, rng)


# ---------------------------------------------------------------------------
# Gate placement at each control point
# ---------------------------------------------------------------------------

def _place_gates(
    ctrl_arr: np.ndarray,
    cfg: RacingTrackCfg,
    rng: random.Random,
) -> tuple[list[tuple[float, float, float, float]], np.ndarray, np.ndarray]:
    """Place gate post pairs at each control point, perpendicular to local tangent.

    Returns:
        gate_posts         : list of (x, y, z_center, radius), 2 posts per gate
        tangents           : (N, 2) normalised tangent vectors at each checkpoint
        checkpoint_heights : (N,) z of each checkpoint target
    """
    n = len(ctrl_arr)
    h = cfg.wall_height
    half_w = cfg.track_width / 2
    half = cfg.size / 2 - 0.08  # small margin from boundary
    # Clamp gate centres so both posts always land inside the arena with clearance.
    # The perpendicular offset is half_w, so the centre must stay at least
    # (half_w + pillar_radius_max + small buffer) from every wall.
    cp_margin = half_w + cfg.pillar_radius_max + 0.05
    cp_limit = half - cp_margin

    gate_posts: list[tuple[float, float, float, float]] = []
    tangents = np.zeros((n, 2), dtype=np.float32)
    checkpoint_heights = np.zeros(n, dtype=np.float32)

    for i in range(n):
        # Chord tangent: direction from previous to next control point
        tang = ctrl_arr[(i + 1) % n] - ctrl_arr[(i - 1) % n]
        tang_len = float(np.linalg.norm(tang))
        tang = tang / tang_len if tang_len > 1e-9 else np.array([1.0, 0.0], dtype=np.float32)
        tangents[i] = tang

        perp = np.array([-tang[1], tang[0]], dtype=np.float32)
        # Clamp the gate centre so posts fit symmetrically inside the arena.
        cp = ctrl_arr[i]
        cx = float(np.clip(cp[0], -cp_limit, cp_limit))
        cy = float(np.clip(cp[1], -cp_limit, cp_limit))
        r = rng.uniform(cfg.pillar_radius_min, cfg.pillar_radius_max)

        for sign in (+1.0, -1.0):
            px = cx + sign * float(perp[0]) * half_w
            py = cy + sign * float(perp[1]) * half_w
            gate_posts.append((px, py, h / 2, r))

        if cfg.vary_checkpoint_heights:
            checkpoint_heights[i] = rng.uniform(cfg.checkpoint_height_min, cfg.checkpoint_height_max)
        else:
            checkpoint_heights[i] = cfg.checkpoint_height

    return gate_posts, tangents, checkpoint_heights


# ---------------------------------------------------------------------------
# Scatter + hanging bar placement
# ---------------------------------------------------------------------------

def _place_hanging_bars(
    cfg: RacingTrackCfg,
    ctrl_arr: np.ndarray,
    tangents: np.ndarray,
    rng: random.Random,
) -> list[tuple]:
    """Standalone horizontal bars at random mid-points between consecutive gates.

    Returns list of (cx, cy, base_cz, length, tang, amplitude, frequency, phase).
    amplitude=0 when oscillate_hanging_bars=False.
    """
    if cfg.n_hanging_bars <= 0:
        return []

    n = len(ctrl_arr)
    half = cfg.size / 2 - 0.2
    bars: list[tuple] = []
    for _ in range(n * 4):
        if len(bars) >= cfg.n_hanging_bars:
            break
        i = rng.randint(0, n - 1)
        cp_a = ctrl_arr[i]
        cp_b = ctrl_arr[(i + 1) % n]
        cx = float((cp_a[0] + cp_b[0]) / 2)
        cy = float((cp_a[1] + cp_b[1]) / 2)
        if abs(cx) > half or abs(cy) > half:
            continue
        cz = rng.uniform(0.25, cfg.wall_height - 0.15)
        frac = rng.uniform(cfg.hanging_bar_length_min, cfg.hanging_bar_length_max)
        length = cfg.track_width * frac
        if cfg.oscillate_hanging_bars:
            amp = rng.uniform(cfg.osc_amplitude_min, cfg.osc_amplitude_max)
            freq = rng.uniform(cfg.osc_frequency_min, cfg.osc_frequency_max)
            phase = rng.uniform(0.0, 2 * math.pi)
        else:
            amp, freq, phase = 0.0, 1.0, 0.0
        bars.append((cx, cy, cz, length, tangents[i], amp, freq, phase))
    return bars


def _place_scatter_obstacles(
    cfg: RacingTrackCfg,
    rng: random.Random,
    gate_posts: list[tuple],
    checkpoint_centers: list[tuple[float, float]],
) -> list[tuple[float, float, float, float, bool]]:
    """Place random cylinders/boxes avoiding gate posts, checkpoints, and spawn area.

    Returns list of (x, y, z_center, size, is_box).
    """
    if cfg.n_scatter_obstacles <= 0:
        return []

    half = cfg.size / 2 - 0.2
    spawn_clear = 0.7
    h = cfg.wall_height
    # Minimum clearance between a scatter obstacle center and a checkpoint center.
    # Must be large enough that the drone can pass through the gate without the
    # obstacle blocking the path to the goal.
    cp_clearance = cfg.track_width / 2 + cfg.scatter_radius_max
    obstacles: list[tuple[float, float, float, float, bool]] = []

    for _ in range(cfg.n_scatter_obstacles * 12):
        if len(obstacles) >= cfg.n_scatter_obstacles:
            break
        x = rng.uniform(-half, half)
        y = rng.uniform(-half, half)

        if math.hypot(x, y) < spawn_clear:
            continue

        size = rng.uniform(cfg.scatter_radius_min, cfg.scatter_radius_max)
        is_box = rng.random() < cfg.scatter_box_prob

        # Reject if too close to a gate post
        if any(math.hypot(x - gx, y - gy) < size + gr + 0.3 for gx, gy, _, gr in gate_posts):
            continue
        # Reject if too close to a checkpoint center (goal target)
        if any(math.hypot(x - cx, y - cy) < cp_clearance for cx, cy in checkpoint_centers):
            continue
        # Reject if too close to another scatter obstacle
        if any(math.hypot(x - ox, y - oy) < size + os + 0.25 for ox, oy, _, os, _ in obstacles):
            continue

        obstacles.append((x, y, h / 2, size, is_box))

    return obstacles


# ---------------------------------------------------------------------------
# RacingTrackBuilder
# ---------------------------------------------------------------------------

class RacingTrackBuilder:
    def __init__(self, cfg: RacingTrackCfg):
        self.cfg = cfg
        self._gate_posts_per_layout: list[list[tuple]] = []
        self._scatter_per_layout: list[list[tuple]] = []
        self._centerline_checkpoints_per_layout: list[np.ndarray] = []
        self._checkpoint_tangents_per_layout: list[np.ndarray] = []
        self._crossbars_per_layout: list[list[tuple]] = []
        self._hanging_bars_per_layout: list[list[tuple]] = []

    def build(self, scene, env_origins) -> None:
        """Generate n_layouts distinct tracks, spawn layout-0 into env_0 for cloning."""
        cfg = self.cfg
        n_layouts = cfg.n_layouts
        base_seed = cfg.seed if cfg.seed is not None else random.randint(0, 2 ** 30)
        perimeter = _build_perimeter_walls(cfg)

        for layout_id in range(n_layouts):
            rng = random.Random(base_seed + layout_id)
            _, ctrl_arr = _generate_centerline(cfg, rng)
            gate_posts, tangents, checkpoint_heights = _place_gates(ctrl_arr, cfg, rng)
            n_cp = len(ctrl_arr)
            checkpoint_centers = [
                (
                    (gate_posts[i * 2][0] + gate_posts[i * 2 + 1][0]) / 2.0,
                    (gate_posts[i * 2][1] + gate_posts[i * 2 + 1][1]) / 2.0,
                )
                for i in range(n_cp)
            ]
            scatter = _place_scatter_obstacles(cfg, rng, gate_posts, checkpoint_centers)

            checkpoints = np.zeros((len(ctrl_arr), 3), dtype=np.float32)
            for i in range(len(ctrl_arr)):
                pa = gate_posts[i * 2]
                pb = gate_posts[i * 2 + 1]
                checkpoints[i, 0] = (pa[0] + pb[0]) / 2.0
                checkpoints[i, 1] = (pa[1] + pb[1]) / 2.0
            checkpoints[:, 2] = checkpoint_heights

            # Crossbars: one per gate, sitting just below the top of the posts
            crossbars: list[tuple] = []
            if cfg.add_crossbars:
                for i in range(len(ctrl_arr)):
                    cz = rng.uniform(cfg.crossbar_height_min, cfg.crossbar_height_max)
                    length = cfg.track_width + 2 * cfg.pillar_radius_max
                    pa = gate_posts[i * 2]
                    pb = gate_posts[i * 2 + 1]
                    cx = (pa[0] + pb[0]) / 2.0
                    cy = (pa[1] + pb[1]) / 2.0
                    crossbars.append((cx, cy, cz, length, tangents[i]))

            hanging = _place_hanging_bars(cfg, ctrl_arr, tangents, rng)

            self._gate_posts_per_layout.append(gate_posts)
            self._scatter_per_layout.append(scatter)
            self._centerline_checkpoints_per_layout.append(checkpoints)
            self._checkpoint_tangents_per_layout.append(tangents)
            self._crossbars_per_layout.append(crossbars)
            self._hanging_bars_per_layout.append(hanging)

        gate_posts_0 = self._gate_posts_per_layout[0]
        scatter_0 = self._scatter_per_layout[0]

        if cfg.spawn_walls:
            for wall_idx, (pos, size, rot_deg) in enumerate(perimeter):
                _spawn_wall(
                    f"/World/envs/env_0/racing_track/wall_{wall_idx:03d}",
                    pos, size, rot_deg,
                )
            for post_idx, (px, py, pz, pr) in enumerate(gate_posts_0):
                _spawn_gate_post(
                    f"/World/envs/env_0/racing_track/gate_{post_idx:03d}",
                    (px, py, pz), radius=pr, height=cfg.wall_height,
                )

        for obs_idx, (ox, oy, oz, size, is_box) in enumerate(scatter_0):
            path = f"/World/envs/env_0/racing_track/scatter_{obs_idx:03d}"
            if is_box:
                _spawn_scatter_box(path, (ox, oy, oz), size, cfg.wall_height,
                                   rot_deg=random.uniform(0, 90))
            else:
                _spawn_scatter_cylinder(path, (ox, oy, oz), radius=size, height=cfg.wall_height)

        # Crossbars (layout 0)
        for cb_idx, (cx, cy, cz, length, tang) in enumerate(self._crossbars_per_layout[0]):
            _spawn_crossbar(
                f"/World/envs/env_0/racing_track/crossbar_{cb_idx:03d}",
                (cx, cy, cz), length, cfg.crossbar_radius, tang,
            )

        # Hanging bars (layout 0) — always spawn cfg.n_hanging_bars slots; pad with off-scene dummies
        canonical_hanging = list(self._hanging_bars_per_layout[0])
        dummy_tang = np.array([1.0, 0.0], dtype=np.float32)
        while len(canonical_hanging) < cfg.n_hanging_bars:
            canonical_hanging.append((0.0, 0.0, -10.0, 0.1, dummy_tang, 0.0, 1.0, 0.0))
        self._hanging_bars_per_layout[0] = canonical_hanging
        for hb_idx, (hx, hy, hz, length, tang, *_osc) in enumerate(canonical_hanging):
            _spawn_hanging_bar(
                f"/World/envs/env_0/racing_track/hanging_{hb_idx:03d}",
                (hx, hy, hz), length, cfg.hanging_bar_radius, tang,
            )

    def apply_layouts_to_envs(self, num_envs: int, env_origins) -> None:
        """Reposition gate posts and scatter obstacles for each non-zero layout env."""
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        n_layouts = self.cfg.n_layouts
        canonical_n_gates = len(self._gate_posts_per_layout[0])
        canonical_n_scatter = len(self._scatter_per_layout[0])
        missing = 0

        for env_id in range(num_envs):
            layout_id = env_id % n_layouts
            if layout_id == 0:
                continue

            gate_posts = self._gate_posts_per_layout[layout_id]
            scatter = self._scatter_per_layout[layout_id]

            for i in range(canonical_n_gates):
                prim = stage.GetPrimAtPath(
                    f"/World/envs/env_{env_id}/racing_track/gate_{i:03d}"
                )
                if not prim.IsValid():
                    missing += 1
                    continue
                if i < len(gate_posts):
                    _set_translate_op(prim, gate_posts[i][0], gate_posts[i][1], gate_posts[i][2])
                else:
                    _set_translate_op(prim, 0.0, 0.0, -10.0)

            for i in range(canonical_n_scatter):
                prim = stage.GetPrimAtPath(
                    f"/World/envs/env_{env_id}/racing_track/scatter_{i:03d}"
                )
                if not prim.IsValid():
                    missing += 1
                    continue
                if i < len(scatter):
                    _set_translate_op(prim, scatter[i][0], scatter[i][1], scatter[i][2])
                else:
                    _set_translate_op(prim, 0.0, 0.0, -10.0)

            # Crossbars (translate + re-orient since gate tangents change per layout)
            crossbars = self._crossbars_per_layout[layout_id]
            canonical_n_crossbars = len(self._crossbars_per_layout[0])
            for i in range(canonical_n_crossbars):
                prim = stage.GetPrimAtPath(
                    f"/World/envs/env_{env_id}/racing_track/crossbar_{i:03d}"
                )
                if not prim.IsValid():
                    missing += 1
                    continue
                if i < len(crossbars):
                    cx, cy, cz, _, tang = crossbars[i]
                    _set_translate_op(prim, cx, cy, cz)
                    _set_orient_op(prim, _tang_to_perp_quat(tang))
                else:
                    _set_translate_op(prim, 0.0, 0.0, -10.0)

            # Hanging bars
            dummy_tang = np.array([1.0, 0.0], dtype=np.float32)
            hanging = list(self._hanging_bars_per_layout[layout_id])
            canonical_n_hanging = len(self._hanging_bars_per_layout[0])
            while len(hanging) < canonical_n_hanging:
                hanging.append((0.0, 0.0, -10.0, 0.1, dummy_tang, 0.0, 1.0, 0.0))
            for i in range(canonical_n_hanging):
                prim = stage.GetPrimAtPath(
                    f"/World/envs/env_{env_id}/racing_track/hanging_{i:03d}"
                )
                if not prim.IsValid():
                    missing += 1
                    continue
                hx, hy, hz, _, tang, *_osc = hanging[i]
                _set_translate_op(prim, hx, hy, hz)
                _set_orient_op(prim, _tang_to_perp_quat(tang))

        if missing > 0:
            print(f"[RacingTrackBuilder] WARNING: {missing} prims missing in apply_layouts_to_envs")
