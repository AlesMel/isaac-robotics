"""
Ring obstacles for the Crazyflie labyrinth.
A ring is N thin cuboid arc-segments arranged around a circle,
leaving a clear circular opening the drone must fly through.

Coordinate convention: ring centre = local origin, ring face normal = +Y
(before tilt rotation is applied). All positions are env-local.
"""

from __future__ import annotations
import dataclasses
import math
from typing import Sequence

import isaaclab.sim as sim_utils


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class RingCfg:
    """One ring instance."""
    # geometry
    radius: float = 0.5          # inner radius of the opening (metres)
    tube_radius: float = 0.05    # thickness of each segment
    n_segments: int = 12         # more = smoother ring
    # pose (env-local)
    pos: tuple[float, float, float] = (0.0, 0.0, 1.0)
    yaw_deg: float = 0.0         # ring faces this compass bearing
    tilt_deg: float = 0.0        # tip ring forward/back (pitch, ±40° max)
    # visuals
    color: tuple[float, float, float] = (0.9, 0.5, 0.1)  # amber


@dataclasses.dataclass
class RingChallengeCfg:
    """Collection of rings for one env."""
    n_rings: int = 3
    min_radius: float = 0.35
    max_radius: float = 0.65
    max_tilt_deg: float = 30.0   # increases with difficulty
    ring_color: tuple[float, float, float] = (0.9, 0.5, 0.1)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _ring_aabb(ring: RingCfg) -> tuple[
    tuple[float, float, float], tuple[float, float, float]
]:
    """
    Axis-aligned bounding box of a ring (env-local).
    Conservative — wraps the full torus including tube.
    Returns (min_xyz, max_xyz).
    """
    r = ring.radius + ring.tube_radius
    cx, cy, cz = ring.pos
    return (cx - r, cy - r, cz - r), (cx + r, cy + r, cz + r)


def _ring_segment_specs(ring: RingCfg) -> list[
    tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float, float]]
]:
    """
    Return list of (translation_env_local, scale, quaternion_wxyz) for
    each cuboid segment making up the ring.

    Segments are placed around the ring circumference. Each segment is a
    thin rectangular block tangent to the circle.
    Segment dimensions: arc_length × tube_diameter × tube_diameter
    """
    specs = []
    n = ring.n_segments
    r = ring.radius
    cx, cy, cz = ring.pos
    yaw = math.radians(ring.yaw_deg)
    tilt = math.radians(ring.tilt_deg)

    # arc length per segment (chord approximation is fine for n>=8)
    arc_len = 2 * math.pi * r / n
    seg_size = (arc_len * 1.05, ring.tube_radius * 2, ring.tube_radius * 2)  # 1.05: 5% overlap so adjacent segments don't leave visible gaps

    for i in range(n):
        angle = 2 * math.pi * i / n

        # position on unit ring in XZ plane (normal = +Y in ring frame)
        # this makes the ring vertical — drone flies through along Y
        rx = math.cos(angle) * r
        ry = 0.0
        rz = math.sin(angle) * r

        # 1. tilt (pitch ring forward/back around ring-local X)
        rx2 = rx
        ry2 = ry * math.cos(tilt) - rz * math.sin(tilt)
        rz2 = ry * math.sin(tilt) + rz * math.cos(tilt)

        # 2. yaw (rotate around world Z to face desired direction)
        rx3 = rx2 * math.cos(yaw) - ry2 * math.sin(yaw)
        ry3 = rx2 * math.sin(yaw) + ry2 * math.cos(yaw)
        rz3 = rz2

        world_pos = (cx + rx3, cy + ry3, cz + rz3)

        # tangent direction in ring frame (XZ plane)
        tx = -math.sin(angle)
        ty = 0.0
        tz = math.cos(angle)

        # apply same rotations to tangent
        tx2 = tx
        ty2 = ty * math.cos(tilt) - tz * math.sin(tilt)
        tz2 = ty * math.sin(tilt) + tz * math.cos(tilt)

        tx3 = tx2 * math.cos(yaw) - ty2 * math.sin(yaw)
        ty3 = tx2 * math.sin(yaw) + ty2 * math.cos(yaw)
        tz3 = tz2

        # quaternion that rotates X-axis onto tangent
        quat = _quat_from_x_to_vec(tx3, ty3, tz3)
        specs.append((world_pos, seg_size, quat))

    return specs


def _quat_from_x_to_vec(vx: float, vy: float, vz: float) -> tuple[float, float, float, float]:
    """Quaternion (w,x,y,z) rotating +X onto the given (unit) vector."""
    norm = math.sqrt(vx**2 + vy**2 + vz**2)
    if norm < 1e-6:
        return (1.0, 0.0, 0.0, 0.0)
    vx, vy, vz = vx / norm, vy / norm, vz / norm
    # cross product of X=(1,0,0) and v
    cx, cy, cz = 0.0, -vz, vy         # (0,0,0) x ... simplifies
    dot = vx                           # X · v
    # handle antiparallel
    if dot < -1.0 + 1e-6:
        return (0.0, 0.0, 1.0, 0.0)  # 180° around Z
    w = 1.0 + dot
    norm2 = math.sqrt(w**2 + cx**2 + cy**2 + cz**2)
    if norm2 < 1e-6:
        return (1.0, 0.0, 0.0, 0.0)
    return (w / norm2, cx / norm2, cy / norm2, cz / norm2)


# ---------------------------------------------------------------------------
# Spawning
# ---------------------------------------------------------------------------

_RING_MATERIAL_CACHE: dict[tuple, sim_utils.PreviewSurfaceCfg] = {}


def spawn_ring(
    base_prim_path: str,
    ring: RingCfg,
    env_origin: tuple[float, float, float],
) -> None:
    """
    Spawn all cuboid segments of one ring into the USD stage.
    base_prim_path: e.g. "/World/envs/env_0/ring_0"
    env_origin: world-space offset for this environment.
    """
    color = ring.color
    if color not in _RING_MATERIAL_CACHE:
        _RING_MATERIAL_CACHE[color] = sim_utils.PreviewSurfaceCfg(
            diffuse_color=color, emissive_color=tuple(c * 0.15 for c in color)  # 0.15: subtle glow, avoids over-brightening the scene
        )
    mat = _RING_MATERIAL_CACHE[color]

    for seg_idx, (local_pos, seg_size, quat) in enumerate(
        _ring_segment_specs(ring)
    ):
        world_pos = (
            env_origin[0] + local_pos[0],
            env_origin[1] + local_pos[1],
            env_origin[2] + local_pos[2],
        )
        cfg = sim_utils.CuboidCfg(
            size=seg_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=mat,
        )
        cfg.func(
            f"{base_prim_path}/seg_{seg_idx:03d}",
            cfg,
            translation=world_pos,
            orientation=quat,
        )