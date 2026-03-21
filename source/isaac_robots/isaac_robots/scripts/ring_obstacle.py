"""
Ring obstacles for the Crazyflie labyrinth.

Each ring is a single trimesh torus mesh spawned as one USD prim with
convex-decomposition collision.  This is far cheaper than the old
N-cuboid arc-segment approach (1 prim vs 64 per ring).

Coordinate convention:
  Ring face normal (flythrough axis) = (-sin(yaw), cos(yaw), 0) before tilt.
  All positions are env-local; env_origin is added inside spawn_ring().
"""

from __future__ import annotations
import dataclasses
import math


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class RingCfg:
    """One ring instance."""
    # geometry
    radius: float = 0.5          # inner radius of the opening (metres)
    tube_radius: float = 0.05    # cross-section radius of the torus tube
    n_segments: int = 32         # kept for API compatibility; unused by mesh rings
    # pose (env-local)
    pos: tuple[float, float, float] = (0.0, 0.0, 1.0)
    yaw_deg: float = 0.0         # ring faces this compass bearing
    tilt_deg: float = 0.0        # pitch ring forward/back (±40° max)
    # visuals
    color: tuple[float, float, float] = (0.9, 0.5, 0.1)  # amber


@dataclasses.dataclass
class RingChallengeCfg:
    """Collection of rings for one env."""
    n_rings: int = 3
    min_radius: float = 0.35
    max_radius: float = 0.65
    max_tilt_deg: float = 30.0
    ring_color: tuple[float, float, float] = (0.9, 0.5, 0.1)


# ---------------------------------------------------------------------------
# Geometry helpers (kept for labyrinth_builder compatibility)
# ---------------------------------------------------------------------------

def _ring_aabb(
    ring: RingCfg,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Conservative axis-aligned bounding box of a ring (env-local).
    Returns (min_xyz, max_xyz).
    """
    r = ring.radius + ring.tube_radius
    cx, cy, cz = ring.pos
    return (cx - r, cy - r, cz - r), (cx + r, cy + r, cz + r)


# ---------------------------------------------------------------------------
# Spawning — single USD mesh prim per ring
# ---------------------------------------------------------------------------

def ring_yaw_tilt_to_quat(yaw_deg: float, tilt_deg: float) -> tuple[float, float, float, float]:
    """Compose yaw (Z) and tilt-90 (X) rotations into a single (w,x,y,z) quaternion.

    Equivalent to RotateZ(yaw) * RotateX(tilt - 90) but produces a single
    orthonormal orientation — no risk of non-orthonormal xform matrices.
    """
    # RotateX(tilt - 90)
    ax = math.radians(tilt_deg - 90.0) / 2.0
    qx = (math.cos(ax), math.sin(ax), 0.0, 0.0)
    # RotateZ(yaw)
    az = math.radians(yaw_deg) / 2.0
    qz = (math.cos(az), 0.0, 0.0, math.sin(az))
    # Compose: qz * qx  (outermost * innermost)
    w = qz[0]*qx[0] - qz[1]*qx[1] - qz[2]*qx[2] - qz[3]*qx[3]
    x = qz[0]*qx[1] + qz[1]*qx[0] + qz[2]*qx[3] - qz[3]*qx[2]
    y = qz[0]*qx[2] - qz[1]*qx[3] + qz[2]*qx[0] + qz[3]*qx[1]
    z = qz[0]*qx[3] + qz[1]*qx[2] - qz[2]*qx[1] + qz[3]*qx[0]
    return (w, x, y, z)


def spawn_ring(
    base_prim_path: str,
    ring: RingCfg,
    env_origin: tuple[float, float, float],
) -> None:
    """Spawn a torus ring as a single USD mesh prim with convex-decomposition
    collision and a kinematic rigid body.

    base_prim_path: e.g. "/World/envs/env_0/labyrinth/ring_00"
    env_origin: world-space offset for this environment.
    """
    import trimesh
    from pxr import UsdGeom, UsdPhysics, Gf, Vt
    import omni.usd

    # ------------------------------------------------------------------
    # Build torus mesh
    # ------------------------------------------------------------------
    torus = trimesh.creation.torus(
        major_radius=ring.radius,
        minor_radius=ring.tube_radius,
        major_sections=32,
        minor_sections=8,
    )

    # ------------------------------------------------------------------
    # Spawn USD mesh prim with a single translate + orient quaternion.
    # Using one OrientOp instead of separate RotateZ + RotateX avoids
    # non-orthonormal xform matrices that crash VSCode/Isaac Sim.
    # ------------------------------------------------------------------
    stage = omni.usd.get_context().get_stage()
    mesh = UsdGeom.Mesh.Define(stage, base_prim_path)

    verts = torus.vertices
    faces = torus.faces
    mesh.GetPointsAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in verts])
    )
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(faces)))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(faces.flatten().tolist()))

    # World position = env_origin + ring.pos
    wx = env_origin[0] + ring.pos[0]
    wy = env_origin[1] + ring.pos[1]
    wz = env_origin[2] + ring.pos[2]
    mesh.AddTranslateOp().Set(Gf.Vec3d(wx, wy, wz))

    # Single quaternion orientation (always orthonormal)
    w, x, y, z = ring_yaw_tilt_to_quat(ring.yaw_deg, ring.tilt_deg)
    mesh.AddOrientOp().Set(Gf.Quatf(w, x, y, z))

    # Display color (amber glow)
    r, g, b = ring.color
    mesh.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(r, g, b)]))

    prim = mesh.GetPrim()

    # ------------------------------------------------------------------
    # Physics: kinematic rigid body + mesh collision
    # ------------------------------------------------------------------
    rb_api = UsdPhysics.RigidBodyAPI.Apply(prim)
    rb_api.GetKinematicEnabledAttr().Set(True)

    UsdPhysics.CollisionAPI.Apply(prim)
    mesh_col = UsdPhysics.MeshCollisionAPI.Apply(prim)
    # convexDecomposition gives a good concave approx for the torus hole
    mesh_col.GetApproximationAttr().Set("convexDecomposition")
