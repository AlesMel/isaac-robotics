"""Render a headless preview PNG of the combined UR3e + Hand-E USD.

Loads the assembled robot, aims a camera at the tool0 frame (where the gripper
is mounted), and writes a still image so the gripper mount orientation can be
inspected without a GUI. Run with the Isaac Sim / Isaac Lab Python env.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_usd() -> Path:
    return _repo_root() / "source" / "isaac_robots" / "data" / "ur3e" / "ur3e_robotiq_hande.usd"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=Path, default=_default_usd(), help="Combined robot USD to preview.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_repo_root() / "_hande_preview",
        help="Directory for the rendered PNG(s).",
    )
    return parser.parse_args()


def _log(message: str) -> None:
    print(f"[hande-preview] {message}", flush=True)


def main() -> None:
    args = _parse_args()
    usd_path = args.usd.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})

    exit_code = 0
    try:
        import omni.replicator.core as rep
        import omni.usd
        from isaacsim.core.utils.stage import add_reference_to_stage, create_new_stage, update_stage
        from pxr import Gf, Usd, UsdGeom, UsdLux

        if not usd_path.is_file():
            raise FileNotFoundError(f"Combined USD not found: {usd_path}")

        create_new_stage()
        stage = omni.usd.get_context().get_stage()
        _log(f"Loading {usd_path}")
        add_reference_to_stage(str(usd_path), "/World/Robot")

        dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome.CreateIntensityAttr(2500.0)
        key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
        key.CreateIntensityAttr(3500.0)

        # Wait for the referenced assets (ur3e.usd + Hand-E from Nucleus) to load.
        for _ in range(240):
            update_stage()
            if omni.usd.get_context().get_stage_loading_status()[2] == 0:
                break

        # --- Decisive numeric check: where is the gripper relative to tool0's axes? ---
        tool0 = stage.GetPrimAtPath("/World/Robot/tool0")
        gripper = stage.GetPrimAtPath("/World/Robot/Robotiq_Hand_E")
        target = (0.0, 0.0, 0.3)
        if tool0.IsValid() and gripper.IsValid():
            t0_mat = omni.usd.get_world_transform_matrix(tool0)
            p_t0 = t0_mat.ExtractTranslation()
            rot = t0_mat.ExtractRotationMatrix()  # rows are tool0's world-space axes
            x_axis, y_axis, z_axis = rot.GetRow(0), rot.GetRow(1), rot.GetRow(2)

            bbox_cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
            )
            g_range = bbox_cache.ComputeWorldBound(gripper).ComputeAlignedRange()
            g_center = g_range.GetMidpoint()
            v = g_center - p_t0
            v_local = Gf.Vec3d(v * x_axis, v * y_axis, v * z_axis)  # offset in tool0 frame
            _log(f"tool0 world pos = {tuple(round(c, 4) for c in p_t0)}")
            _log(f"gripper center world = {tuple(round(c, 4) for c in g_center)}")
            _log(f"gripper offset in tool0 frame (x,y,z) = {tuple(round(c, 4) for c in v_local)}")

            # Extent of the gripper along tool0 +Z (flange normal): the far end
            # (~fingertips) is a good value for jaw_center_offset_local.
            mn, mx = g_range.GetMin(), g_range.GetMax()
            corners = [
                Gf.Vec3d(cx, cy, cz)
                for cx in (mn[0], mx[0])
                for cy in (mn[1], mx[1])
                for cz in (mn[2], mx[2])
            ]
            z_proj = [(c - p_t0) * z_axis for c in corners]
            _log(
                f"gripper extent along tool0 +Z: min={min(z_proj):.4f} max={max(z_proj):.4f} "
                f"(jaw_center_offset_local ~ {0.5 * (min(z_proj) + max(z_proj)):.3f}..{max(z_proj):.3f})"
            )

            # Per-mesh positions in the tool0 frame -> which way the fingers point
            # (Z) and how they're spun (jaw-opening axis = X vs Y spread). The
            # Hand-E meshes are instanced, so traverse instance proxies.
            parts = []
            for prim in Usd.PrimRange(gripper, Usd.TraverseInstanceProxies()):
                if prim.IsA(UsdGeom.Gprim):
                    c = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange().GetMidpoint()
                    d = c - p_t0
                    parts.append((d * z_axis, d * x_axis, d * y_axis, prim.GetName()))
            parts.sort()
            _log(f"gripper leaf meshes found: {len(parts)}")
            for pz, px, py, n in parts[:30]:
                _log(f"  part tool0 (x,y,z)=({px:+.3f},{py:+.3f},{pz:+.3f}) '{n}'")
            fingers = [p for p in parts if any(k in p[3].lower() for k in ("finger", "pad", "doigt"))]
            if fingers:
                fz = sum(p[0] for p in fingers) / len(fingers)
                _log(
                    f"FINGERS VERDICT: mean finger tool0 +Z = {fz:+.4f} -> "
                    + ("point OUT (+Z, good)" if fz > 0 else "point toward the arm (-Z, wrong)")
                )
                xr = max(p[1] for p in fingers) - min(p[1] for p in fingers)
                yr = max(p[2] for p in fingers) - min(p[2] for p in fingers)
                _log(
                    f"jaw-opening spread: tool0 X range={xr:.4f}, Y range={yr:.4f} "
                    f"(larger axis = direction the jaws open)"
                )

            dom = max(range(3), key=lambda i: abs(v_local[i]))
            axis_name = ["X", "Y", "Z"][dom]
            sign = "+" if v_local[dom] >= 0 else "-"
            verdict = (
                "OK: gripper points along tool0 +Z (out of the flange)."
                if dom == 2 and v_local[2] > 0
                else f"TILTED: gripper points along tool0 {sign}{axis_name}, not +Z."
            )
            _log(f"VERDICT -> dominant axis {sign}{axis_name}. {verdict}")
            target = (float(g_center[0]), float(g_center[1]), float(g_center[2]))
        else:
            _log("tool0 or gripper prim not found; framing the robot origin instead.")
        _log(f"Aiming camera at gripper center = {target}")

        # Two 3/4 views from opposite sides so the mount orientation is unambiguous.
        views = {
            "view_a": (target[0] + 0.55, target[1] + 0.55, target[2] + 0.30),
            "view_b": (target[0] + 0.55, target[1] - 0.55, target[2] + 0.30),
        }
        for name, cam_pos in views.items():
            camera = rep.create.camera(position=cam_pos, look_at=target)
            render_product = rep.create.render_product(camera, (1280, 960))
            writer = rep.WriterRegistry.get("BasicWriter")
            writer.initialize(output_dir=str(out_dir / name), rgb=True)
            writer.attach([render_product])
            rep.orchestrator.step()
            rep.orchestrator.wait_until_complete()
            writer.detach()
            render_product.destroy()
            _log(f"Wrote preview '{name}' to {out_dir / name}")
    except BaseException as exc:
        exit_code = 1
        print(f"[hande-preview] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
    finally:
        simulation_app.close()
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
