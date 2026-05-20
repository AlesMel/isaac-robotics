"""Build a combined UR3e + Robotiq Hand-E USD.

Run this with the Isaac Sim / Isaac Lab Python environment, not plain system
Python. The script uses Isaac Sim's robot assembler to make the Hand-E a single
attachment inside the UR3e articulation, then writes the USD expected by the
UR3e Hand-E lift-cube task:

    source/isaac_robots/data/ur3e/ur3e_robotiq_hande.usd

This mirrors ``build_ur3e_robotiq_2f85_usd.py``; the only differences are the
default gripper source asset, the child prim name, and the output path.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_ur3e_usd() -> Path:
    return _repo_root() / "source" / "isaac_robots" / "data" / "ur3e" / "ur3e.usd"


def _default_output_usd() -> Path:
    return _repo_root() / "source" / "isaac_robots" / "data" / "ur3e" / "ur3e_robotiq_hande.usd"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ur3e-usd", type=Path, default=_default_ur3e_usd(), help="Path to the source UR3e USD.")
    parser.add_argument(
        "--gripper-usd",
        default=os.getenv("UR3E_ROBOTIQ_HANDE_BODY_USD_PATH"),
        help=(
            "Path or Nucleus URL to the standalone Robotiq Hand-E USD. Defaults to the "
            "UR3E_ROBOTIQ_HANDE_BODY_USD_PATH env var, then to "
            "<Isaac assets root>/Isaac/Robots/Robotiq/Hand-E/Robotiq_Hand_E_edit.usd."
        ),
    )
    parser.add_argument("--output-usd", type=Path, default=_default_output_usd(), help="Output combined USD path.")
    parser.add_argument("--robot-path", default="/Robot", help="Root prim path for the assembled robot.")
    parser.add_argument("--base-mount", default="tool0", help="UR3e body/frame used as the Hand-E mount.")
    parser.add_argument(
        "--gripper-path",
        default="Robotiq_Hand_E",
        help="Child prim path under --robot-path where the Hand-E reference is added.",
    )
    parser.add_argument(
        "--gripper-mount",
        default=None,
        help=(
            "Hand-E mount frame/body. If omitted, the script searches for a likely base link "
            "inside the Hand-E asset."
        ),
    )
    parser.add_argument(
        "--mount-rpy-deg",
        type=float,
        nargs=3,
        default=(90.0, 0.0, 0.0),
        metavar=("ROLL", "PITCH", "YAW"),
        help=(
            "Corrective rotation (degrees, about the tool0 X/Y/Z axes) applied to the gripper at "
            "the mount. Default (90, 0, 0) is the confirmed correction for the Hand-E: its base_link "
            "frame mounts ~90deg sideways and flipped, and +90 about X points the fingers out along "
            "tool0 +Z. The 3rd value (yaw) spins about the tool long axis if you want a different "
            "jaw-opening direction."
        ),
    )
    parser.add_argument(
        "--mount-offset-xyz",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.076),
        metavar=("X", "Y", "Z"),
        help=(
            "Corrective translation (m, along the tool0 X/Y/Z axes) applied to the gripper at the "
            "mount. Default +0.076 m along Z seats the Hand-E coupling on the flange; without it the "
            "gripper's base_link frame leaves the coupling ~7.6 cm behind tool0 (buried in the wrist)."
        ),
    )
    parser.add_argument("--headless", action="store_true", default=True, help="Run Isaac Sim headless.")
    return parser.parse_args()


def _log(message: str) -> None:
    print(f"[ur3e-hande-builder] {message}", flush=True)


def _as_abs_prim_path(root_path: str, child_path: str) -> str:
    if child_path.startswith("/"):
        return child_path
    return f"{root_path.rstrip('/')}/{child_path}"


def _as_relative_reference(asset_path: Path, anchor_dir: Path) -> str:
    try:
        return asset_path.relative_to(anchor_dir).as_posix()
    except ValueError:
        return Path(os.path.relpath(asset_path, anchor_dir)).as_posix()


def _wait_for_stage_load(update_stage, omni_usd_context, max_frames: int = 240) -> None:
    for _ in range(max_frames):
        update_stage()
        if omni_usd_context.get_stage_loading_status()[2] == 0:
            return
    raise TimeoutError("Timed out waiting for referenced USD assets to load.")


def _find_gripper_mount(stage, attach_path: str, requested_mount: str | None):
    from pxr import Usd, UsdPhysics

    if requested_mount:
        mount_path = _as_abs_prim_path(attach_path, requested_mount)
        prim = stage.GetPrimAtPath(mount_path)
        if prim.IsValid():
            return mount_path
        raise RuntimeError(f"Requested Hand-E mount prim does not exist: {mount_path}")

    attach_prim = stage.GetPrimAtPath(attach_path)
    if not attach_prim.IsValid():
        raise RuntimeError(f"Hand-E attachment prim does not exist: {attach_path}")

    scored_candidates: list[tuple[int, str]] = []
    for prim in Usd.PrimRange(attach_prim):
        name = prim.GetName().lower()
        path = prim.GetPath().pathString
        score = 0
        if "base" in name:
            score += 10
        if "link" in name:
            score += 5
        if "mount" in name:
            score += 3
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            score += 20
        if score > 0:
            scored_candidates.append((score, path))

    if scored_candidates:
        scored_candidates.sort(reverse=True)
        return scored_candidates[0][1]

    return attach_path


def _log_joint_names(stage, robot_path) -> None:
    """Log the physics joints under the assembled robot so slider names can be confirmed."""
    from pxr import Usd

    names: list[str] = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_path)):
        if "Joint" in prim.GetTypeName():
            names.append(prim.GetName())
    _log(f"Assembled physics joints ({len(names)}): {names}")
    _log(
        "Confirm the Hand-E finger slider names above match 'Slider_.*' in "
        "UR3E_ROBOTIQ_HANDE_CFG and RobotiqHandEGripperCfg; update both if they differ."
    )


def main() -> None:
    args = _parse_args()
    _log("Starting Isaac Sim.")

    # Import Isaac Sim only after argument parsing so --help works in plain Python.
    try:
        from isaacsim import SimulationApp
    except ImportError:
        from omni.isaac.kit import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless})

    exit_code = 0
    try:
        import omni.usd
        from isaacsim.core.utils.extensions import enable_extension

        _log("Enabling robot assembler and Nucleus storage extensions.")
        enable_extension("isaacsim.robot_setup.assembler")
        enable_extension("isaacsim.storage.native")
        from isaacsim.core.utils.stage import add_reference_to_stage, create_new_stage, update_stage
        from isaacsim.robot_setup.assembler import RobotAssembler
        from isaacsim.storage.native import get_assets_root_path
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

        ur3e_usd = args.ur3e_usd.resolve()
        output_usd = args.output_usd.resolve()
        _log(f"UR3e source USD: {ur3e_usd}")
        _log(f"Output USD: {output_usd}")
        if not ur3e_usd.is_file():
            raise FileNotFoundError(f"UR3e USD does not exist: {ur3e_usd}")

        hande_usd = args.gripper_usd
        if hande_usd is None:
            assets_root = get_assets_root_path()
            if assets_root is None:
                raise RuntimeError("Could not resolve Isaac Sim assets root for the Hand-E USD.")
            hande_usd = f"{assets_root}/Isaac/Robots/Robotiq/Hand-E/Robotiq_Hand_E_edit.usd"
        _log(f"Hand-E source USD: {hande_usd}")

        if not create_new_stage():
            raise RuntimeError("Isaac Sim failed to create a new USD stage.")
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Isaac Sim did not return a current USD stage.")
        robot_path = args.robot_path.rstrip("/")
        gripper_path = _as_abs_prim_path(robot_path, args.gripper_path)
        base_mount_path = _as_abs_prim_path(robot_path, args.base_mount)

        _log(f"Referencing UR3e at {robot_path}.")
        add_reference_to_stage(str(ur3e_usd), robot_path)
        robot_prim = stage.GetPrimAtPath(robot_path)
        if not robot_prim.IsValid():
            raise RuntimeError(f"UR3e prim was not created at {robot_path}.")
        stage.SetDefaultPrim(robot_prim)
        _log(f"Referencing Hand-E at {gripper_path}.")
        add_reference_to_stage(str(hande_usd), gripper_path)

        _log("Waiting for referenced assets to load.")
        _wait_for_stage_load(update_stage, omni.usd.get_context())

        if not stage.GetPrimAtPath(base_mount_path).IsValid():
            raise RuntimeError(f"UR3e mount prim does not exist: {base_mount_path}")

        gripper_mount_path = _find_gripper_mount(stage, gripper_path, args.gripper_mount)
        _log(f"UR3e mount: {base_mount_path}")
        _log(f"Hand-E mount: {gripper_mount_path}")

        # Align the selected Hand-E mount frame to the UR3e mount frame.
        gripper_prim = stage.GetPrimAtPath(gripper_path)
        gripper_mount_prim = stage.GetPrimAtPath(gripper_mount_path)
        base_mount_prim = stage.GetPrimAtPath(base_mount_path)

        mount_pose = omni.usd.get_world_transform_matrix(base_mount_prim)
        attachment_pose = omni.usd.get_local_transform_matrix(gripper_mount_prim)

        # Optional corrective transform, expressed in the tool0 frame: rotate the
        # gripper by --mount-rpy-deg (about tool0 X/Y/Z) then translate it by
        # --mount-offset-xyz. Default is identity, which makes the gripper mount
        # frame coincide exactly with tool0. Use this to fix a gripper whose base
        # frame does not match the flange convention (mounts rotated/offset).
        roll, pitch, yaw = args.mount_rpy_deg
        correction = Gf.Matrix4d(1.0)
        correction.SetRotate(
            Gf.Rotation(Gf.Vec3d(1, 0, 0), roll)
            * Gf.Rotation(Gf.Vec3d(0, 1, 0), pitch)
            * Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw)
        )
        correction.SetTranslateOnly(Gf.Vec3d(*args.mount_offset_xyz))
        if any(args.mount_rpy_deg) or any(args.mount_offset_xyz):
            _log(
                f"Applying mount correction: rpy_deg={tuple(args.mount_rpy_deg)} "
                f"offset_xyz={tuple(args.mount_offset_xyz)}"
            )

        base_mount_pose = attachment_pose.GetInverse() * correction * mount_pose
        attachment_parent_pose = omni.usd.get_world_transform_matrix(gripper_prim.GetParent())
        attachment_pose_local = base_mount_pose * attachment_parent_pose.GetInverse()

        xform = UsdGeom.Xformable(gripper_prim)
        xform.ClearXformOpOrder()
        xform.AddTransformOp().Set(attachment_pose_local)

        assembler = RobotAssembler()
        _log("Assembling Hand-E into the UR3e articulation.")
        assembler.assemble_rigid_bodies(
            base_path=robot_path,
            attach_path=gripper_path,
            base_mount_frame=base_mount_path,
            attach_mount_frame=gripper_mount_path,
            mask_all_collisions=False,
        )

        # The assembler removes the attached asset's articulation root. Keep an
        # explicit defensive pass so this file never loads as two articulations.
        for prim in Usd.PrimRange(stage.GetPrimAtPath(gripper_path)):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
                prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)

        articulation_roots = [
            prim.GetPath().pathString
            for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_path))
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        ]
        if len(articulation_roots) != 1:
            raise RuntimeError(
                f"Expected exactly one articulation root under {robot_path}, found {articulation_roots}"
            )

        _log_joint_names(stage, robot_path)

        # Keep the saved asset portable and robot-only. The source UR3e lives
        # beside the generated file, while Hand-E remains a Nucleus reference.
        robot_spec = stage.GetRootLayer().GetPrimAtPath(robot_path)
        if robot_spec is None:
            raise RuntimeError(f"Could not find root layer spec for {robot_path}.")
        robot_spec.referenceList.prependedItems = [Sdf.Reference(_as_relative_reference(ur3e_usd, output_usd.parent))]
        for cleanup_path in ("/Render", "/OmniverseKit_Persp"):
            if stage.GetPrimAtPath(cleanup_path).IsValid():
                stage.RemovePrim(cleanup_path)

        output_usd.parent.mkdir(parents=True, exist_ok=True)
        _log("Exporting combined USD.")
        if not stage.GetRootLayer().Export(str(output_usd)):
            raise RuntimeError(f"USD export returned false for: {output_usd}")
        if not output_usd.is_file():
            raise RuntimeError(f"USD export completed but the file is missing: {output_usd}")

        _log(f"Wrote {output_usd}")
        _log(f"Articulation root: {articulation_roots[0]}")
    except BaseException as exc:
        exit_code = 1
        print(f"[ur3e-hande-builder] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
    finally:
        simulation_app.close()
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
