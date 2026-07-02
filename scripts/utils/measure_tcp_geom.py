"""Measure the REAL TCP offset for the assembled gripper from its finger
GEOMETRY (world bounding box), not body origins. body_pos_w is useless for
Robotiq grippers because every link's origin sits at the coupling; the fingers
live in the mesh geometry. This computes the lowest point of the gripper meshes
(= fingertips) and reports the wrist->fingertip distance to use as OffsetCfg z.

    python scripts/utils/measure_tcp_geom.py --task Isaac-Lift-Cube-UR3e-2F140-Play-v0
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-UR3e-2F140-Play-v0")
parser.add_argument("--pad", type=float, default=0.012,
                    help="Subtract this from the tip to land the TCP between the pads, not at the very tip.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import gymnasium as gym

import omni.usd
from pxr import Usd, UsdGeom

import isaaclab_tasks  # noqa: F401
import isaac_robots.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(args_cli.task, device="cpu", num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    env.reset()
    zero = torch.zeros((base.num_envs, base.action_manager.total_action_dim), device=base.device)
    for _ in range(5):
        env.step(zero)

    robot = base.scene["robot"]
    names = robot.data.body_names
    wi = names.index("wrist_3_link")
    wrist = robot.data.body_pos_w[0, wi].cpu().numpy()

    stage = omni.usd.get_context().get_stage()
    # find the gripper root prim (robotiq_base_link) for env 0
    grip_root = None
    for prim in stage.Traverse():
        p = prim.GetPath().pathString
        if "env_0" in p and prim.GetName() == "robotiq_base_link":
            grip_root = prim
            break
    if grip_root is None:
        raise RuntimeError("Could not find robotiq_base_link under env_0.")

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )
    bound = cache.ComputeWorldBound(grip_root)
    rng = bound.ComputeAlignedRange()
    gmin = np.array(rng.GetMin())
    gmax = np.array(rng.GetMax())

    tip_z = gmin[2]                 # lowest gripper point = fingertips (pointing down)
    offset_to_tip = wrist[2] - tip_z
    offset_tcp = offset_to_tip - args_cli.pad

    print("\n================ TCP FROM GEOMETRY ================")
    print(f"  wrist_3_link world z      : {wrist[2]:.4f}")
    print(f"  gripper mesh z-range      : [{gmin[2]:.4f}, {gmax[2]:.4f}]  (extent {gmax[2]-gmin[2]:.4f} m)")
    print(f"  wrist -> fingertip (tip)  : {offset_to_tip:.4f} m")
    print(f"  suggested OffsetCfg z     : {offset_tcp:.4f} m   (tip - pad {args_cli.pad})")
    print("\n  -> set offset=OffsetCfg(pos=(0.0, 0.0, {:.3f})) in joint_pos_env_cfg.py".format(offset_tcp))

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
