"""Load ONLY the standalone NVIDIA Robotiq 2F-140 (no arm) and print every
rigid-body world position, so we can tell whether the gripper is collapsed in
the SOURCE asset or only after assembling it onto the UR3e.

Run with the Isaac Sim / Isaac Lab python env:
    python scripts/utils/inspect_2f140_standalone.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--usd",
    type=str,
    default=None,
    help="Override path/URL to the standalone 2F-140 USD. Defaults to the Nucleus Robotiq_2F_140_edit.usd.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


def main():
    usd = args_cli.usd or f"{ISAAC_NUCLEUS_DIR}/Robots/Robotiq/2F-140/Robotiq_2F_140_physics_edit.usd"
    print(f"\n[inspect-2f140] loading standalone gripper: {usd}")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device="cpu"))
    grip = Articulation(
        ArticulationCfg(
            prim_path="/World/Gripper",
            spawn=sim_utils.UsdFileCfg(usd_path=usd),
            actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)},
        )
    )
    sim.reset()
    for _ in range(3):
        sim.step()
        grip.update(sim.get_physics_dt())

    names = grip.data.body_names
    pos = grip.data.body_pos_w[0].cpu().numpy()
    print("\n================ STANDALONE 2F-140 BODY POSITIONS ================")
    for n, p in zip(names, pos):
        print(f"  {n:40s} {np.round(p, 4)}")

    zs = pos[:, 2]
    print(f"\n  z-extent of gripper bodies: {zs.max() - zs.min():.4f} m  (a real 2F-140 is ~0.15-0.20 m tall)")
    print(f"  joint_names: {grip.data.joint_names}")

    sim.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
