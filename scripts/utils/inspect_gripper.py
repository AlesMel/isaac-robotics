"""Dump body + joint names from the assembled USD, exactly as IsaacLab orders them.
Run:  ./isaaclab.sh -p inspect_gripper.py
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaac_robots.tasks.direct._shared.assets import UR3E_2F85_USD

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device="cpu"))
robot = Articulation(ArticulationCfg(
    prim_path="/World/Robot",
    spawn=sim_utils.UsdFileCfg(usd_path=UR3E_2F85_USD),
    actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)},
))
sim.reset()

print("\n=== BODIES ===")
for n in robot.body_names:
    print("  ", n)
print("\n=== JOINTS ===")
for n in robot.joint_names:
    print("  ", n)
app.close()