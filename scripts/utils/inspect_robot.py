"""Print the real rigid-body prim paths. Run: ./isaaclab.sh -p find_prim_paths.py"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
import omni.usd
from pxr import UsdPhysics
from isaac_robots.tasks.direct._shared.assets import UR3E_2F85_USD


sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device="cpu"))
Articulation(ArticulationCfg(
    prim_path="/World/Robot",
    spawn=sim_utils.UsdFileCfg(usd_path=UR3E_2F85_USD),
    actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)},
))
sim.reset()

stage = omni.usd.get_context().get_stage()
print("\n=== articulation-root / body prim paths under /World/Robot ===")
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if not path.startswith("/World/Robot"):
        continue
    marks = []
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI): marks.append("ARTICULATION_ROOT")
    if prim.HasAPI(UsdPhysics.RigidBodyAPI): marks.append("body")
    if marks:
        print(f"  {path}   <- {' '.join(marks)}")
app.close()