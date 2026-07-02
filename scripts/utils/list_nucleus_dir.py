"""List the contents of a Nucleus/asset-server folder, to discover which Robotiq
2F-140 USD variants actually exist. Run with the Isaac Sim / Isaac Lab env:

    python scripts/utils/list_nucleus_dir.py
    python scripts/utils/list_nucleus_dir.py --sub Robots/Robotiq/2F-140
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--sub", type=str, default="Robots/Robotiq/2F-140",
                    help="Path under the Isaac nucleus root to list.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.client
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


def list_dir(url):
    result, entries = omni.client.list(url)
    print(f"\n[list] {url}  -> {result}")
    for e in entries:
        flags = e.flags
        is_dir = bool(flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN)
        kind = "DIR " if is_dir else "file"
        print(f"  {kind}  {e.relative_path}")
    return entries


def main():
    base = f"{ISAAC_NUCLEUS_DIR}/{args_cli.sub.strip('/')}"
    # Also try the parent Robotiq folder in case 2F-140 lives elsewhere.
    list_dir(base)
    if "2F-140" in args_cli.sub:
        list_dir(f"{ISAAC_NUCLEUS_DIR}/Robots/Robotiq")


if __name__ == "__main__":
    main()
    simulation_app.close()
