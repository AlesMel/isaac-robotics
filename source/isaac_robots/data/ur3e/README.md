# UR3e USD asset

Drop the UR3e USD here as `ur3e.usd`. The config in
[_shared/assets.py](../../isaac_robots/tasks/direct/_shared/assets.py) reads
this path by default; override with `UR3E_USD_PATH=/abs/path/to/ur3e.usd` if
you keep the asset elsewhere.

## Generating the USD from the official URDF

Pick one of the two paths below. The Python-only path works on Windows
without ROS.

### Option A: Python-only (works on Windows, macOS, Linux)

The `xacro` tool is published on PyPI -- no ROS install required.

```bash
# 1. Get the upstream description package.
git clone --depth 1 https://github.com/UniversalRobots/Universal_Robots_ROS2_Description ur_description

# 2. Install the standalone xacro processor into your isaaclab env.
pip install xacro

# 3. Expand the xacro into a flat URDF.
#    Note: `python -m xacro` does NOT work -- the package has no
#    __main__.py. Use the installed console script or the cli submodule.
xacro ur_description/urdf/ur.urdf.xacro ^
    name:=ur3e ur_type:=ur3e ^
    -o ur3e.urdf
# ...or, if xacro is not on PATH:
# python -m xacro.cli ur_description/urdf/ur.urdf.xacro name:=ur3e ur_type:=ur3e -o ur3e.urdf

# 4. Convert URDF -> USD with Isaac Lab's converter (run from repo root).
python -m isaaclab.scripts.convert_urdf ^
    ur3e.urdf ^
    source/isaac_robots/data/ur3e/ur3e.usd ^
    --fix-base --make-instanceable
```

(On bash/PowerShell replace the trailing `^` with `\` or remove the line
breaks.)

### Option B: Pre-flattened URDF (skip xacro entirely)

If you cannot install `xacro` for any reason, fetch a URDF that has already
been macro-expanded. The `ros-industrial/universal_robot` repo ships ready-
to-use URDFs:

```bash
# Download a single resolved URDF directly:
curl -L -o ur3e.urdf https://raw.githubusercontent.com/ros-industrial/universal_robot/melodic-devel/ur_e_description/urdf/ur3e_robot.urdf

# Then run step 4 above (convert_urdf).
```

Inspect the file before converting -- it must reference local mesh paths
(usually `package://ur_description/meshes/ur3e/...`). You may need to
either provide the meshes alongside or rewrite those paths to absolute
file paths so `convert_urdf` can resolve them. The `--mesh-dir` flag on
the converter helps:

```bash
python -m isaaclab.scripts.convert_urdf \
    ur3e.urdf \
    source/isaac_robots/data/ur3e/ur3e.usd \
    --fix-base --make-instanceable \
    --mesh-dir ur_description/meshes/ur3e/visual
```

**Why `--fix-base` and *not* `--merge-joints`:** the URDF carries a
`base_link → world` fixed joint that anchors the arm. `--merge-joints`
collapses it (and the `tool0` TCP frame) into the parent link, leaving the
articulation free-floating. `--fix-base` keeps the world-anchor and
preserves `tool0`, which is the same frame `ur_rtde.getActualTCPPose()`
reports on hardware -- observations transfer cleanly.

### Verifying the result

After conversion, open `ur3e.usd` in Isaac Sim and check:

* The articulation has 6 actuated joints with the names listed in
  [_shared/assets.py](../../isaac_robots/tasks/direct/_shared/assets.py).
* A body named `tool0` exists -- that's what `cfg.ee_body_name` points to.
  If it's named `flange` or `wrist_3_link` instead, change `ee_body_name`
  on the env config to match.
* The base link is at the origin and z-up.
