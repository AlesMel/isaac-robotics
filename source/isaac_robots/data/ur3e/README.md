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

## Building UR3e + Robotiq 2F-85

The manager-based Robotiq stack task expects a single combined articulation at:

```text
source/isaac_robots/data/ur3e/ur3e_robotiq_2f85.usd
```

Build it with Isaac Sim / Isaac Lab Python:

```bash
python scripts/ur3e/build_ur3e_robotiq_2f85_usd.py
```

The script references the local `ur3e.usd`, references the official Robotiq
2F-85 asset from Isaac Sim assets, uses Isaac Sim's robot assembler to attach
the gripper to `tool0`, and saves the combined USD.

## Building UR3e + Robotiq Hand-E

The UR3e Hand-E lift-cube task (`Isaac-Robots-UR3e-Lift-Cube-HandE-Direct-v0`)
expects a single combined articulation at:

```text
source/isaac_robots/data/ur3e/ur3e_robotiq_hande.usd
```

Build it with Isaac Sim / Isaac Lab Python:

```bash
python scripts/ur3e/build_ur3e_robotiq_hande_usd.py
```

The script mirrors the 2F-85 builder: it references the local `ur3e.usd`,
references the official Robotiq **Hand-E** asset from the Isaac Sim library
(`Isaac/Robots/Robotiq/Hand-E/Robotiq_Hand_E_edit.usd`), assembles it onto
`tool0`, and saves the combined USD.

**Mount orientation:** the Hand-E `base_link` frame does not match the flange
convention, so the assembler mounts it tilted ~90deg sideways and with the
coupling ~7.6 cm behind `tool0` (buried in the wrist). The build script's
`--mount-rpy-deg` / `--mount-offset-xyz` (both in the tool0 frame) correct this,
but **default to `0 0 0` / `0 0 0` (no correction)** — so a plain build produces
the tilted/buried mount. The **known-good correction** for this asset is:

```bash
python scripts/ur3e/build_ur3e_robotiq_hande_usd.py --mount-rpy-deg 90 0 90 --mount-offset-xyz 0 0 0.076
```

Verified on the assembled USD: this points the fingers out along tool0 +Z
(finger pads at +Z `0.092..0.105`, mean `0.096`), seats the coupling at the
flange (screws at `z~0`), and opens the jaws along tool0 X. The 3rd value (yaw)
spins the jaws about the tool axis: `90 0 0` opens them along Y, `90 0 90` along
X. The committed `ur3e_robotiq_hande.usd` was built with `90 0 90`.

You can sanity-check the mount without a GUI with
`scripts/ur3e/render_hande_preview.py`, which prints the gripper's offset/extent
in the tool0 frame and writes preview PNGs to `_hande_preview/`.

Path overrides (env vars):

* `UR3E_ROBOTIQ_HANDE_BODY_USD_PATH` — point the *builder* at a different
  standalone Hand-E source USD (defaults to the Isaac library asset).
* `UR3E_ROBOTIQ_HANDE_USD_PATH` — point the *runtime* config
  ([_shared/assets.py](../../isaac_robots/tasks/direct/_shared/assets.py),
  `UR3E_ROBOTIQ_HANDE_CFG`) at the combined USD if you keep it elsewhere.

### Grasp model: real friction physics

The Hand-E performs a **real friction grasp** (like the manager-based 2F-85
stack task), not a kinematic attach. The fingers are driven to a commanded
width by the slider actuators and the cube is held purely by simulated PhysX
contact + friction. The gripper's `update_attachment` is a no-op, so the cube
only rises if it is physically gripped.

### Tuning caveats (confirm against the assembled USD)

The build script logs the assembled physics joint names. Confirm them, then
adjust if needed:

* **Slider joint names.** The config assumes the two finger sliders are named
  `Slider_1` / `Slider_2` (matched by the regex `Slider_.*`). If the asset uses
  other names, update both `UR3E_ROBOTIQ_HANDE_CFG` (init pose + `gripper_slide`
  actuator) in [_shared/assets.py](../../isaac_robots/tasks/direct/_shared/assets.py)
  and `gripper_joint_names_expr` in
  [robotiq_hande.py](../../isaac_robots/tasks/direct/_shared/grippers/robotiq_hande.py).
* **Slider travel.** The Hand-E sliders travel **0 → 0.025 m** (verified from the
  USD joint limits; `lowerLimit=0`, `upperLimit=0.025`). `finger_open_pos` /
  `finger_closed_pos` in `RobotiqHandEGripperCfg` must stay in that range —
  defaults are `0.0` (open) and `0.025` (closed). Driving outside it (e.g. the
  old `-0.025`) over-extends the fingers past the housing. Swap the two values
  if open/close feels inverted in the task.
* **Grip force.** The `gripper_slide` actuator `effort_limit_sim` (default
  `20.0` N) is the steady clamping force. Raise it if the cube slips out, lower
  it if the cube gets flung.
* **Finger friction.** Stable grasping needs friction between the finger pads
  and the cube. It comes from the USD collider materials multiplied by the
  scene physics material (`static_friction = dynamic_friction = 1.0`). If the
  cube slips despite enough grip force, add a high-friction physics material to
  the finger pads / cube.
* **Jaw-center offset.** `jaw_center_offset_local` (default `(0, 0, 0.096)`,
  measured on the corrected USD) is the tool0→jaw-center distance used for the
  grasp-readiness reward metric. Finger pads sit at tool0 +Z `0.092..0.105`. This
  value assumes the known-good mount correction above; re-measure if you rebuild
  without it.
* **Cube size vs jaw opening.** The Hand-E has only a ~50 mm stroke. The base
  task's cube (~80 mm) is likely too wide to grasp -- shrink the cube for the
  Hand-E variant (`UR3eLiftCubeHandEEnvCfg`) once you have measured both in
  Isaac Sim. See the class docstring in
  [ur3e_lift_cube_env_cfg.py](../../isaac_robots/tasks/direct/isaac_ur3e_lift_cube/ur3e_lift_cube_env_cfg.py).

Grasp detection (for the reward `is_holding` term) is a contact-sensor-free
heuristic: the cube is "held" when it is inside the jaw region, aligned, and the
fingers are clamped. The honest lift signal remains the cube's physical height.
