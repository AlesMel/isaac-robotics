# Crazyflie Isaac Lab

Modular Isaac Lab environment scaffold for the Bitcraze Crazyflie with config-driven optional sensors.

## Install

Install this repository into the same Python environment used by Isaac Lab / Isaac Sim.

Editable install of the local package:

```bash
pip install -e .
```

Optional training dependencies:

```bash
pip install -e .[train]
```

## Two-Repository Workflow

This project is designed to live in its own repository while Isaac Lab lives in a separate checkout.

Example layout:

```text
/home/user/IsaacLab
/home/user/isaac-robotics
```

Recommended flow:

1. Activate or use the Python environment that Isaac Lab uses.
2. Install this repository into that same environment.
3. Launch `train.py` through Isaac Lab's launcher so Omni / Isaac Sim modules are available.

### Option A: Editable install + launcher

From this repository:

```bash
cd /path/to/isaac-robotics
python -m pip install -e .
python -m pip install -e .[train]
```

Then launch through the Isaac Lab checkout:

```bash
cd /path/to/IsaacLab
./isaaclab.sh -p /path/to/isaac-robotics/train.py --headless
```

### Option B: Use the included launcher wrapper

This repository ships with `run.sh`, which forwards execution to an external Isaac Lab checkout.

```bash
cd /path/to/isaac-robotics
./run.sh /path/to/IsaacLab --headless
```

Or set the path once:

```bash
export ISAACLAB_DIR=/path/to/IsaacLab
./run.sh --headless --algorithm ppo --enable-lidar
```

`run.sh` automatically:

- Locates `isaaclab.sh` or `isaac-sim.sh`
- Adds this repo's `src` directory to `PYTHONPATH`
- Launches `train.py` through Isaac Lab so `omni` imports work

## Training Examples

Headless PPO with state only:

```bash
./run.sh /path/to/IsaacLab --headless --algorithm ppo
```

PPO with LiDAR enabled:

```bash
./run.sh /path/to/IsaacLab --headless --algorithm ppo --enable-lidar
```

SAC with camera enabled:

```bash
./run.sh /path/to/IsaacLab --headless --algorithm sac --enable-camera
```

TD3 with both sensors enabled:

```bash
./run.sh /path/to/IsaacLab --headless --algorithm td3 --enable-lidar --enable-camera
```

## Runtime Notes

- Do not run `python train.py` directly unless your Python process was started by Isaac Lab / Isaac Sim.
- The `train.py` script bootstraps Isaac Lab with `AppLauncher`, but the launcher itself must still come from the Isaac Lab environment.
- If your Nucleus paths differ from the defaults, set `ISAAC_NUCLEUS_DIR` or `CRAZYFLIE_USD_PATH` before launch.
- Observation size is computed dynamically from `SensorSelectionCfg`, so skrl model input dimensions change automatically with enabled sensors.

## Notes

- Isaac Lab / Isaac Sim are expected to be installed separately in the target environment.
- Sensor activation is controlled through `SensorSelectionCfg` in `src/crazyflie_lab/config/sensors.py`.
- Training entrypoint: `train.py`
- Convenience launcher: `run.sh`
