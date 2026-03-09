# Crazyflie Isaac Lab

Modular Isaac Lab environment scaffold for the Bitcraze Crazyflie with config-driven optional sensors.

## Install

Editable install of the local package:

```bash
pip install -e .
```

Optional training dependencies:

```bash
pip install -e .[train]
```

## Notes

- Isaac Lab / Isaac Sim are expected to be installed separately in the target environment.
- Sensor activation is controlled through `SensorSelectionCfg` in `src/crazyflie_lab/config/sensors.py`.
- Training entrypoint: `train.py`
