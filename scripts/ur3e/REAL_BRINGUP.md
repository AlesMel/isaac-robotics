# UR3e Real-Hardware Bring-Up

Use this path in order. Keep the workspace empty, keep one hand near the
e-stop, and start with the robot speed slider low. The scripts cannot verify
physical clearance.

## Package

```bash
python -m pip install -r requirements-real-ur3e.txt
```

This environment currently has `ur-rtde==1.6.3` installed.

## Read-Only

This lab robot has been seen at `147.175.108.138`. Set the robot IP if it
changes:

```bash
export ROBOT_IP=147.175.108.138
```

Read one state sample:

```bash
python scripts/ur3e/safe_bringup.py status
```

Watch for stable, stationary state:

```bash
python scripts/ur3e/safe_bringup.py watch --samples 20 --interval 0.25
```

## Control Without Intended Motion

First print the plan only:

```bash
python scripts/ur3e/safe_bringup.py hold --seconds 2
```

Then execute a current-position hold:

```bash
python scripts/ur3e/safe_bringup.py hold --seconds 2 --execute --accept-risk
```

## Tiny Motion

Plan a tiny wrist-3 move only:

```bash
python scripts/ur3e/safe_bringup.py tiny-movej --joint-index 5 --delta-rad 0.003
```

Execute only after the printed target looks sane:

```bash
python scripts/ur3e/safe_bringup.py tiny-movej --joint-index 5 --delta-rad 0.003 --execute --accept-risk
```

After that passes, test the streaming control path:

```bash
python scripts/ur3e/safe_bringup.py tiny-servoj --joint-index 5 --delta-rad -0.003 --execute --accept-risk
```

## Home Position

Do not send a direct move to the full sim home pose. First inspect the plan:

```bash
python scripts/ur3e/safe_bringup.py home-plan
```

Then move only one bounded chunk toward home:

```bash
python scripts/ur3e/safe_bringup.py home-step --max-step-rad 0.05 --execute --accept-risk
```

Repeat `home-plan` and `home-step` only while every printed `next_q` looks
sane for the physical workspace.

## Learned Policy, Guarded

The retrained sim-to-real policy should use the new position-only 21D
observation layout. `policy-preview`, `policy-debug`, and `policy-step`
auto-detect 21D vs legacy 25D from the checkpoint.

Preview one policy action with the target equal to the current TCP position.
This sends no motion command:

```bash
python scripts/ur3e/safe_bringup.py policy-preview
```

Preview a tiny target offset:

```bash
python scripts/ur3e/safe_bringup.py policy-preview --target-offset 0 0 0.005
```

Debug the observation and compare quaternion sign conventions without sending
motion:

```bash
python scripts/ur3e/safe_bringup.py policy-debug --full
```

The debug output compares raw, negated, positive-w, positive-max, and
negative-max quaternion sign conventions. Do not use a non-default convention
for `policy-step` until its preview is clearly better and understood.

Only after the previewed `guarded_delta_q` is tiny and sensible, execute
exactly one guarded policy step:

```bash
python scripts/ur3e/safe_bringup.py policy-step --target-offset 0 0 0.005 --execute --accept-risk
```

Do not run a continuous learned-policy loop on hardware until the single-step
tests are boring, repeatable, and bounded in the real workspace.
