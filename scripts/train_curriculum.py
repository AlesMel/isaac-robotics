"""Multi-stage curriculum training for labyrinth environments.

Chains training runs at increasing difficulty levels, loading the previous
checkpoint as starting weights for each new stage.

Usage:
    python scripts/train_curriculum.py --challenge corridor --num_envs 4096
    python scripts/train_curriculum.py --challenge room_maze --difficulties 0.3 0.5 0.7 0.9
    python scripts/train_curriculum.py --challenge pillar_forest --timesteps-per-stage 2000000
"""

import argparse
import glob
import os
import subprocess
import sys

CHALLENGE_TO_TASK = {
    "corridor": "Isaac-Robots-Corridor-Direct-v0",
    "gate_slalom": "Isaac-Robots-GateSlalom-Direct-v0",
    "pillar_forest": "Isaac-Robots-PillarForest-Direct-v0",
    "vertical_layers": "Isaac-Robots-VerticalLayers-Direct-v0",
    "room_maze": "Isaac-Robots-RoomMaze-Direct-v0",
}


def find_latest_checkpoint(log_root: str) -> str | None:
    """Find the most recent checkpoint file under a log directory."""
    pattern = os.path.join(log_root, "**", "checkpoints", "best_agent.pt")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        # fall back to any .pt file
        pattern = os.path.join(log_root, "**", "checkpoints", "*.pt")
        matches = glob.glob(pattern, recursive=True)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(description="Curriculum training for labyrinth tasks.")
    parser.add_argument(
        "--challenge", type=str, required=True,
        choices=list(CHALLENGE_TO_TASK.keys()),
        help="Labyrinth challenge type.",
    )
    parser.add_argument(
        "--difficulties", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9],
        help="Difficulty levels to train through (ascending).",
    )
    parser.add_argument(
        "--timesteps-per-stage", type=int, default=2_000_000,
        help="Training timesteps per difficulty stage.",
    )
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--algorithm", type=str, default="PPO", choices=["PPO", "SAC"],
        help="RL algorithm.",
    )
    parser.add_argument("--headless", action="store_true", help="Run headless.")
    args = parser.parse_args()

    task = CHALLENGE_TO_TASK[args.challenge]
    checkpoint = None

    for i, difficulty in enumerate(sorted(args.difficulties)):
        print(f"\n{'='*60}")
        print(f"  Stage {i+1}/{len(args.difficulties)}: difficulty={difficulty}")
        if checkpoint:
            print(f"  Resuming from: {checkpoint}")
        print(f"{'='*60}\n")

        cmd = [
            sys.executable, "scripts/skrl/train.py",
            f"--task={task}",
            f"--algorithm={args.algorithm}",
            f"--seed={args.seed}",
            f"--max_iterations={args.timesteps_per_stage // 48}",  # approx: timesteps / rollouts
        ]
        if args.num_envs is not None:
            cmd.append(f"--num_envs={args.num_envs}")
        if args.headless:
            cmd.append("--headless")
        if checkpoint:
            cmd.append(f"--checkpoint={checkpoint}")

        # Override difficulty via Hydra
        cmd.append(f"env.labyrinth.difficulty={difficulty}")

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if result.returncode != 0:
            print(f"[ERROR] Stage {i+1} failed with return code {result.returncode}")
            sys.exit(result.returncode)

        # Find the checkpoint produced by this stage
        log_root = os.path.join("logs", "skrl", "labyrinth_direct")
        checkpoint = find_latest_checkpoint(log_root)
        if checkpoint is None:
            print(f"[WARN] No checkpoint found after stage {i+1}, next stage starts fresh.")

    print(f"\n{'='*60}")
    print(f"  Curriculum training complete!")
    if checkpoint:
        print(f"  Final checkpoint: {checkpoint}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
