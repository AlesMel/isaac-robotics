"""Standalone viewer for the procedural labyrinth generator.

Usage:
    python scripts/view_labyrinth.py --challenge corridor --difficulty 0.5
    python scripts/view_labyrinth.py --challenge pillar_forest --difficulty 0.8 --size 8
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize labyrinth layouts in Isaac Sim.")
parser.add_argument("--challenge", type=str, default="corridor",
                    choices=["corridor", "gate_slalom", "pillar_forest", "vertical_layers", "room_maze"],
                    help="Labyrinth challenge type.")
parser.add_argument("--difficulty", type=float, default=0.5, help="Difficulty 0.0 (easy) to 1.0 (hard).")
parser.add_argument("--size", type=float, default=6.0, help="Bounding box size in meters.")
parser.add_argument("--wall_height", type=float, default=1.5, help="Wall height in meters.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--num_goals", type=int, default=5, help="Number of goal markers to display.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import random
import isaaclab.sim as sim_utils

from isaac_robots.scripts.labyrinth_builder import (
    LabyrinthCfg,
    _CHALLENGES,
    _spawn_wall,
    _place_rings,
)
from isaac_robots.scripts.ring_obstacle import spawn_ring
from isaac_robots.scripts.goal_sampler import OccupancyGrid, GoalSampler


def main():
    # -- Simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=1 / 60)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(0.0, -args_cli.size, args_cli.size), target=(0.0, 0.0, args_cli.wall_height / 2))

    # -- Ground plane
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/GroundPlane", ground_cfg)

    # -- Light
    light_cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.95, 0.95, 1.0))
    light_cfg.func("/World/Light", light_cfg)

    # -- Build labyrinth at the origin
    cfg = LabyrinthCfg(
        challenge=args_cli.challenge,
        size=args_cli.size,
        wall_height=args_cli.wall_height,
        seed=args_cli.seed,
        difficulty=args_cli.difficulty,
    )
    rng = random.Random(cfg.seed)
    wall_specs = _CHALLENGES[cfg.challenge](cfg, rng)

    print(f"[INFO] Challenge: {cfg.challenge}  |  Difficulty: {cfg.difficulty}  |  Walls: {len(wall_specs)}")

    # -- Spawn walls
    for idx, (pos, size, rot_deg) in enumerate(wall_specs):
        prim_path = f"/World/labyrinth/wall_{idx:03d}"
        _spawn_wall(prim_path, pos, size, rot_deg)

    # -- Spawn rings
    rings = _place_rings(cfg, wall_specs, rng)
    print(f"[INFO] Rings: {len(rings)}")
    origin = (0.0, 0.0, 0.0)
    for ring_idx, ring in enumerate(rings):
        spawn_ring(f"/World/labyrinth/ring_{ring_idx:02d}", ring, origin)

    # -- Ring waypoint goals (approach → center → exit per ring)
    grid = OccupancyGrid(size=cfg.size, resolution=0.12)
    grid.mark_wall_specs(wall_specs)
    grid.mark_rings(rings)
    sampler = GoalSampler(grid, spawn_local=(0.0, 0.0, 0.3), min_dist_from_spawn=1.0, seed=cfg.seed)
    goals = sampler.sample_ring_waypoints(rings)

    # approach = yellow, center = green, exit = cyan (repeating pattern of 3)
    colors = [
        ((1.0, 0.8, 0.1), (0.4, 0.3, 0.0)),   # approach: yellow
        ((0.1, 0.9, 0.2), (0.0, 0.4, 0.1)),    # center:   green
        ((0.1, 0.8, 0.9), (0.0, 0.3, 0.4)),    # exit:     cyan
    ]
    for i, (gx, gy, gz) in enumerate(goals):
        diffuse, emissive = colors[i % 3]
        sphere_cfg = sim_utils.SphereCfg(
            radius=0.08,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=diffuse, emissive_color=emissive),
        )
        sphere_cfg.func(f"/World/labyrinth/goal_{i:02d}", sphere_cfg, translation=(gx, gy, gz))

    print(f"[INFO] Ring waypoints: {len(goals)} (yellow=approach, green=center, cyan=exit)")

    # -- Spawn marker at drone origin (blue sphere)
    spawn_cfg = sim_utils.SphereCfg(
        radius=0.06,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 1.0), emissive_color=(0.1, 0.2, 0.5)),
    )
    spawn_cfg.func("/World/labyrinth/spawn_marker", spawn_cfg, translation=(0.0, 0.0, 0.3))

    # -- Play simulation so you can orbit the camera
    sim.reset()
    print("[INFO] Labyrinth spawned. Use the viewport to orbit/inspect. Close the window to quit.")
    print("[INFO] Blue = spawn  |  Yellow = approach  |  Green = ring center  |  Cyan = exit")
    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()
