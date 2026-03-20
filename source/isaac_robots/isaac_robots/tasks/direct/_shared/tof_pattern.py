import torch
from isaaclab.sensors.ray_caster import patterns
from isaaclab.utils import configclass


def crazyflie_tof_pattern(cfg: "CrazyflieToFPatternCfg", device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """6 orthogonal rays: Multi-ranger (5) + Flow Deck down (1)."""
    directions = torch.tensor([
        [ 1.0,  0.0,  0.0],  # front
        [-1.0,  0.0,  0.0],  # back
        [ 0.0,  1.0,  0.0],  # left
        [ 0.0, -1.0,  0.0],  # right
        [ 0.0,  0.0,  1.0],  # up
        [ 0.0,  0.0, -1.0],  # down
    ], dtype=torch.float32, device=device)
    starts = torch.zeros_like(directions)
    return starts, directions


@configclass
class CrazyflieToFPatternCfg(patterns.PatternBaseCfg):
    func = crazyflie_tof_pattern
