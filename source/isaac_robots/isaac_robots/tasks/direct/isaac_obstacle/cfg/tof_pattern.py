import torch
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns
from isaaclab.utils import configclass

def crazyflie_tof_pattern(cfg: "CrazyflieToFPatternCfg", device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates 6 orthogonal rays simulating the Crazyflie 
    Multi-ranger (5) and Flow Deck (1).
    """
    # Ray directions [X, Y, Z] relative to the drone's body
    directions = torch.tensor([
        [ 1.0,  0.0,  0.0],  # 0: Front (+X)
        [-1.0,  0.0,  0.0],  # 1: Back  (-X)
        [ 0.0,  1.0,  0.0],  # 2: Left  (+Y)
        [ 0.0, -1.0,  0.0],  # 3: Right (-Y)
        [ 0.0,  0.0,  1.0],  # 4: Up    (+Z)
        [ 0.0,  0.0, -1.0],  # 5: Down  (-Z) - Flow Deck
    ], dtype=torch.float32, device=device)
    
    # All rays start at the exact origin (0, 0, 0) of the sensor frame
    starts = torch.zeros_like(directions)
    
    return starts, directions

@configclass
class CrazyflieToFPatternCfg(patterns.PatternBaseCfg):
    """Configuration class to register the custom ToF pattern."""
    func = crazyflie_tof_pattern