"""Shared constants for labyrinth training and evaluation scripts."""

# All challenge variants use the single registered Labyrinth task ID.
# The challenge type is selected via LabyrinthEnvCfg.labyrinth.challenge.
_TASK_ID = "Isaac-Robots-Labyrinth-Direct-v0"

CHALLENGE_TO_TASK = {
    "corridor":        _TASK_ID,
    "gate_slalom":     _TASK_ID,
    "pillar_forest":   _TASK_ID,
    "vertical_layers": _TASK_ID,
    "room_maze":       _TASK_ID,
}
