"""Task-specific environment wrappers that allow the fine_grained tracking
reward kernel to be swapped at config-time via a ``kernel_name`` parameter.

Each wrapper subclasses the upstream / project-local env config and overrides
the ``object_goal_tracking_fine_grained`` reward term to use the kernel
specified in :mod:`velgate_bench.kernels`.

Gym IDs follow the pattern::

    Velgate-Bench-<Task>-<Kernel>-v0

Examples::

    Velgate-Bench-Franka-Lift-Tanh-v0
    Velgate-Bench-Franka-Lift-Gaussian-v0
    Velgate-Bench-Franka-Lift-VelocityGatedTanh-v0
    Velgate-Bench-UR3e-HandE-Lift-Tanh-v0
    Velgate-Bench-UR3e-HandE-Lift-Gaussian-v0
    Velgate-Bench-UR3e-HandE-Lift-VelocityGatedTanh-v0

Importing this package triggers registration via the ``register_all()``
helper, idempotently. The sweep orchestrator imports it once at start.
"""

from .franka_lift import register_franka_lift_variants
from .ur3e_lift import register_ur3e_lift_variants


def register_all() -> None:
    """Register all (task, kernel) gym variants. Idempotent."""
    register_franka_lift_variants()
    register_ur3e_lift_variants()


# Side-effect: register on package import. Idempotent (each register_X_variants
# checks `if gym_id in gym.registry` before adding). This matches the standard
# Gymnasium pattern (e.g. `import gym_robotics` to make its IDs visible).
register_all()


__all__ = ["register_all"]
