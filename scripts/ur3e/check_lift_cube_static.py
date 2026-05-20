"""Dependency-free static checks for the UR3e lift-cube task.

Run from the repo root with:

    python3 scripts/ur3e/check_lift_cube_static.py
"""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _class_attr_node(module_path: Path, class_name: str, attr_name: str) -> ast.AST:
    module = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    if stmt.target.id == attr_name:
                        return stmt.value
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name) and target.id == attr_name:
                            return stmt.value
    raise AssertionError(f"{class_name}.{attr_name} not found in {module_path}")


def _class_assignment(module_path: Path, class_name: str, attr_name: str):
    return ast.literal_eval(_class_attr_node(module_path, class_name, attr_name))


def _class_bases(module_path: Path, class_name: str) -> list[str]:
    module = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return [base.id for base in node.bases if isinstance(base, ast.Name)]
    raise AssertionError(f"class {class_name} not found in {module_path}")


def _class_methods(module_path: Path, class_name: str) -> set[str]:
    module = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                stmt.name
                for stmt in node.body
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    raise AssertionError(f"class {class_name} not found in {module_path}")


def check_suction_axes() -> None:
    suction_path = (
        REPO_ROOT
        / "source/isaac_robots/isaac_robots/tasks/direct/_shared/grippers/suction.py"
    )
    assert _class_assignment(suction_path, "SuctionGripperCfg", "approach_axis_local") == (
        0.0,
        0.0,
        1.0,
    )
    assert _class_assignment(suction_path, "SuctionGripperCfg", "cup_axis_local") == (
        0.0,
        0.0,
        -1.0,
    )
    assert _class_assignment(suction_path, "SuctionGripperCfg", "cup_tip_offset_local") == (
        0.0,
        0.0,
        0.03,
    )


def check_cube_sampling_inside_table() -> None:
    cfg_path = (
        REPO_ROOT
        / "source/isaac_robots/isaac_robots/tasks/direct/isaac_ur3e_lift_cube/"
        / "ur3e_lift_cube_env_cfg.py"
    )
    cfg_text = cfg_path.read_text(encoding="utf-8")
    assert "Props/Mounts/SeattleLabTable/table_instanceable.usd" in cfg_text

    x_min, x_max = _class_assignment(cfg_path, "UR3eLiftCubeEnvCfg", "cube_pos_x_range")
    y_min, y_max = _class_assignment(cfg_path, "UR3eLiftCubeEnvCfg", "cube_pos_y_range")
    cube_half_extent = _class_assignment(cfg_path, "UR3eLiftCubeEnvCfg", "cube_half_extent")

    # UR base x/y are flipped into Isaac world x/y by _ur_base_flip = [-1, -1, 1].
    world_x_min, world_x_max = sorted((-x_min, -x_max))
    world_y_min, world_y_max = sorted((-y_min, -y_max))

    table_center_x, table_center_y = 0.5, 0.0
    table_half_x, table_half_y = 0.4, 0.4
    assert world_x_min - cube_half_extent >= table_center_x - table_half_x
    assert world_x_max + cube_half_extent <= table_center_x + table_half_x
    assert world_y_min - cube_half_extent >= table_center_y - table_half_y
    assert world_y_max + cube_half_extent <= table_center_y + table_half_y


def check_hande_gripper_cfg() -> None:
    hande_path = (
        REPO_ROOT
        / "source/isaac_robots/isaac_robots/tasks/direct/_shared/grippers/robotiq_hande.py"
    )
    # A real-physics parallel jaw must NOT be a suction gripper -- it implements
    # the gripper interface directly.
    assert "GripperBase" in _class_bases(hande_path, "RobotiqHandEGripper")
    assert "SuctionGripper" not in _class_bases(hande_path, "RobotiqHandEGripper")
    assert "GripperCfg" in _class_bases(hande_path, "RobotiqHandEGripperCfg")
    assert "SuctionGripperCfg" not in _class_bases(hande_path, "RobotiqHandEGripperCfg")

    # Action/observation contract the lift env sizes its spaces from.
    assert _class_assignment(hande_path, "RobotiqHandEGripperCfg", "action_dim") == 1
    assert _class_assignment(hande_path, "RobotiqHandEGripperCfg", "obs_dim") == 2

    # class_type must point at the Hand-E gripper class (non-literal -> check the name).
    class_type_node = _class_attr_node(hande_path, "RobotiqHandEGripperCfg", "class_type")
    assert isinstance(class_type_node, ast.Name) and class_type_node.id == "RobotiqHandEGripper"

    # The lift env calls these on the gripper unconditionally; they must exist.
    # In particular update_attachment must be present (and a no-op) for a real
    # friction grasp, since the env always calls it.
    methods = _class_methods(hande_path, "RobotiqHandEGripper")
    for required in (
        "setup_scene",
        "apply_action",
        "get_observation",
        "reset",
        "register_graspable_object",
        "compute_grasp_metrics",
        "update_attachment",
        "is_holding",
    ):
        assert required in methods, f"RobotiqHandEGripper is missing {required}"


def check_hande_env_cfg_and_registration() -> None:
    cfg_path = (
        REPO_ROOT
        / "source/isaac_robots/isaac_robots/tasks/direct/isaac_ur3e_lift_cube/"
        / "ur3e_lift_cube_env_cfg.py"
    )
    assert "UR3eLiftCubeEnvCfg" in _class_bases(cfg_path, "UR3eLiftCubeHandEEnvCfg")

    init_path = (
        REPO_ROOT
        / "source/isaac_robots/isaac_robots/tasks/direct/isaac_ur3e_lift_cube/__init__.py"
    )
    init_text = init_path.read_text(encoding="utf-8")
    assert "Isaac-Robots-UR3e-Lift-Cube-HandE-Direct-v0" in init_text
    assert "UR3eLiftCubeHandEEnvCfg" in init_text


def main() -> None:
    check_suction_axes()
    check_cube_sampling_inside_table()
    check_hande_gripper_cfg()
    check_hande_env_cfg_and_registration()
    print("UR3e lift-cube static checks passed.")


if __name__ == "__main__":
    main()
