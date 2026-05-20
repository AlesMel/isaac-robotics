"""Single-cup suction gripper (Robotiq EPick-style).

Grasping model -- read this before extending:

    The cup does NOT use a real fixed joint at runtime. While suction is
    "on" and a registered graspable object is within ``grasp_distance`` of
    the TCP, the object's root pose is overwritten each physics step so it
    tracks the TCP transform. This is a kinematic approximation -- there is
    no slip, no leak, no finite hold force. Any policy that exploits magic
    adhesion in sim will fail on the real robot.

The implementation is intentionally simple: tasks register one graspable rigid
object and the gripper kinematically attaches it when the TCP is close to the
object's grasp point, aligned with the surface normal, and suction is on.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, quat_apply, subtract_frame_transforms

from .base import GripperBase, GripperCfg


class SuctionGripper(GripperBase):
    """Kinematic single-cup suction gripper.

    Action layout (per env): ``[suction_cmd]`` in ``[-1, 1]``.
        > 0 -> suction on, attempt to grasp.
        <= 0 -> suction off, release.

    Observation layout (per env): ``[is_holding]`` (0.0 or 1.0).
    """

    def __init__(self, cfg: "SuctionGripperCfg", env) -> None:
        super().__init__(cfg, env)
        self._is_on = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        self._is_holding = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        self._object: RigidObject | None = None
        self._tcp_body_id: int | None = None

        self._grasp_offset_w = torch.zeros(self._num_envs, 3, device=self._device)
        self._surface_normal_w = torch.zeros(self._num_envs, 3, device=self._device)
        self._surface_normal_w[:, 2] = 1.0
        self._approach_axis_local = torch.tensor(
            cfg.approach_axis_local, dtype=torch.float, device=self._device
        ).repeat(self._num_envs, 1)

        self._held_object_pos_tcp = torch.zeros(self._num_envs, 3, device=self._device)
        self._held_object_quat_tcp = torch.zeros(self._num_envs, 4, device=self._device)
        self._held_object_quat_tcp[:, 0] = 1.0
        self._cup: RigidObject | None = None

    def setup_scene(self) -> None:
        """Spawn a kinematic collision body for the suction cup.

        This mirrors Isaac Lab's UR10 suction examples more closely than a
        marker: the cup is a real rigid object with collision geometry, but
        it is kinematically driven from the robot TCP so it can be swapped
        later for a proper fixed-joint gripper USD.
        """
        if not self.cfg.visualize:
            return

        cup_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/SuctionCup",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=sim_utils.CylinderCfg(
                radius=self.cfg.visual_cup_radius,
                height=self.cfg.visual_cup_length,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                    max_depenetration_velocity=5.0,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=self.cfg.contact_offset,
                    rest_offset=0.0,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=self.cfg.visual_color),
            ),
        )
        self._cup = RigidObject(cup_cfg)
        self._env.scene.rigid_objects["suction_cup"] = self._cup

    @property
    def is_on(self) -> torch.Tensor:
        return self._is_on

    @property
    def is_holding(self) -> torch.Tensor:
        return self._is_holding

    def register_graspable_object(
        self,
        rigid_object: "RigidObject",
        tcp_body_id: int,
        grasp_offset_w: tuple[float, float, float] = (0.0, 0.0, 0.0),
        surface_normal_w: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> None:
        """Register the single rigid object this suction cup can hold."""
        self._object = rigid_object
        self._tcp_body_id = tcp_body_id
        self._grasp_offset_w[:] = torch.tensor(grasp_offset_w, dtype=torch.float, device=self._device)

        normal = torch.tensor(surface_normal_w, dtype=torch.float, device=self._device)
        normal = normal / torch.clamp(torch.linalg.norm(normal), min=1.0e-6)
        self._surface_normal_w[:] = normal

    def set_debug_vis(self, visible: bool) -> None:
        del visible

    def visualize(self) -> None:
        self.update_collision_body()

    def reset(self, env_ids: torch.Tensor) -> None:
        self._is_on[env_ids] = False
        self._is_holding[env_ids] = False
        self._held_object_pos_tcp[env_ids] = 0.0
        self._held_object_quat_tcp[env_ids] = torch.tensor(
            (1.0, 0.0, 0.0, 0.0), dtype=torch.float, device=self._device
        )
        if self._cup is not None:
            self._cup.reset(env_ids)
            self.update_collision_body(env_ids)

    def apply_action(self, action: torch.Tensor) -> None:
        # action shape: (num_envs, 1) -- threshold at zero.
        self._is_on = action[:, 0] > 0.0

        # Release immediately when suction is off.
        self._is_holding = torch.logical_and(self._is_holding, self._is_on)

        if self._object is None or self._tcp_body_id is None:
            return

        distance, alignment, ready = self.compute_grasp_metrics()
        del distance, alignment

        newly_grasped = torch.logical_and(torch.logical_and(self._is_on, ~self._is_holding), ready)
        if torch.any(newly_grasped):
            self._store_attachment_offsets(newly_grasped)
            self._is_holding = torch.logical_or(self._is_holding, newly_grasped)

    def get_observation(self) -> torch.Tensor:
        return self._is_holding.float().unsqueeze(-1)

    def compute_grasp_metrics(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return distance, alignment cosine, and readiness for the registered object."""
        if self._object is None or self._tcp_body_id is None:
            zeros = torch.zeros(self._num_envs, device=self._device)
            ready = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
            return zeros, zeros, ready

        tcp_quat_w = self._env._robot.data.body_quat_w[:, self._tcp_body_id]
        cup_tip_w = self._cup_tip_pos_w()
        grasp_point_w = self._object.data.root_pos_w + self._grasp_offset_w

        distance = torch.linalg.norm(grasp_point_w - cup_tip_w, dim=-1)
        approach_axis_w = quat_apply(tcp_quat_w, self._approach_axis_local)
        alignment = torch.sum(-approach_axis_w * self._surface_normal_w, dim=-1)
        ready = torch.logical_and(distance <= self.cfg.grasp_distance, alignment >= self.cfg.normal_align_cos)
        return distance, alignment, ready

    def _store_attachment_offsets(self, env_mask: torch.Tensor) -> None:
        env_ids = torch.nonzero(env_mask, as_tuple=False).squeeze(-1)
        tcp_quat_w = self._env._robot.data.body_quat_w[env_ids, self._tcp_body_id]
        cup_tip_w = self._cup_tip_pos_w()[env_ids]
        object_pos_w = self._object.data.root_pos_w[env_ids]
        object_quat_w = self._object.data.root_quat_w[env_ids]
        rel_pos, rel_quat = subtract_frame_transforms(cup_tip_w, tcp_quat_w, object_pos_w, object_quat_w)
        self._held_object_pos_tcp[env_ids] = rel_pos
        self._held_object_quat_tcp[env_ids] = rel_quat

    def update_attachment(self) -> None:
        """Snap the registered object to the TCP in environments where suction holds it."""
        if self._object is None or self._tcp_body_id is None:
            return

        env_ids = torch.nonzero(self._is_holding, as_tuple=False).squeeze(-1)
        if env_ids.numel() == 0:
            return

        tcp_quat_w = self._env._robot.data.body_quat_w[env_ids, self._tcp_body_id]
        cup_tip_w = self._cup_tip_pos_w()[env_ids]
        object_pos_w, object_quat_w = combine_frame_transforms(
            cup_tip_w,
            tcp_quat_w,
            self._held_object_pos_tcp[env_ids],
            self._held_object_quat_tcp[env_ids],
        )
        object_pose_w = torch.cat((object_pos_w, object_quat_w), dim=-1)
        object_velocity_w = torch.zeros(env_ids.numel(), 6, device=self._device)
        self._object.write_root_pose_to_sim(object_pose_w, env_ids=env_ids)
        self._object.write_root_velocity_to_sim(object_velocity_w, env_ids=env_ids)

    def update_collision_body(self, env_ids: torch.Tensor | None = None) -> None:
        if self._cup is None or self._tcp_body_id is None:
            return

        if env_ids is None:
            env_ids = torch.arange(self._num_envs, dtype=torch.long, device=self._device)

        tcp_quat_w = self._env._robot.data.body_quat_w[env_ids, self._tcp_body_id]
        cup_tip_w = self._cup_tip_pos_w()[env_ids]
        cup_axis_local = self._local_tensor(self.cfg.cup_axis_local).repeat(env_ids.numel(), 1)
        cup_center_w = cup_tip_w + quat_apply(tcp_quat_w, cup_axis_local * (0.5 * self.cfg.visual_cup_length))
        cup_pose_w = torch.cat((cup_center_w, tcp_quat_w), dim=-1)
        cup_velocity_w = torch.zeros(env_ids.numel(), 6, device=self._device)
        self._cup.write_root_pose_to_sim(cup_pose_w, env_ids=env_ids)
        self._cup.write_root_velocity_to_sim(cup_velocity_w, env_ids=env_ids)

    def _cup_tip_pos_w(self) -> torch.Tensor:
        tcp_pos_w = self._env._robot.data.body_pos_w[:, self._tcp_body_id]
        tcp_quat_w = self._env._robot.data.body_quat_w[:, self._tcp_body_id]
        return tcp_pos_w + quat_apply(
            tcp_quat_w,
            self._local_tensor(self.cfg.cup_tip_offset_local).repeat(self._num_envs, 1),
        )

    def _local_tensor(self, value: tuple[float, float, float]) -> torch.Tensor:
        return torch.tensor(value, dtype=torch.float, device=self._device)


@configclass
class SuctionGripperCfg(GripperCfg):
    class_type: type = SuctionGripper
    action_dim: int = 1
    obs_dim: int = 1

    tcp_frame_name: str = "tool0"
    """Body frame of the suction cup tip on the arm."""

    grasp_distance: float = 0.02
    """Max distance (m) from TCP to the registered grasp point."""

    normal_align_cos: float = 0.7
    """Min alignment cosine between cup approach axis and surface normal."""

    approach_axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0)
    """TCP-local direction that points from the cup toward the contacted surface."""

    cup_axis_local: tuple[float, float, float] = (0.0, 0.0, -1.0)
    """TCP-local direction from the cup tip back through the visual cup body."""

    cup_tip_offset_local: tuple[float, float, float] = (0.0, 0.0, 0.03)
    """TCP-local offset from tool0 to the suction contact point."""

    visualize: bool = True
    """Draw a visual-only suction cup marker at the TCP frame."""

    visual_stem_radius: float = 0.012
    visual_stem_length: float = 0.035
    visual_cup_radius: float = 0.035
    visual_cup_length: float = 0.04
    visual_color: tuple[float, float, float] = (0.95, 0.12, 0.02)
    contact_offset: float = 0.005
