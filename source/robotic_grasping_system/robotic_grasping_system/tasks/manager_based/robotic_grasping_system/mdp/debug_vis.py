"""Debug visualization for approach-angle and grasp reward frames.

Uses VisualizationMarkers (the proper Isaac Lab API) instead of debug_draw.

Usage in play.py (or any eval script), before the step loop:

    from robotic_grasping_system.tasks.manager_based.robotic_grasping_system import mdp
    mdp.enable_debug_vis()           # turn on
    mdp.set_debug_vis_envs([0])      # which envs to show (default: [0])

Visualization key
-----------------
Hand frame (FRAME marker at R_hand_base):
  Red   arrow : hand X axis  — the approach / closing direction
  Green arrow : hand Y axis
  Blue  arrow : hand Z axis

Object axes:
  Yellow cylinder : object long  axis  (fingers should be perpendicular to this)
  Cyan   cylinder : object short axis

Grasp axis:
  Magenta cylinder : thumb-tip → finger-centroid span
  White  sphere    : object centre position
  Orange sphere    : projection of object centre onto grasp axis
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.math import matrix_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg

# ---------------------------------------------------------------------------
# Module-level toggle
# ---------------------------------------------------------------------------
_debug_vis_enabled: bool = False
_debug_vis_env_ids: list[int] = [0]

# Lazy-initialized markers (created on first use, after sim is running)
_hand_frame_markers: VisualizationMarkers | None = None
_obj_long_markers:   VisualizationMarkers | None = None
_obj_short_markers:  VisualizationMarkers | None = None
_grasp_axis_markers: VisualizationMarkers | None = None
_obj_pos_markers:    VisualizationMarkers | None = None
_proj_pos_markers:   VisualizationMarkers | None = None


def enable_debug_vis() -> None:
    global _debug_vis_enabled
    _debug_vis_enabled = True
    print("[debug_vis] enabled")


def disable_debug_vis() -> None:
    global _debug_vis_enabled
    _debug_vis_enabled = False


def set_debug_vis_envs(env_ids: list[int]) -> None:
    global _debug_vis_env_ids
    _debug_vis_env_ids = list(env_ids)


def is_debug_vis_enabled() -> bool:
    return _debug_vis_enabled


# ---------------------------------------------------------------------------
# Math helper: quaternion rotating +Z to an arbitrary unit vector d
# Isaac Lab quaternion convention: [w, x, y, z]
# ---------------------------------------------------------------------------

def _quat_z_to_vec(d: torch.Tensor) -> torch.Tensor:
    """(N, 3) unit vectors → (N, 4) quaternions [w,x,y,z] rotating +Z onto d."""
    # half-vector method: h = normalize(Z + d)
    h = d.clone()
    h[..., 2] = h[..., 2] + 1.0           # Z + d
    h_len = torch.norm(h, dim=-1, keepdim=True)

    # Antiparallel case (d ≈ -Z): rotate π around X  →  q = [0, 1, 0, 0]
    antiparallel = (h_len.squeeze(-1) < 1e-3)
    h_len = h_len.clamp(min=1e-6)
    h = h / h_len

    # cross(Z, h) where Z=(0,0,1):  (-h.y, h.x, 0)
    w  =  h[..., 2:3]          # dot(Z, h) = h.z
    qx = -h[..., 1:2]          # -h.y
    qy =  h[..., 0:1]          # h.x
    qz =  torch.zeros_like(w)
    q  = torch.cat([w, qx, qy, qz], dim=-1)

    ap = torch.zeros_like(q)
    ap[..., 1] = 1.0            # [0, 1, 0, 0]
    q = torch.where(antiparallel.unsqueeze(-1), ap, q)
    return q / torch.norm(q, dim=-1, keepdim=True).clamp(min=1e-6)


# ---------------------------------------------------------------------------
# Marker factory helpers
# ---------------------------------------------------------------------------

def _cylinder_cfg(prim_path: str, color: tuple[float, float, float]) -> VisualizationMarkersCfg:
    """Thin cylinder pointing along +Z, height=1 (scaled per-instance)."""
    return VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "cyl": sim_utils.CylinderCfg(
                radius=0.006,
                height=1.0,
                axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=color,
                    emissive_color=color,
                ),
            )
        },
    )


def _sphere_cfg(prim_path: str, color: tuple[float, float, float], radius: float = 0.018) -> VisualizationMarkersCfg:
    return VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "sph": sim_utils.SphereCfg(
                radius=radius,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=color,
                    emissive_color=color,
                ),
            )
        },
    )


def _frame_cfg(prim_path: str, scale: float = 0.15) -> VisualizationMarkersCfg:
    """Standard RGB coordinate frame marker (Isaac Lab built-in)."""
    from isaaclab.markers.config import FRAME_MARKER_CFG
    cfg = FRAME_MARKER_CFG.copy()
    cfg.prim_path = prim_path
    cfg.markers["frame"].scale = (scale, scale, scale)
    return cfg


# ---------------------------------------------------------------------------
# Lazy initializer
# ---------------------------------------------------------------------------

def _ensure_markers_created() -> bool:
    global _hand_frame_markers, _obj_long_markers, _obj_short_markers
    global _grasp_axis_markers, _obj_pos_markers, _proj_pos_markers

    if _hand_frame_markers is not None:
        return True
    try:
        _hand_frame_markers = VisualizationMarkers(_frame_cfg("/Visuals/dbg/hand_frame"))
        _obj_long_markers   = VisualizationMarkers(_cylinder_cfg("/Visuals/dbg/obj_long",  (1.0, 1.0, 0.0)))
        _obj_short_markers  = VisualizationMarkers(_cylinder_cfg("/Visuals/dbg/obj_short", (0.0, 1.0, 1.0)))
        _grasp_axis_markers = VisualizationMarkers(_cylinder_cfg("/Visuals/dbg/grasp_axis",(1.0, 0.0, 1.0)))
        _obj_pos_markers    = VisualizationMarkers(_sphere_cfg("/Visuals/dbg/obj_pos",   (1.0, 1.0, 1.0)))
        _proj_pos_markers   = VisualizationMarkers(_sphere_cfg("/Visuals/dbg/proj_pos",  (1.0, 0.5, 0.0)))
        return True
    except Exception as e:
        print(f"[debug_vis] Could not create markers: {e}")
        return False


# ---------------------------------------------------------------------------
# Approach-angle visualization
# ---------------------------------------------------------------------------

def draw_approach_angle_frames(env: "ManagerBasedRLEnv",
                                robot_cfg: "SceneEntityCfg",
                                object_cfg: "SceneEntityCfg") -> None:
    if not _debug_vis_enabled:
        return
    if not _ensure_markers_created():
        return

    ids = [i for i in _debug_vis_env_ids if i < env.num_envs]
    if not ids:
        return
    env_ids = torch.tensor(ids, device=env.device)
    N = len(env_ids)
    SCALE = 0.15  # axis arrow length in metres

    robot = env.scene[robot_cfg.name]
    obj   = env.scene[object_cfg.name]

    # Hand base pose — always use R_hand_base regardless of robot_cfg.body_names
    hand_body_ids, _ = robot.find_bodies(["R_hand_base"])
    hand_pos  = robot.data.body_pos_w[env_ids, hand_body_ids[0], :]   # (N, 3)
    hand_quat = robot.data.body_quat_w[env_ids, hand_body_ids[0], :]  # (N, 4)

    # Object axes
    obj_pos  = obj.data.root_pos_w[env_ids]          # (N, 3)
    obj_quat = obj.data.root_quat_w[env_ids]         # (N, 4)
    obj_mat  = matrix_from_quat(obj_quat)
    obj_local_x = obj_mat[..., 0]
    obj_local_z = obj_mat[..., 2]

    if hasattr(env, "object_bbox_dims"):
        bbox = env.object_bbox_dims[env_ids]
        x_longer   = (bbox[:, 0] > bbox[:, 2]).unsqueeze(-1)
        long_axis  = torch.where(x_longer, obj_local_x, obj_local_z)
        short_axis = torch.where(x_longer, obj_local_z, obj_local_x)
    else:
        long_axis  = obj_local_x
        short_axis = obj_local_z

    # --- Hand frame (uses hand quaternion directly — shows X/Y/Z in red/green/blue) ---
    _hand_frame_markers.visualize(
        translations=hand_pos,
        orientations=hand_quat,
    )

    # --- Object long axis (yellow cylinder) ---
    long_dir  = long_axis / torch.norm(long_axis,  dim=-1, keepdim=True).clamp(min=1e-6)
    long_mid  = obj_pos + long_dir * (SCALE / 2.0)
    long_quat = _quat_z_to_vec(long_dir)
    _obj_long_markers.visualize(
        translations=long_mid,
        orientations=long_quat,
        scales=torch.tensor([[1.0, 1.0, SCALE]] * N, device=env.device),
    )

    # --- Object short axis (cyan cylinder) ---
    short_dir  = short_axis / torch.norm(short_axis, dim=-1, keepdim=True).clamp(min=1e-6)
    short_mid  = obj_pos + short_dir * (SCALE / 2.0)
    short_quat = _quat_z_to_vec(short_dir)
    _obj_short_markers.visualize(
        translations=short_mid,
        orientations=short_quat,
        scales=torch.tensor([[1.0, 1.0, SCALE]] * N, device=env.device),
    )


# ---------------------------------------------------------------------------
# Grasp visualization
# ---------------------------------------------------------------------------

def draw_grasp_frames(env: "ManagerBasedRLEnv",
                      robot_cfg: "SceneEntityCfg",
                      object_cfg: "SceneEntityCfg") -> None:
    if not _debug_vis_enabled:
        return
    if not _ensure_markers_created():
        return

    ids = [i for i in _debug_vis_env_ids if i < env.num_envs]
    if not ids:
        return
    env_ids = torch.tensor(ids, device=env.device)
    N = len(env_ids)

    robot = env.scene[robot_cfg.name]
    obj   = env.scene[object_cfg.name]

    thumb_ids,  _ = robot.find_bodies(["R_th_tip"])
    finger_ids, _ = robot.find_bodies(["R_ff_tip", "R_mf_tip", "R_rf_tip", "R_lf_tip"])

    thumb_pos  = robot.data.body_pos_w[env_ids, thumb_ids[0], :]
    finger_pos = robot.data.body_pos_w[env_ids][:, finger_ids, :].mean(dim=1)
    obj_pos    = obj.data.root_pos_w[env_ids]

    grasp_vec = finger_pos - thumb_pos
    grasp_len = torch.norm(grasp_vec, dim=-1, keepdim=True).clamp(min=1e-6)
    grasp_dir = grasp_vec / grasp_len

    # Projection of object onto the grasp axis
    t = ((obj_pos - thumb_pos) * grasp_dir).sum(dim=-1, keepdim=True)
    t = (t / grasp_len).clamp(0.0, 1.0)
    closest = thumb_pos + t * grasp_vec

    # --- Grasp axis cylinder (magenta): midpoint, oriented Z→grasp_dir, length=grasp_len ---
    axis_mid  = thumb_pos + grasp_vec * 0.5
    axis_quat = _quat_z_to_vec(grasp_dir)
    axis_len  = grasp_len.squeeze(-1)   # (N,)
    scales_axis = torch.stack([torch.ones(N, device=env.device),
                                torch.ones(N, device=env.device),
                                axis_len], dim=-1)  # (N, 3)
    _grasp_axis_markers.visualize(
        translations=axis_mid,
        orientations=axis_quat,
        scales=scales_axis,
    )

    # --- Object centre sphere (white) ---
    _obj_pos_markers.visualize(translations=obj_pos)

    # --- Projection sphere (orange) ---
    _proj_pos_markers.visualize(translations=closest)
