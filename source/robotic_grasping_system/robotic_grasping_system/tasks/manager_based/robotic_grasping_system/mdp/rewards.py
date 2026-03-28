"""Reward terms: reach -> grasp -> lift.

Based on CrossDex (ICLR 2025) and Dexterous Functional Grasping (CoRL 2023).
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from . import debug_vis as _dbv

_FINGERTIP_BODIES = ["R_ff_tip", "R_mf_tip", "R_rf_tip", "R_lf_tip", "R_th_tip"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _mean_fingertip_dist(robot, obj_pos_w: torch.Tensor, body_names: list[str]) -> torch.Tensor:
    """Mean Euclidean distance from named fingertips to the object centre. (N,)"""
    ids, _ = robot.find_bodies(body_names)
    ft_pos = robot.data.body_pos_w[:, ids, :]      
    return torch.norm(ft_pos - obj_pos_w.unsqueeze(1), dim=-1).mean(dim=-1)


def _update_contact_running_mean(env: "ManagerBasedRLEnv", contact_vals: torch.Tensor, window: int = 2000) -> None:
    """Circular buffer of per-step mean fingertip contact count (0–5).

    Stores env._contact_running_mean (float) — used by finger_close_reward
    and lift_reward to decide when enough fingers are touching the object.
    """
    if not hasattr(env, "_contact_mean_buf"):
        env._contact_mean_buf = torch.zeros(window, device=env.device)
        env._contact_mean_idx = 0
    env._contact_mean_buf[env._contact_mean_idx] = contact_vals.mean().detach()
    env._contact_mean_idx = (env._contact_mean_idx + 1) % window
    env._contact_running_mean = env._contact_mean_buf.mean().item()


def _update_grasp_running_mean(env: "ManagerBasedRLEnv", grasp_vals: torch.Tensor, window: int = 2000) -> None:
    """Maintain a circular buffer of per-step mean grasp reward.

    Stores env._grasp_running_mean (float) — used by lift_reward to decide
    when grasping is consistent enough to start training lift.
    """
    if not hasattr(env, "_grasp_mean_buf"):
        env._grasp_mean_buf = torch.zeros(window, device=env.device)
        env._grasp_mean_idx = 0
    env._grasp_mean_buf[env._grasp_mean_idx] = grasp_vals.mean().detach()
    env._grasp_mean_idx = (env._grasp_mean_idx + 1) % window
    env._grasp_running_mean = env._grasp_mean_buf.mean().item()


def _proximity(dist: torch.Tensor, blend_distance: float, blend_sharpness: float) -> torch.Tensor:
    """Sigmoid weight: ~0 when far, ~1 when close.

    Transition window (5% → 95%): blend_distance ± 3 / blend_sharpness
      e.g. 0.15 m, sharpness=25 → window [0.03, 0.27] m
    """
    return torch.sigmoid(blend_sharpness * (blend_distance - dist))


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def reach_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Mean fingertip-to-object distance, faded out as the hand gets close.
    Applied with a negative weight in the config.  Returns (N,) >= 0.
    """
    robot = env.scene[robot_cfg.name]
    obj   = env.scene[object_cfg.name]

    mean_dist = _mean_fingertip_dist(robot, obj.data.root_pos_w, robot_cfg.body_names)
    return mean_dist


def grasp_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    centering_scale: float = 0.05,
    blend_distance: float = 0.15,
    blend_sharpness: float = 25.0,
) -> torch.Tensor:
    """Reward for enclosing the object between thumb and the 4 fingers.

    enclosure : object centre is at the midpoint of the grasp axis  (t ≈ 0.5)
    centering : object centre is close to the grasp axis, normalised by centering_scale

    centering_scale : perpendicular distance at which centering = 0  (default 0.05 m)
      prevents high reward when the hand is far away but geometrically aligned

    Gated by the same proximity weight as reach/approach_angle so the reward
    is only active when the hand is actually near the object.

    Returns (N,) in [0, 1].
    """
    robot = env.scene[robot_cfg.name]
    obj   = env.scene[object_cfg.name]

    thumb_ids,  _ = robot.find_bodies(["R_th_tip"])
    finger_ids, _ = robot.find_bodies(["R_ff_tip", "R_mf_tip", "R_rf_tip", "R_lf_tip"])
    thumb_pos  = robot.data.body_pos_w[:, thumb_ids[0], :]            # (N, 3)
    finger_pos = robot.data.body_pos_w[:, finger_ids, :].mean(dim=1)  # (N, 3)
    obj_pos    = obj.data.root_pos_w                                   # (N, 3)

    # Grasp axis: thumb → finger centroid
    grasp_vec = finger_pos - thumb_pos
    grasp_len = torch.norm(grasp_vec, dim=-1, keepdim=True).clamp(min=1e-6)
    grasp_dir = grasp_vec / grasp_len

    # t ∈ [0, 1]: projection of object onto the grasp axis
    #   t=0 → at thumb,  t=0.5 → centred,  t=1 → at finger centroid
    t = ((obj_pos - thumb_pos) * grasp_dir).sum(dim=-1) / grasp_len.squeeze(-1)
    t = t.clamp(0.0, 1.0)

    # Enclosure: tent function, peaks at t=0.5
    enclosure = 1.0 - (2.0 * t - 1.0).abs()

    # Centering: normalised by centering_scale so reward reaches 0 at ~centering_scale metres
    # (1 - perp_dist/scale).clamp(0) → 1.0 when on axis, 0.0 when perp_dist >= centering_scale
    closest   = thumb_pos + t.unsqueeze(-1) * grasp_vec
    perp_dist = torch.norm(obj_pos - closest, dim=-1)
    centering = (1.0 - perp_dist / centering_scale).clamp(min=0.0)

    # Proximity gate: suppress reward when the hand is not yet close to the object
    proximity = _proximity(
        _mean_fingertip_dist(robot, obj_pos, _FINGERTIP_BODIES),
        blend_distance, blend_sharpness,
    )

    result = enclosure * centering * proximity
    _update_grasp_running_mean(env, result)
    if _dbv.is_debug_vis_enabled():
        _dbv.draw_grasp_frames(env, robot_cfg, object_cfg)
        _dbv.draw_approach_angle_frames(env, robot_cfg, object_cfg)
    return result


def finger_close_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    contact_activation: float = 3.0,
) -> torch.Tensor:
    """Penalise the thumb-to-finger-centroid distance when enough fingers touch the object.

    Inactive until the running mean of fingertip_contact_reward exceeds contact_activation.
    Applied with a negative weight: smaller dist → less penalty → more reward.

    Returns (N,) >= 0.
    """
    if getattr(env, "_contact_running_mean", 0.0) < contact_activation:
        return torch.zeros(env.num_envs, device=env.device)

    robot = env.scene[robot_cfg.name]

    thumb_ids,  _ = robot.find_bodies(["R_th_tip"])
    finger_ids, _ = robot.find_bodies(["R_ff_tip", "R_mf_tip", "R_rf_tip", "R_lf_tip"])

    thumb_pos  = robot.data.body_pos_w[:, thumb_ids[0], :]            # (N, 3)
    finger_pos = robot.data.body_pos_w[:, finger_ids, :].mean(dim=1)  # (N, 3)

    return torch.norm(finger_pos - thumb_pos, dim=-1)                  # (N,)


def approach_angle_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    blend_distance: float = 0.15,
    blend_sharpness: float = 25.0,
) -> torch.Tensor:
    """Reward for keeping the hand X axis perpendicular to the object's long axis.

    alignment = 1 - |dot(hand_X, long_axis)|
      1.0 → hand_X is perpendicular to the long axis  (good)
      0.0 → hand_X is parallel to the long axis       (bad)

    Faded in as the hand gets close — complement of reach_reward's fade-out.
    Use identical blend_distance / blend_sharpness in both reward terms.

    Returns (N,) in [0, 1].
    """
    robot = env.scene[robot_cfg.name]
    obj   = env.scene[object_cfg.name]

    if not hasattr(env, "object_bbox_dims"):
        return torch.zeros(env.num_envs, device=env.device)

    # Hand X axis (approach direction) from R_hand_base orientation
    hand_ids, _ = robot.find_bodies(robot_cfg.body_names)
    hand_x = matrix_from_quat(robot.data.body_quat_w[:, hand_ids[0], :])[..., 0]  

    # Object long axis: whichever of local X or Z is longer per-env
    obj_mat   = matrix_from_quat(obj.data.root_quat_w)                          
    x_longer  = env.object_bbox_dims[:, 0] > env.object_bbox_dims[:, 2]          
    long_axis = torch.where(x_longer.unsqueeze(-1), obj_mat[..., 0], obj_mat[..., 2])

    alignment = 1.0 - (hand_x * long_axis).sum(dim=-1).abs()    
    proximity = _proximity(
        _mean_fingertip_dist(robot, obj.data.root_pos_w, _FINGERTIP_BODIES),
        blend_distance, blend_sharpness,
    )

    _dbv.draw_approach_angle_frames(env, robot_cfg, object_cfg)
    return alignment * proximity


def object_vel(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalise high object velocity (discourages knocking the object over)."""
    obj = env.scene[object_cfg.name]
    return torch.norm(obj.data.root_lin_vel_w, dim=-1)


def lift_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
    contact_activation: float = 3.0,
) -> torch.Tensor:
    """Returns -1 while the object is in contact with the table, 0 when lifted.

    Inactive (returns 0) until the running mean of fingertip_contact_reward exceeds
    contact_activation. This prevents penalising the agent for not lifting before
    it has learned to grasp.
    """
    if getattr(env, "_contact_running_mean", 0.0) < contact_activation:
        return torch.zeros(env.num_envs, device=env.device)

    contact_sensor = env.scene[sensor_cfg.name]
    force_mag = torch.norm(contact_sensor.data.net_forces_w[:, 0, :], dim=-1)
    return -(force_mag > threshold).float()


def fingertip_contact_reward(
    env: ManagerBasedRLEnv,
    sensor_names: list[str],
    threshold: float = 0.5,
) -> torch.Tensor:
    """Number of fingertips in contact with the object. (N,) in [0, 5].

    Each sensor covers one fingertip filtered to the object only (1:1 prim ratio).
    threshold : minimum net force magnitude [N] to count as contact.
    """
    contacts = []
    for name in sensor_names:
        sensor = env.scene[name]
        force_mag = torch.norm(sensor.data.net_forces_w[:, 0, :], dim=-1) 
        contacts.append((force_mag > threshold).float())
    result = torch.stack(contacts, dim=-1).sum(dim=-1) 
    _update_contact_running_mean(env, result)
    return result
