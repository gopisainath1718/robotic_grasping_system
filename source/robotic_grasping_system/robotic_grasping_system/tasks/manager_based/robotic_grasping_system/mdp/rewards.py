from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_FINGERTIP_BODIES = ["R_ff_tip", "R_mf_tip", "R_rf_tip", "R_lf_tip", "R_th_tip"]

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _mean_fingertip_dist(
    robot,
    obj_pos_w: torch.Tensor,
    body_names: list[str]
    ) -> torch.Tensor:
    """Mean Euclidean distance from named fingertips to the object centre.
    """
    ids, _ = robot.find_bodies(body_names)
    ft_pos = robot.data.body_pos_w[:, ids, :]      
    return torch.norm(ft_pos - obj_pos_w.unsqueeze(1), dim=-1).mean(dim=-1)


def _update_contact_running_mean(
    env: "ManagerBasedRLEnv",
    contact_vals: torch.Tensor,
    window: int = 2000
    ) -> None:
    """Circular buffer of per-step mean fingertip contact count.

    Stores _contact_running_mean - used by finger_close_reward, lift_reward and lift_vel_reward
    to decide when enough fingers are touching the object.
    """
    if not hasattr(env, "_contact_mean_buf"):
        env._contact_mean_buf = torch.zeros(window, device=env.device)
        env._contact_mean_idx = 0
    env._contact_mean_buf[env._contact_mean_idx] = contact_vals.mean().detach()
    env._contact_mean_idx = (env._contact_mean_idx + 1) % window
    env._contact_running_mean = env._contact_mean_buf.mean().item()


def _update_grasp_running_mean(
    env: "ManagerBasedRLEnv",
    grasp_vals: torch.Tensor,
    window: int = 2000
) -> None:
    """Circular buffer of per-step mean grasp reward.

    Stores _grasp_running_mean - used by lift_reward to decide
    when grasping is consistent enough to start training lift.
    """
    if not hasattr(env, "_grasp_mean_buf"):
        env._grasp_mean_buf = torch.zeros(window, device=env.device)
        env._grasp_mean_idx = 0
    env._grasp_mean_buf[env._grasp_mean_idx] = grasp_vals.mean().detach()
    env._grasp_mean_idx = (env._grasp_mean_idx + 1) % window
    env._grasp_running_mean = env._grasp_mean_buf.mean().item()


def _proximity(
    dist: torch.Tensor,
    blend_distance: float,
    blend_sharpness: float
    ) -> torch.Tensor:
    """Sigmoid weight: ~0 when far, ~1 when close.
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
    """Mean fingertip-to-object distance.
    """
    robot = env.scene[robot_cfg.name]
    obj   = env.scene[object_cfg.name]

    mean_dist = _mean_fingertip_dist(robot, obj.data.root_pos_w, robot_cfg.body_names)
    return mean_dist


def finger_close_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    contact_activation: float = 3.0,
) -> torch.Tensor:
    """Reward closeness between thumb and finger centroid when enough fingers touch the object.

    Inactive until the running mean of fingertip_contact_reward exceeds contact_activation.
    Returns 1/(1 + dist): closer fingers means higher reward.
    """
    if getattr(env, "_contact_running_mean", 0.0) < contact_activation:
        return torch.zeros(env.num_envs, device=env.device)

    robot = env.scene[robot_cfg.name]

    thumb_ids,  _ = robot.find_bodies(["R_th_tip"])
    finger_ids, _ = robot.find_bodies(["R_ff_tip", "R_mf_tip", "R_rf_tip", "R_lf_tip"])

    thumb_pos  = robot.data.body_pos_w[:, thumb_ids[0], :]          
    finger_pos = robot.data.body_pos_w[:, finger_ids, :].mean(dim=1)

    dist = torch.norm(finger_pos - thumb_pos, dim=-1)          
    return 1.0 / (1.0 + dist)


def object_lateral_vel(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalise lateral object velocity: discourages knocking/sliding."""
    obj = env.scene[object_cfg.name]
    return torch.norm(obj.data.root_lin_vel_w[:, :2], dim=-1)



def object_lift_vel(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    contact_activation: float = 3.5,
) -> torch.Tensor:
    """Reward upward object velocity. Gated by contact_running_mean like lift_reward."""
    if getattr(env, "_contact_running_mean", 0.0) < contact_activation:
        return torch.zeros(env.num_envs, device=env.device)

    obj = env.scene[object_cfg.name]
    return obj.data.root_lin_vel_w[:, 2].clamp(min=0.0)


def lift_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
    contact_activation: float = 3.0,
) -> torch.Tensor:
    """Returns +1 when the object is not in contact with the table, 0 otherwise.

    Inactive until the running mean of fingertip_contact_reward exceeds
    contact_activation. This prevents rewarding the agent for lifting before
    it has learned to grasp.
    """
    if getattr(env, "_contact_running_mean", 0.0) < contact_activation:
        return torch.zeros(env.num_envs, device=env.device)

    contact_sensor = env.scene[sensor_cfg.name]
    force_mag = torch.norm(contact_sensor.data.net_forces_w[:, 0, :], dim=-1)
    return (force_mag <= threshold).float()


def fingertip_contact_reward(
    env: ManagerBasedRLEnv,
    sensor_names: list[str],
    threshold: float = 0.5,
) -> torch.Tensor:
    """Number of fingertips in contact with the object.
    """
    contacts = []
    for name in sensor_names:
        sensor = env.scene[name]
        force_mag = torch.norm(sensor.data.force_matrix_w[:, 0, 0, :], dim=-1)
        contacts.append((force_mag > threshold).float())
    result = torch.stack(contacts, dim=-1).sum(dim=-1) 
    _update_contact_running_mean(env, result)
    return result

# ---------------------------------------------------------------------------
# Currently unused
# ---------------------------------------------------------------------------

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
    """
    robot = env.scene[robot_cfg.name]
    obj   = env.scene[object_cfg.name]

    thumb_ids,  _ = robot.find_bodies(["R_th_tip"])
    finger_ids, _ = robot.find_bodies(["R_ff_tip", "R_mf_tip", "R_rf_tip", "R_lf_tip"])
    thumb_pos  = robot.data.body_pos_w[:, thumb_ids[0], :]           
    finger_pos = robot.data.body_pos_w[:, finger_ids, :].mean(dim=1) 
    obj_pos    = obj.data.root_pos_w                                 

    grasp_vec = finger_pos - thumb_pos
    grasp_len = torch.norm(grasp_vec, dim=-1, keepdim=True).clamp(min=1e-6)
    grasp_dir = grasp_vec / grasp_len

    t = ((obj_pos - thumb_pos) * grasp_dir).sum(dim=-1) / grasp_len.squeeze(-1)
    t = t.clamp(0.0, 1.0)


    enclosure = 1.0 - (2.0 * t - 1.0).abs()

    closest   = thumb_pos + t.unsqueeze(-1) * grasp_vec
    perp_dist = torch.norm(obj_pos - closest, dim=-1)
    centering = (1.0 - perp_dist / centering_scale).clamp(min=0.0)

    proximity = _proximity(
        _mean_fingertip_dist(robot, obj_pos, _FINGERTIP_BODIES),
        blend_distance, blend_sharpness,
    )

    result = enclosure * centering * proximity
    _update_grasp_running_mean(env, result)
    return result

def approach_angle_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    blend_distance: float = 0.15,
    blend_sharpness: float = 25.0,
) -> torch.Tensor:
    """Reward for keeping the hand X axis perpendicular to the object's long axis.

    Faded in as the hand gets close — complement of reach_reward's fade-out.
    Use identical blend_distance / blend_sharpness in both reward terms.
    """
    robot = env.scene[robot_cfg.name]
    obj   = env.scene[object_cfg.name]

    if not hasattr(env, "object_bbox_dims"):
        return torch.zeros(env.num_envs, device=env.device)

    hand_ids, _ = robot.find_bodies(robot_cfg.body_names)
    hand_x = matrix_from_quat(robot.data.body_quat_w[:, hand_ids[0], :])[..., 0]  

    obj_mat   = matrix_from_quat(obj.data.root_quat_w)                          
    x_longer  = env.object_bbox_dims[:, 0] > env.object_bbox_dims[:, 2]          
    long_axis = torch.where(x_longer.unsqueeze(-1), obj_mat[..., 0], obj_mat[..., 2])

    alignment = 1.0 - (hand_x * long_axis).sum(dim=-1).abs()    
    proximity = _proximity(
        _mean_fingertip_dist(robot, obj.data.root_pos_w, _FINGERTIP_BODIES),
        blend_distance, blend_sharpness,
    )

    return alignment * proximity