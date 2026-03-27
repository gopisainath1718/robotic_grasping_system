"""Reward terms: reach -> grasp -> lift.

Based on CrossDex (ICLR 2025) and Dexterous Functional Grasping (CoRL 2023).
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reach_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Exponential reward for fingertip proximity to object.

    Returns: (num_envs,) in [0, 1]
    """
    robot = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    body_ids, _ = robot.find_bodies(robot_cfg.body_names)
    fingertip_pos = robot.data.body_pos_w[:, body_ids, :]
    obj_pos = obj.data.root_pos_w.unsqueeze(1)

    distances = torch.norm(fingertip_pos - obj_pos, dim=-1)
    mean_dist = distances.mean(dim=-1)

    return mean_dist


def grasp_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Grasp reward for dexterous hand.
    
    Checks: Thumb opposes other fingers (squeeze)
    """
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]

    # 1. Thumb-finger opposition
    thumb_ids, _ = robot.find_bodies(["R_th_tip"])
    finger_ids, _ = robot.find_bodies(["R_ff_tip", "R_mf_tip", "R_rf_tip", "R_lf_tip"])
    thumb_pos = robot.data.body_pos_w[:, thumb_ids[0], :]
    other_pos = robot.data.body_pos_w[:, finger_ids, :].mean(dim=1)
    thumb_finger_dist = torch.norm(thumb_pos - other_pos, dim=-1)

    # 2. object mid point
    midpoint = (thumb_pos + other_pos) / 2.0
    obj_to_mid = torch.norm(obj.data.root_pos_w - midpoint, dim=-1)

    reward = obj_to_mid + thumb_finger_dist
    return reward


def object_vel(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    obj = env.scene[object_cfg.name]
    return torch.norm(obj.data.root_lin_vel_w, dim=-1)


def lift_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: int = 1.0
) -> torch.Tensor:
    """Negative reward if object is in contact with table.
    
    Returns: 0.0 if lifted, -1.0 if on table
    """
    contact_sensor = env.scene[sensor_cfg.name]
    force = contact_sensor.data.net_forces_w[:, 0, :]
    force_mag = torch.norm(force, dim=-1)

    on_table = (force_mag > threshold).float()
    return -on_table
