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
    fingertip_pos = robot.data.body_pos_w[:, body_ids, :]  # (N, 5, 3)
    obj_pos = obj.data.root_pos_w.unsqueeze(1)  # (N, 1, 3)

    distances = torch.norm(fingertip_pos - obj_pos, dim=-1)  # (N, 5)
    mean_dist = distances.mean(dim=-1)

    return torch.exp(-10.0 * mean_dist)


def grasp_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    contact_threshold: float = 0.02,
) -> torch.Tensor:
    """Reward for multi-finger proximity to object.

    Combines per-finger exponential proximity with a bonus
    when >= 3 fingers are within contact_threshold.

    Returns: (num_envs,)
    """
    robot = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    body_ids, _ = robot.find_bodies(robot_cfg.body_names)
    fingertip_pos = robot.data.body_pos_w[:, body_ids, :]  # (N, 5, 3)
    obj_pos = obj.data.root_pos_w.unsqueeze(1)  # (N, 1, 3)

    distances = torch.norm(fingertip_pos - obj_pos, dim=-1)  # (N, 5)

    per_finger = torch.exp(-50.0 * distances)
    num_contacting = (distances < contact_threshold).float().sum(dim=-1)

    reward = per_finger.sum(dim=-1) / 5.0
    multi_bonus = (num_contacting >= 3).float() * 0.5

    return reward + multi_bonus


def lift_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    min_lift_height: float = 0.08,
) -> torch.Tensor:
    """Reward for lifting object above initial height.

    Continuous reward proportional to height gained + bonus for clearing threshold.

    Returns: (num_envs,)
    """
    obj: RigidObject = env.scene[object_cfg.name]
    obj_z = obj.data.root_pos_w[:, 2]  # (N,)

    if hasattr(env, "initial_object_z"):
        ref_z = env.initial_object_z
    else:
        ref_z = 0.6

    height_gained = torch.clamp(obj_z - ref_z, min=0.0)
    continuous = torch.clamp(height_gained / min_lift_height, max=1.0)
    bonus = (height_gained > min_lift_height).float()

    return continuous + bonus
