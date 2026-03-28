from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def body_pos_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Positions of specified bodies relative to the environment origin, flattened.

    Mirrors isaaclab's body_pose_w but returns positions only (no quaternion).

    Returns: (num_envs, num_bodies * 3)
    """
    asset = env.scene[asset_cfg.name]
    body_ids, _ = asset.find_bodies(asset_cfg.body_names)
    positions_w = asset.data.body_pos_w[:, body_ids, :]          # (N, B, 3)
    positions_env = positions_w - env.scene.env_origins.unsqueeze(1)  # (N, B, 3)
    return positions_env.reshape(env.num_envs, -1)


def fingertip_to_object(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Vectors from each fingertip to object center, flattened.

    Returns: (num_envs, num_fingertips * 3)
    """
    robot = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    body_ids, _ = robot.find_bodies(robot_cfg.body_names)
    fingertip_pos = robot.data.body_pos_w[:, body_ids, :]  # (N, 5, 3)
    obj_pos = obj.data.root_pos_w.unsqueeze(1)  # (N, 1, 3)

    diff = obj_pos - fingertip_pos  # (N, 5, 3)
    return diff.reshape(env.num_envs, -1)


def object_bbox_dims(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Returns per-env object bounding box dimensions (local space) as observation.

    Returns: (num_envs, 3) — [x_size, y_size, z_size] in local object frame
    """
    if not hasattr(env, "object_bbox_dims"):
        return torch.zeros(env.num_envs, 3, device=env.device)
    return env.object_bbox_dims


def object_encoding(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
        
    obj: RigidObject = env.scene[object_cfg.name]
