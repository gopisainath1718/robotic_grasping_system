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
    """World-frame positions of specified bodies, flattened.

    Returns: (num_envs, num_bodies * 3)
    """
    asset = env.scene[asset_cfg.name]
    body_ids, _ = asset.find_bodies(asset_cfg.body_names)
    positions = asset.data.body_pos_w[:, body_ids, :]
    return positions.reshape(env.num_envs, -1)


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object relative to the robot root.

    Returns: (num_envs, 3)
    """
    robot = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    return obj.data.root_pos_w - robot.data.root_pos_w


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
