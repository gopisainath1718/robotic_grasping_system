"""Custom event terms for domain randomization.

Handles table height randomization and coordinated object placement.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def randomize_table_height(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    table_cfg: SceneEntityCfg,
    delta_range: tuple[float, float] = (-0.10, 0.10),
) -> None:
    """Randomize table height by shifting z from current world position."""
    table = env.scene[table_cfg.name]
    num_resets = len(env_ids)
    device = env.device

    delta = torch.rand(num_resets, device=device) * (delta_range[1] - delta_range[0]) + delta_range[0]

    table_quat = table.data.root_quat_w[env_ids].clone()  # (N, 4)

    # TODO: change the hardcaoded values
    env_origins = env.scene.env_origins[env_ids]  # (N, 3)
    table_pos = env_origins.clone()
    table_pos[:, 0] += 0.5   # table x offset from init_state
    table_pos[:, 1] += 0.0   # table y offset
    table_pos[:, 2] += 0.25 + delta  # init z + delta

    table.write_root_pose_to_sim(
        torch.cat([table_pos, table_quat], dim=-1),
        env_ids,
    )

    # Store delta for object placement
    if not hasattr(env, "table_z"):
        env.table_z = torch.zeros(env.num_envs, device=device)
    env.table_z[env_ids] = table_pos[:, 2]


def randomize_object_placement(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg,
    x_range: tuple[float, float] = (-0.05, 0.05),
    y_range: tuple[float, float] = (-0.05, 0.05),
) -> None:
    """Place object on table with random x, y, yaw."""
    obj = env.scene[object_cfg.name]
    num_resets = len(env_ids)
    device = env.device

    obj_quat = obj.data.root_quat_w[env_ids].clone()

    env_origins = env.scene.env_origins[env_ids]
    obj_pos = env_origins.clone()

    obj_pos[:, 0] += 0.6 + torch.rand(num_resets, device=device) * (x_range[1] - x_range[0]) + x_range[0]
    obj_pos[:, 1] += 0.0 + torch.rand(num_resets, device=device) * (y_range[1] - y_range[0]) + y_range[0]
    obj_pos[:, 2] = 0.25 + env.table_z[env_ids]

    # TODO: add random yaw
    # yaw = torch.rand(num_resets, device=device) * 2.0 * 3.14159 - 3.14159
    # quat = torch.zeros(num_resets, 4, device=device)
    # quat[:, 0] = torch.cos(yaw / 2.0)
    # quat[:, 3] = torch.sin(yaw / 2.0)

    obj.write_root_pose_to_sim(torch.cat([obj_pos, obj_quat], dim=-1), env_ids)

    # Store for lift reward
    if not hasattr(env, "initial_object_z"):
        env.initial_object_z = torch.zeros(env.num_envs, device=device)
    env.initial_object_z[env_ids] = obj_pos[:, 2]