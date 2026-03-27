"""Custom termination terms for Vega grasping environment."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_dropped(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_height: float = 0.0,
) -> torch.Tensor:
    """Terminate if object falls below the minimum height (off table / ground).

    Returns: (num_envs,) boolean tensor.
    """
    obj: RigidObject = env.scene[asset_cfg.name]
    return obj.data.root_pos_w[:, 2] < min_height


def object_lifted_success(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    hold_steps: int = 50,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Terminate (success) when object has no table contact for hold_steps consecutive steps.

    Uses the table contact sensor — if the contact force magnitude stays below
    force_threshold for hold_steps in a row, the object is considered successfully lifted.

    Args:
        sensor_cfg: Contact sensor on the table.
        hold_steps: Consecutive no-contact steps required for success.
        force_threshold: Force (N) below which contact is considered absent.

    Returns: (num_envs,) boolean tensor — True for envs that succeeded.
    """
    contact_sensor = env.scene[sensor_cfg.name]
    force = contact_sensor.data.net_forces_w[:, 0, :]
    force_mag = torch.norm(force, dim=-1)
    not_on_table = force_mag < force_threshold

    if not hasattr(env, "_lift_hold_counter"):
        env._lift_hold_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    # increment where no contact, reset where contact
    env._lift_hold_counter = torch.where(
        not_on_table, env._lift_hold_counter + 1, torch.zeros_like(env._lift_hold_counter)
    )

    # reset counter for freshly reset envs
    env._lift_hold_counter = torch.where(
        env.episode_length_buf == 0, torch.zeros_like(env._lift_hold_counter), env._lift_hold_counter
    )

    return env._lift_hold_counter >= hold_steps
