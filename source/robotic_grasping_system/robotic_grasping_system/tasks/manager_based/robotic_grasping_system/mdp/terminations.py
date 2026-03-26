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
