# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp

##
# defining both robot and actuator configs in the same file for simplicity
##

HAND_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=["R_arm.*"],
    stiffness= 17453292716032.0,
    damping= 1745.32922,
    # effort_limit = ,
)

ARM_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=["R_ff.*", "R_lf.*", "R_mf.*", "R_rf.*", "R_th.*"],
    stiffness= 17453292716032.0,
    damping= 1745.32922,
    # effort_limit = ,
)


VEGA_UPPER_BODY_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        #TODO: Update the USD path to relative path
        usd_path = f"/home/rainier/Downloads/dexmate_assignment/vega_upper_body-vega_1/vega_upper_body.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6), # 0.3
        joint_pos={".*": 0.0},
        # pos=(0.0, 0.0, 0.24),
        # joint_pos={
        #        "fl_hx": 0.0,  
        #        "fr_hx": 0.0,  
        #        "hl_hx": 0.0,  
        #        "hr_hx": 0.0,  

        #        "fl_hy": 0.9,  
        #        "fr_hy": 0.9,  
        #        "hl_hy": 0.9,   
        #        "hr_hy": 0.9,

        #        "fl_kn": -0.83,  
        #        "fr_kn": -0.83,  
        #        "hl_kn": -0.83,  
        #        "hr_kn": -0.83,
        # },
        joint_vel={".*": 0.0},
    ),
    actuators={"hand": HAND_ACTUATOR_CFG, "arm" : ARM_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)


##
# Scene definition
##


@configclass
class RoboticGraspingSystemSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = VEGA_UPPER_BODY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


# @configclass
# class EventCfg:
#     """Configuration for events."""

#     # reset
#     reset_cart_position = EventTerm(
#         func=mdp.reset_joints_by_offset,
#         mode="reset",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
#             "position_range": (-1.0, 1.0),
#             "velocity_range": (-0.5, 0.5),
#         },
#     )



@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class RoboticGraspingSystemEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: RoboticGraspingSystemSceneCfg = RoboticGraspingSystemSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 500
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation