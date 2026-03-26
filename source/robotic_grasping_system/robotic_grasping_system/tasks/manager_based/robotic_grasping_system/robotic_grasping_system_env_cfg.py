# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp

##
# defining both robot and actuator configs in the same file for simplicity
##

YCB_OBJECTS = {
    "cracker_box": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
    "sugar_box": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
    "tomato_soup_can": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
    "mustard_bottle": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
    # "foam_brick": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/061_foam_brick.usd",
}

# -- Approximate bounding box dimensions [L, W, H] in meters for each object
YCB_BBOX = {
    "cracker_box": [0.16, 0.06, 0.21],
    "sugar_box": [0.09, 0.04, 0.18],
    "tomato_soup_can": [0.07, 0.07, 0.10],
    "mustard_bottle": [0.06, 0.06, 0.19],
    # "foam_brick":  [0.06, 0.06, 0.19],
}

#TODO: Update the USD path to relative path
# -- Robot USD path
VEGA_USD = "/home/rainier/Downloads/dexmate_assignment/vega_upper_body-vega_1/vega_upper_body.usd"

# -- Right arm + hand joint names
RIGHT_ARM_JOINTS = [f"R_arm_j{i}" for i in range(1, 8)]  # 7 DOF
RIGHT_HAND_JOINTS = [
    "R_ff_j1", "R_ff_j2",   # fore finger
    "R_mf_j1", "R_mf_j2",   # middle finger
    "R_rf_j1", "R_rf_j2",   # ring finger
    "R_lf_j1", "R_lf_j2",   # little finger
    "R_th_j0", "R_th_j1", "R_th_j2",  # thumb
]
RIGHT_JOINTS = RIGHT_ARM_JOINTS + RIGHT_HAND_JOINTS  # 18 DOF total

# -- Fingertip body names
FINGERTIP_BODIES = ["R_ff_tip", "R_mf_tip", "R_rf_tip", "R_lf_tip", "R_th_tip"]


#TODO: fine tune these values
HAND_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=["R_arm.*"],
    stiffness= 17.4533,
    damping= 0.01745,
    # effort_limit = ,
)

ARM_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=["R_ff.*", "R_lf.*", "R_mf.*", "R_rf.*", "R_th.*"],
    stiffness= 17.4533,
    damping= 0.01745,
    # effort_limit = ,
)


VEGA_UPPER_BODY_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path = VEGA_USD,
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
    #TODO: initialize such taht left arm wont come in between
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5), # 0.3
        joint_pos={".*": 0.0},
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


    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    dome_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 0.8, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.4, 0.3, 0.2),
            ),
        ),
        #TODO: check if we can randomize here itself
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.25),
        ),
    )

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.UsdFileCfg(usd_path=YCB_OBJECTS["cracker_box"]),
                sim_utils.UsdFileCfg(usd_path=YCB_OBJECTS["sugar_box"]),
                sim_utils.UsdFileCfg(usd_path=YCB_OBJECTS["tomato_soup_can"]),
                sim_utils.UsdFileCfg(usd_path=YCB_OBJECTS["mustard_bottle"]),
            ],
            random_choice=True,
        ),
        #TODO: check if we can randomize the yaw
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.6),
            rot=(0.7071068, -0.7071068, 0.0, 0.0),
        ),
    )

    robot: ArticulationCfg = VEGA_UPPER_BODY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")



##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=RIGHT_JOINTS,
        scale=1.0,
        # use_default_offset=True,  #TODO: check this one
        )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Robot state (18 + 18 = 36)
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_JOINTS)},
            )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_JOINTS)},
            )
        # Fingertip positions (5 * 3 = 15)
        #TODO: check this one
        fingertip_pos = ObsTerm(
            func=mdp.body_pos_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=FINGERTIP_BODIES)},
        )
        # Object state (3 + 4 = 7)
        object_pos = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        object_quat = ObsTerm(
            func=mdp.root_quat_w,
            params={"asset_cfg": SceneEntityCfg("object")},
        )
        # Fingertip-to-object vectors (5 * 3 = 15)
        fingertip_to_object = ObsTerm(
            func=mdp.fingertip_to_object,
            params={
                "robot_cfg": SceneEntityCfg("robot", body_names=FINGERTIP_BODIES),
                "object_cfg": SceneEntityCfg("object"),
            },
        )

        #TODO: need to check this and implement
        # # Object identity: bbox dims + one-hot (3 + 4 = 7)
        # object_encoding = ObsTerm(
        #     func=mdp.object_encoding,
        #     params={"object_cfg": SceneEntityCfg("object")},
        # )


        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

#TODO: check all the rewards
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reach = RewTerm(
        func=mdp.reach_reward,
        weight=1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=FINGERTIP_BODIES),
            "object_cfg": SceneEntityCfg("object"),
        },
    )
    grasp = RewTerm(
        func=mdp.grasp_reward,
        weight=5.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=FINGERTIP_BODIES),
            "object_cfg": SceneEntityCfg("object"),
            "contact_threshold": 0.02,
        },
    )
    lift = RewTerm(
        func=mdp.lift_reward,
        weight=10.0,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "min_lift_height": 0.08,
        },
    )
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )
    joint_vel_penalty = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_JOINTS)},
    )

#TODO: add success condition maybe
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # TODO: check this condition
    object_dropped = DoneTerm(
        func=mdp.object_dropped,
        params={"asset_cfg": SceneEntityCfg("object"), "min_height": 0.0},
    )


@configclass
class EventCfg:
    """Configuration for events."""

    # TODO: can only reset required joints if needed
    reset_robot = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.1, 0.1),
            "velocity_range": (0.01, 0.01),
        },
    )

    #TODO: check this one
    reset_objects = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
            "velocity_range": {},
        },
    )

    # TODO: need to add this
    # reset_table = EventTerm(
    #     func=mdp.randomize_table_height,
    #     mode="reset",
    #     params={
    #         "table_cfg": SceneEntityCfg("table"),
    #         "delta_range": (-0.10, 0.10),
    #     },
    # )


##
# Environment configuration
##


@configclass
class RoboticGraspingSystemEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: RoboticGraspingSystemSceneCfg = RoboticGraspingSystemSceneCfg(
        num_envs=4096, env_spacing=2.0, replicate_physics=False
    )
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 100
        self.sim.render_interval = self.decimation
