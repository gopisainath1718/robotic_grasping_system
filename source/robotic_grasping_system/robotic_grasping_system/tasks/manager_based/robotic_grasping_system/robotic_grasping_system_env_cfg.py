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
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg

import os as _os

from . import mdp

##
# defining both robot and actuator configs in the same file for simplicity
##

_REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), *[".."] * 6))

YCB_OBJECTS = {
    "mug": _os.path.join(_REPO_ROOT, "ycb_physics", "025_mug.usd"),
    "sugar_box": _os.path.join(_REPO_ROOT, "ycb_physics", "sugar_box.usd"),
    "tomato_soup_can": _os.path.join(_REPO_ROOT, "ycb_physics", "tomato_soup_can.usd"),
    "banana": _os.path.join(_REPO_ROOT, "ycb_physics", "011_banana.usd"),
    "mustard_bottle": _os.path.join(_REPO_ROOT, "ycb_physics", "mustard_bottle.usd"),
}

VEGA_USD = _os.path.join(_REPO_ROOT, "vega_upper_body-vega_1", "vega_upper_body.usd")


RIGHT_ARM_JOINTS = [f"R_arm_j{i}" for i in range(1, 8)]
RIGHT_HAND_JOINTS = [
    "R_ff_j1", "R_ff_j2",               # fore finger     
    "R_mf_j1", "R_mf_j2",               # middle finger   
    "R_rf_j1", "R_rf_j2",               # ring finger   
    "R_lf_j1", "R_lf_j2",               # little finger   
    "R_th_j0", "R_th_j1", "R_th_j2",    # thumb   
]

RIGHT_JOINTS = RIGHT_ARM_JOINTS + RIGHT_HAND_JOINTS

FINGERTIP_BODIES = ["R_ff_l1", "R_mf_l1", "R_rf_l1", "R_lf_l1", "R_th_l1"]

ARM_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=["R_arm.*"],
    stiffness= 100,
    damping= 10.86,
    effort_limit={
        "R_arm_j1": 150,
        "R_arm_j2": 150,
        "R_arm_j3": 80,
        "R_arm_j4": 80,
        "R_arm_j5": 40,
        "R_arm_j6": 40,
        "R_arm_j7": 25
    },
)

HAND_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr = RIGHT_HAND_JOINTS,
    stiffness= 27.08,
    damping= 0.1,
)

# adding high params to make head fixed
HEAD_ACTUATOR_CFG = ImplicitActuatorCfg(
                joint_names_expr=["head_j.*"],
                stiffness=100.0,
                damping=10.0,
                effort_limit = 50,
                armature = 0.2
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
    
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos = {
            "R_arm_j1": -1.5708,   # -90 deg
            "R_arm_j2": 0.0,
            "R_arm_j3": 0.0,
            "R_arm_j4": -2.35619,   # -135 deg
            "R_arm_j5": 0.0,
            "R_arm_j6": 0.0,
            "R_arm_j7": 0.0,

            "L_arm_j1": 1.5708,   # 90 deg
            "L_arm_j2": 0.0,
            "L_arm_j3": 0.0,
            "L_arm_j4": 0.0,   
            "L_arm_j5": 0.0,
            "L_arm_j6": 0.0,
            "L_arm_j7": 0.0,

            "head_j1" : 0.0,
            "head_j2" : 0.0,
            "head_j3" : 0.0
        },
        joint_vel={".*": 0.0},
    ),
    actuators={"hand": HAND_ACTUATOR_CFG, "arm" : ARM_ACTUATOR_CFG,
                "head" : HEAD_ACTUATOR_CFG
                },
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
                diffuse_color=(0., 0.3, 0.2),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.25),
        ),
    )

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.UsdFileCfg(usd_path=YCB_OBJECTS["sugar_box"]),
                sim_utils.UsdFileCfg(usd_path=YCB_OBJECTS["mug"]),
                sim_utils.UsdFileCfg(usd_path=YCB_OBJECTS["banana"]),
                sim_utils.UsdFileCfg(usd_path=YCB_OBJECTS["tomato_soup_can"]),
                sim_utils.UsdFileCfg(usd_path=YCB_OBJECTS["mustard_bottle"]),
            ],
            random_choice=True,
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.55),
            rot=(0.7071068, -0.7071068, 0.0, 0.0),
        ),
    )

    table_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Table"],
    )

    ff_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/R_ff_l2",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )
    mf_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/R_lf_l2",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )
    rf_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/R_mf_l2",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )
    lf_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/R_rf_l2",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )
    th_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/R_th_l2",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )

    robot: ArticulationCfg = VEGA_UPPER_BODY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")



##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_position = mdp.JointPositionActionCfg(asset_name="robot", joint_names=RIGHT_JOINTS, scale=1.0)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
    
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_JOINTS)})
        
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_JOINTS)})
        
        object_pose = ObsTerm(
            func=mdp.body_pose_w,
            params={"asset_cfg": SceneEntityCfg("object")})
        
        #TODO: change this from geometry to depth points
        object_bbox = ObsTerm(func=mdp.object_bbox_dims)

        fingertip_poses = ObsTerm(
            func=mdp.body_pos_w,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=FINGERTIP_BODIES)})
        
        last_actions = ObsTerm(func=mdp.last_action)
        
        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reach = RewTerm(
        func=mdp.reach_reward,
        weight=-2.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=FINGERTIP_BODIES),
            "object_cfg": SceneEntityCfg("object"),
        })

    fingertip_contact = RewTerm(
        func=mdp.fingertip_contact_reward,
        weight=1.0,
        params={
            "sensor_names": ["ff_contact", "mf_contact", "rf_contact", "lf_contact", "th_contact"],
            "threshold": 0.5,
        })

    finger_close = RewTerm(
        func=mdp.finger_close_reward,
        weight=5.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "contact_activation": 3,  # unlock once mean fingertip contacts > 3
        })

    lift = RewTerm(
        func=mdp.lift_reward,
        weight=4.0,
        params={
            "sensor_cfg": SceneEntityCfg("table_contact"),
            "threshold": 1.0,
            "contact_activation": 3.5,  # unlock once mean fingertip contacts > 3.5
        })

    object_lateral_vel = RewTerm(
        func=mdp.object_lateral_vel,
        weight=-0.1,
        params={
            "object_cfg": SceneEntityCfg("object")})

    object_lift_vel = RewTerm(
        func=mdp.object_lift_vel,
        weight=1.0,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "contact_activation": 3.5,  # unlock once mean fingertip contacts > 3.5
        })
    
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.001)

    joint_vel_penalty = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_JOINTS)})
    

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_lifted = DoneTerm(
        func=mdp.object_lifted_success,
        time_out=True,
        params={
            "sensor_cfg": SceneEntityCfg("table_contact"),
            "hold_steps": 50})


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
            "velocity_range": (0.01, 0.01)})


    reset_table = EventTerm(
        func=mdp.randomize_table_height,
        mode="reset",
        params={
            "table_cfg": SceneEntityCfg("table"),
            "delta_range": (-0.1, 0.1)})

    reset_objects = EventTerm(
        func = mdp.randomize_object_placement,
        mode="reset",
        params={
            "object_cfg": SceneEntityCfg("object"),
            "x_range": (-0.1, 0.1),
            "y_range": (-0.1, 0.1)})


# @configclass
# class CurriculumCfg:



##
# Environment configuration
##


@configclass
class RoboticGraspingSystemEnvCfg(ManagerBasedRLEnvCfg):

    scene: RoboticGraspingSystemSceneCfg = RoboticGraspingSystemSceneCfg(num_envs=4096, env_spacing=2.0, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        self.decimation = 2
        self.episode_length_s = 5
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation


@configclass
class RoboticGraspingSystemEnvCfg_PLAY(RoboticGraspingSystemEnvCfg):
    """Evaluation configuration."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.episode_length_s = 10.0

        self.viewer.eye = (4.0, 0.0, 3.0)

        # # --- Disable domain randomization ---
        self.events.reset_robot.params["position_range"] = (0.0, 0.0)
        self.events.reset_robot.params["velocity_range"] = (0.0, 0.0)