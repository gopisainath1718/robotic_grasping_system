"""Bayesian Optimisation of arm or hand joint PD gains (stiffness & damping).

Usage (inside the Isaac Lab environment):
    python optimization/BO.py --mode hand --headless
    python optimization/BO.py --mode arm  --headless --num_envs 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="BO for arm/hand PD gains")
parser.add_argument("--mode", type=str, choices=["arm", "hand"], required=True,
                    help="which joint group to optimise: 'arm' or 'hand'")
parser.add_argument("--num_envs", type=int, default=4,
                    help="parallel robot instances (metrics averaged across them)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
MODE = args.mode
launcher = AppLauncher(args)
simulation_app = launcher.app

# -- Post-launch imports -----------------------------------------------
import json
import numpy as np
import torch
import optuna

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass


##
# Parameters
##

VEGA_USD = str(Path(__file__).resolve().parent.parent / "vega_upper_body-vega_1" / "vega_upper_body.usd")
ARM_JOINTS = [f"R_arm_j{i}" for i in range(1, 8)]

HAND_JOINTS = [
    "R_ff_j1",              # [-1.0946,  0.2891]
    "R_mf_j1",              # [-1.0844,  0.2801]
    "R_rf_j1",              # [-1.0154,  0.2840]
    "R_lf_j1",              # [-1.0118,  0.2811]
    "R_th_j0",              # [-0.0158,  1.6050]
    "R_th_j1",              # [-0.3468,  0.1834]
]

HAND_J2_JOINTS = [
    "R_ff_j2",              # [-1.5500,  0.6396]
    "R_mf_j2",              # [-1.5380,  0.6266]
    "R_rf_j2",              # [-1.4402,  0.6142]
    "R_lf_j2",              # [-1.4614,  0.6208]
    "R_th_j2",              # [-0.6128,  0.3917]
]

NUM_STEPS  = 1000
NUM_TRIALS = 100
SIM_DT     = 1.0 / 200.0


if MODE == "arm":
    OPT_JOINTS   = ARM_JOINTS
    K_BOUNDS     = (20.0, 300.0) 
    D_BOUNDS     = (1.0,  50.0)   
    TARGET_RAD   = [-1.5708, 0.0, 0.0, -2.35619, 0.0, 0.0, 0.0]
    TRACK_SECONDARY = False    
else:  
    OPT_JOINTS   = HAND_JOINTS
    K_BOUNDS     = (1.0,  50.0)
    D_BOUNDS     = (0.1,  10.0)
    TARGET_RAD   = [-0.40, -0.40, -0.37, -0.37, 0.79, -0.08]
    TRACK_SECONDARY = True       

RESULTS_DIR = Path(__file__).resolve().parent.parent / "bo_results"
RESULTS_DIR.mkdir(exist_ok=True)
LOG_PATH  = RESULTS_DIR / f"bo_log_{MODE}.jsonl"
BEST_PATH = RESULTS_DIR / f"best_gains_{MODE}.json"


@configclass
class GainTuningSceneCfg(InteractiveSceneCfg):

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(10.0, 10.0)),
    )

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=VEGA_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["R_arm.*"],
                stiffness=100.0,    # placeholder if arm mode — overridden each trial
                damping=10.86,
                effort_limit={
                    "R_arm_j1": 150, "R_arm_j2": 150, "R_arm_j3": 80,
                    "R_arm_j4": 80,  "R_arm_j5": 40,  "R_arm_j6": 40,
                    "R_arm_j7": 25,
                },
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=HAND_JOINTS,
                stiffness=48.0,    # placeholder if hand mode — overridden each trial
                damping=1.0,
            ),
        },
    )


sim = SimulationContext(SimulationCfg(dt=SIM_DT))
scene = InteractiveScene(
    GainTuningSceneCfg(num_envs=args.num_envs, env_spacing=2.5, replicate_physics=False)
)
sim.reset()

robot         = scene["robot"]
device        = sim.device
num_envs      = args.num_envs
opt_ids,  _   = robot.find_joints(OPT_JOINTS)
num_opt       = len(opt_ids)
if TRACK_SECONDARY:
    sec_ids, _ = robot.find_joints(HAND_J2_JOINTS)

target_pos     = torch.tensor(TARGET_RAD, device=device)
_default_root  = robot.data.default_root_state.clone()       
_default_jpos  = robot.data.default_joint_pos.clone()             
_default_jvel  = torch.zeros_like(robot.data.default_joint_vel) 


def _set_gains(k: float, d: float):
    """Write PD gains to the PhysX joint drives for the optimised joints."""
    robot.write_joint_stiffness_to_sim(
        torch.full((num_envs, num_opt), k, device=device), joint_ids=opt_ids)
    robot.write_joint_damping_to_sim(
        torch.full((num_envs, num_opt), d, device=device), joint_ids=opt_ids)


def _reset():
    """Reset robot to default root pose and zero joint state."""
    ids = torch.arange(num_envs, device=device)
    robot.write_root_state_to_sim(_default_root)
    robot.write_joint_state_to_sim(_default_jpos, _default_jvel)
    robot.reset(ids)


def _step():
    """One physics step: write commands -> step PhysX -> read state."""
    scene.write_data_to_sim()
    sim.step()
    scene.update(SIM_DT)


if TRACK_SECONDARY:
    PENALTY = dict(ss_error=10.0, ss_vel_j1=10.0, ss_vel_j2=10.0, mean_error=10.0)
else:
    PENALTY = dict(ss_error=10.0, ss_vel=10.0, mean_error=10.0)


def evaluate(k: float, d: float) -> dict:
    """Run NUM_STEPS with given gains and return tracking metrics."""
    _set_gains(k, d)
    _reset()

    for _ in range(20):
        _step()

    robot.set_joint_position_target(
        target_pos.expand(num_envs, -1), joint_ids=opt_ids)

    errors, vels = [], []
    vels_sec = [] if TRACK_SECONDARY else None
    for step_i in range(NUM_STEPS):
        _step()
        pos = robot.data.joint_pos[:, opt_ids]
        vel = robot.data.joint_vel[:, opt_ids]

        nan_check = torch.isnan(pos).any() or torch.isnan(vel).any()
        if TRACK_SECONDARY:
            vel_s = robot.data.joint_vel[:, sec_ids]
            nan_check = nan_check or torch.isnan(vel_s).any()

        if nan_check:
            print(f"  [!] NaN detected at step {step_i} — unstable gains, returning penalty")
            return PENALTY

        errors.append((pos - target_pos).abs().mean().item())
        vels.append(vel.abs().mean().item())
        if TRACK_SECONDARY:
            vels_sec.append(vel_s.abs().mean().item())

    ss = int(0.8 * NUM_STEPS)
    if TRACK_SECONDARY:
        return dict(
            ss_error   = float(np.mean(errors[ss:])),
            ss_vel_j1  = float(np.mean(vels[ss:])),
            ss_vel_j2  = float(np.mean(vels_sec[ss:])),
            mean_error = float(np.mean(errors)),
        )
    else:
        return dict(
            ss_error   = float(np.mean(errors[ss:])),
            ss_vel     = float(np.mean(vels[ss:])),
            mean_error = float(np.mean(errors)),
        )


def objective(trial: optuna.Trial) -> float:
    k = trial.suggest_float("stiffness", *K_BOUNDS)
    d = trial.suggest_float("damping",   *D_BOUNDS)

    m = evaluate(k, d)

    gain_reg = 0.05 * (k / K_BOUNDS[1] + d / D_BOUNDS[1])
    if TRACK_SECONDARY:
        cost = m["ss_error"] + 0.5 * m["ss_vel_j1"] + 0.5 * m["ss_vel_j2"] + gain_reg
        vel_str = f"ss_vel_j1={m['ss_vel_j1']:.4f}  ss_vel_j2={m['ss_vel_j2']:.4f}"
    else:
        cost = m["ss_error"] + 0.5 * m["ss_vel"] + gain_reg
        vel_str = f"ss_vel={m['ss_vel']:.4f}"

    record = dict(trial=trial.number, stiffness=round(k, 2),
                  damping=round(d, 2), **m, cost=round(cost, 6))
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")

    print(f"[{trial.number:3d}]  K={k:6.2f}  D={d:5.2f}  "
          f"ss_err={m['ss_error']:.4f}  {vel_str}  cost={cost:.4f}")
    return cost


def main():
    LOG_PATH.write_text("") 

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name=f"{MODE}_gains",
    )
    study.optimize(objective, n_trials=NUM_TRIALS)

    best = study.best_trial
    result = dict(
        stiffness=round(best.params["stiffness"], 2),
        damping=round(best.params["damping"], 2),
        cost=round(best.value, 6),
    )
    BEST_PATH.write_text(json.dumps(result, indent=2) + "\n")

    print(f"\n{'=' * 50}")
    print(f"Best  K={result['stiffness']}  D={result['damping']}  cost={result['cost']}")
    print(f"Saved -> {BEST_PATH}")

    simulation_app.close()


if __name__ == "__main__":
    main()
