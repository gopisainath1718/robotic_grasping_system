"""Bayesian Optimisation of hand-joint PD gains (stiffness & damping).

Spawns the Vega upper-body robot in a minimal Isaac Sim scene, commands
fixed hand-joint targets for 1000 physics steps per trial, and uses Optuna
(TPE sampler) to minimise a composite cost of steady-state angle error,
residual joint velocity, and gain magnitude (prefer smallest gains that
still track well).

Usage (inside the Isaac Lab environment):
    python optimization/BO.py --headless
    python optimization/BO.py --headless --num_envs 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

# -- Isaac Sim launch (MUST run before every Omniverse import) ---------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="BO for hand PD gains")
parser.add_argument("--num_envs", type=int, default=4,
                    help="parallel robot instances (metrics averaged across them)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
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

VEGA_USD = "/home/rainier/Downloads/dexmate_assignment/vega_upper_body-vega_1/vega_upper_body.usd"
ARM_JOINTS = [f"R_arm_j{i}" for i in range(1, 8)]   # kept fixed
# Primary joints — optimised (stiffness & damping applied here)
HAND_JOINTS = [
    "R_ff_j1",              # [-1.0946,  0.2891]
    "R_mf_j1",              # [-1.0844,  0.2801]
    "R_rf_j1",              # [-1.0154,  0.2840]
    "R_lf_j1",              # [-1.0118,  0.2811]
    "R_th_j0",              # [-0.0158,  1.6050]
    "R_th_j1",              # [-0.3468,  0.1834]
]

# Mimic joints — not actuated directly, tracked for stability only
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

K_BOUNDS = (1.0,  50.0)  # stiffness search range
D_BOUNDS = (0.1,   10.0)   # damping   search range

# Target = midpoint of each joint's range
#   R_ff_j1: (-1.0946 + 0.2891) / 2 = -0.40
#   R_mf_j1: (-1.0844 + 0.2801) / 2 = -0.40
#   R_rf_j1: (-1.0154 + 0.2840) / 2 = -0.37
#   R_lf_j1: (-1.0118 + 0.2811) / 2 = -0.37
#   R_th_j0: (-0.0158 + 1.6050) / 2 =  0.79
#   R_th_j1: (-0.3468 + 0.1834) / 2 = -0.08
TARGET_RAD = [-0.40, -0.40, -0.37, -0.37, 0.79, -0.08]

RESULTS_DIR = Path(__file__).resolve().parent.parent / "bo_results"
RESULTS_DIR.mkdir(exist_ok=True)
LOG_PATH  = RESULTS_DIR / "bo_log_hand.jsonl"
BEST_PATH = RESULTS_DIR / "best_gains_hand.json"


##
# Scene: ground plane + robot (no table, no objects)
##


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
                stiffness=100.0,
                damping=10.86,
                effort_limit={
                    "R_arm_j1": 150, "R_arm_j2": 150, "R_arm_j3": 80,
                    "R_arm_j4": 80,  "R_arm_j5": 40,  "R_arm_j6": 40,
                    "R_arm_j7": 25,
                },
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=HAND_JOINTS,
                stiffness=48.0,    # placeholder — overridden each trial
                damping=1.0,
            ),
        },
    )


##
# Simulation bootstrap
##

sim = SimulationContext(SimulationCfg(dt=SIM_DT))
scene = InteractiveScene(
    GainTuningSceneCfg(num_envs=args.num_envs, env_spacing=2.5, replicate_physics=False)
)
sim.reset()

robot         = scene["robot"]
device        = sim.device
num_envs      = args.num_envs
hand_ids,  _  = robot.find_joints(HAND_JOINTS)
hand_j2_ids, _ = robot.find_joints(HAND_J2_JOINTS)
num_hand      = len(hand_ids)

target_hand    = torch.tensor(TARGET_RAD, device=device)            # (6,)
_default_root  = robot.data.default_root_state.clone()              # (N, 13)
_default_jpos  = robot.data.default_joint_pos.clone()               # (N, J)
_default_jvel  = torch.zeros_like(robot.data.default_joint_vel)     # (N, J)


##
# Helpers
##


def _set_gains(k: float, d: float):
    """Write PD gains to the PhysX joint drives for hand joints."""
    robot.write_joint_stiffness_to_sim(
        torch.full((num_envs, num_hand), k, device=device), joint_ids=hand_ids)
    robot.write_joint_damping_to_sim(
        torch.full((num_envs, num_hand), d, device=device), joint_ids=hand_ids)


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


PENALTY = dict(ss_error=10.0, ss_vel_j1=10.0, ss_vel_j2=10.0, mean_error=10.0)


def evaluate(k: float, d: float) -> dict:
    """Run NUM_STEPS with given gains and return tracking metrics."""
    _set_gains(k, d)
    _reset()

    # Warm-up: let the sim settle before commanding targets
    for _ in range(20):
        _step()

    # Set hand target (implicit actuator -> PhysX PD drive target)
    robot.set_joint_position_target(
        target_hand.expand(num_envs, -1), joint_ids=hand_ids)

    errors, vels_j1, vels_j2 = [], [], []
    for _ in range(NUM_STEPS):
        _step()
        pos_j1 = robot.data.joint_pos[:, hand_ids]     # (N, 6) — primary joints
        vel_j1 = robot.data.joint_vel[:, hand_ids]     # (N, 6)
        vel_j2 = robot.data.joint_vel[:, hand_j2_ids]  # (N, 5) — mimic joints

        # Detect simulation explosion — return large penalty so Optuna skips this region
        if torch.isnan(pos_j1).any() or torch.isnan(vel_j1).any() or torch.isnan(vel_j2).any():
            print(f"  [!] NaN detected at step {_} — unstable gains, returning penalty")
            return PENALTY

        errors.append((pos_j1 - target_hand).abs().mean().item())
        vels_j1.append(vel_j1.abs().mean().item())
        vels_j2.append(vel_j2.abs().mean().item())

    # Steady state = last 20% of steps
    ss = int(0.8 * NUM_STEPS)
    return dict(
        ss_error   = float(np.mean(errors[ss:])),
        ss_vel_j1  = float(np.mean(vels_j1[ss:])),
        ss_vel_j2  = float(np.mean(vels_j2[ss:])),
        mean_error = float(np.mean(errors)),
    )


##
# Optuna objective
##


def objective(trial: optuna.Trial) -> float:
    k = trial.suggest_float("stiffness", *K_BOUNDS)
    d = trial.suggest_float("damping",   *D_BOUNDS)

    m = evaluate(k, d)

    # Composite cost:
    #   ss_error    — j1 tracking accuracy (primary objective)
    #   ss_vel_j1   — j1 steady-state oscillation
    #   ss_vel_j2   — j2 mimic joint stability (weighted equally to j1 vel)
    #   gain_reg    — prefer smaller gains that still track well
    gain_reg = 0.05 * (k / K_BOUNDS[1] + d / D_BOUNDS[1])
    cost = m["ss_error"] + 0.5 * m["ss_vel_j1"] + 0.5 * m["ss_vel_j2"] + gain_reg

    record = dict(trial=trial.number, stiffness=round(k, 2),
                  damping=round(d, 2), **m, cost=round(cost, 6))
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")

    print(f"[{trial.number:3d}]  K={k:6.2f}  D={d:5.2f}  "
          f"ss_err={m['ss_error']:.4f}  "
          f"ss_vel_j1={m['ss_vel_j1']:.4f}  ss_vel_j2={m['ss_vel_j2']:.4f}  "
          f"cost={cost:.4f}")
    return cost


##
# Entry point
##


def main():
    LOG_PATH.write_text("")   # clear previous run

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name="hand_gains",
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
