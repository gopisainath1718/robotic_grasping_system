# Robotic Grasping System

A reinforcement learning system for dexterous robotic grasping using the **Vega humanoid upper body** in NVIDIA Isaac Sim + **Isaac Lab**. The agent learns to reach, grasp, and lift diverse YCB objects using a 7-DOF arm and 11-DOF hand, trained with PPO across 4096 parallel environments.

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![Isaac Sim 4.5+](https://img.shields.io/badge/Isaac%20Sim-4.5%2B-green)
![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-1.0%2B-purple)
![CUDA](https://img.shields.io/badge/CUDA-12%2B-orange)

---

## Documentation

| Document | Description |
|---|---|
| [Getting Started](docs/getting_started.md) | Prerequisites, installation, asset setup |
| [Technical Overview](docs/technical_overview.md) | Environment specs, observation/action spaces, task description |
| [Usage](docs/usage.md) | Training, evaluation, Bayesian optimization |
| [Design Decisions](docs/design_decisions.md) | Reward structure, observation design, PD gain tuning, PPO config |
| [Iterations & Challenges](docs/iterations_and_challenges.md) | Training iterations, what worked, what didn't work, challenges |

---

## Project Structure

```
robotic_grasping_system/
├── scripts/rsl_rl/
│   ├── train.py                 # Training entry point
│   ├── play.py                  # Evaluation / inference
│   └── cli_args.py              # CLI argument parsing
├── source/robotic_grasping_system/
│   └── robotic_grasping_system/
│       └── tasks/manager_based/robotic_grasping_system/
│           ├── robotic_grasping_system_env_cfg.py  # Environment config
│           ├── agents/rsl_rl_ppo_cfg.py            # PPO hyperparameters
│           └── mdp/
│               ├── observations.py  # Observation functions
│               ├── rewards.py       # Reward functions
│               ├── terminations.py  # Success / failure conditions
│               ├── events.py        # Domain randomization
│               └── debug_vis.py     # Debug visualization
├── optimization/BO.py               # Bayesian optimization for PD gains
├── bo_results/                      # Optimized gain parameters
├── logs/                            # Training logs & checkpoints
└── docs/                            # Detailed documentation
```

---

## Results

Training plots are located in [`docs/plots/`](docs/plots/). These plots are generated using tensorboard.

---

## Future Work

- Fix and add approach_angle for better pre-grasp hand orientation
- Depth-based object encoding replace bounding box with point cloud features
- Curriculum learning for automatic difficulty scaling based on success rate (currently implemented only for rewards)
- Hyperparameter optimization (PPO gains via Optuna/Bayesian optimization)
### Alternative Approaches

- **Imitation Learning** — collect demonstrations and train a BC policy directly, bypassing reward engineering
- **Offline RL on top of BC** — pretrain with behavior cloning, then finetune with offline RL (e.g., CQL, IQL) using the collected demonstrations to improve beyond the demonstrator

---

## References

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)
- [YCB Object Dataset](https://www.ycbbenchmarks.com/)
- [CrossDex](https://github.com/PKU-RL/CrossDex) — reward design inspiration
- [DexterousHands](https://github.com/PKU-MARL/DexterousHands) — dexterous manipulation reference
- [UniDexGrasp2](https://github.com/PKU-EPIC/UniDexGrasp2) — universal dexterous grasping reference
- [Residual Off-Policy RL](https://github.com/amazon-far/residual-offpolicy-rl) — residual policy learning reference
