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
| [Technical Overview](docs/technical_overview.md) | Environment specs, observation/action spaces, task description |
| [Getting Started](docs/getting_started.md) | Prerequisites, installation, asset setup |
| [Usage](docs/usage.md) | Training, evaluation, baselines, Bayesian optimization |
| [Design Decisions](docs/design_decisions.md) | Reward structure, observation design, PD gain tuning, PPO config |
| [Challenges & Iterations](docs/challenges.md) | What didn't work, domain randomization, contact sensor tuning |

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

> **Training is currently in progress.**

```bash
tensorboard --logdir logs/rsl_rl/robotic_grasping_system/
```

---

## Future Work

- **Enable domain randomization** incrementally for sim-to-real robustness
- **Fix `approach_angle_reward`** for better pre-grasp hand orientation
- **Depth-based object encoding** — replace bounding box with point cloud features
- **Curriculum learning** — automatic difficulty scaling based on success rate
- **Sim-to-real transfer** — deploy trained policy on physical Vega hardware

---

## References

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)
- [CrossDex: Cross-Embodiment Dexterous Grasping (ICLR 2025)](https://arxiv.org/abs/2403.09181) — reward design inspiration
- [Dexterous Functional Grasping (CoRL 2023)](https://dexfunc.github.io/) — reward design inspiration
- [YCB Object Dataset](https://www.ycbbenchmarks.com/)
