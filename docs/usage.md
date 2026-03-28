# Usage

## Training

```bash
# Train with PPO (RSL-RL) — 4096 parallel environments
python scripts/rsl_rl/train.py --task Robotic-Grasping-System-v0 --num_envs 4096

# Train with video logging
python scripts/rsl_rl/train.py --task Robotic-Grasping-System-v0 --num_envs 4096 --video

# Resume from checkpoint
python scripts/rsl_rl/train.py --task Robotic-Grasping-System-v0 \
    --load_run <run_folder> --checkpoint <model.pt>
```

Training logs are saved to `logs/rsl_rl/robotic_grasping_system/<timestamp>/`.

## Evaluation

```bash
# Run trained policy (32 envs, 10s episodes)
python scripts/rsl_rl/play.py --task Robotic-Grasping-System-Play-v0 \
    --load_run <run_folder> --checkpoint <model.pt>
```

## Baselines

```bash
# Random agent — establishes exploration baseline
python scripts/random_agent.py --task Robotic-Grasping-System-v0

# Zero-action agent — gravity-only baseline
python scripts/zero_agent.py --task Robotic-Grasping-System-v0
```

## Bayesian Optimization (PD Gain Tuning)

```bash
# Optimize hand joint stiffness and damping
python optimization/BO.py
```

Results saved to `bo_results/`.
