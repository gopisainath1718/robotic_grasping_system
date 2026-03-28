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
# Run trained policy
python scripts/rsl_rl/play.py --task Robotic-Grasping-System-Play-v0 \
    --load_run <run_folder> --checkpoint <model.pt>
```

## Bayesian Optimization (PD Gain Tuning)

Optimizes stiffness and damping for either arm or hand joints using Optuna's TPE sampler.

```bash
# Optimize hand joint PD gains
python optimization/BO.py --mode hand --headless

# Optimize arm joint PD gains
python optimization/BO.py --mode arm --headless

# With more parallel envs for averaging
python optimization/BO.py --mode hand --headless --num_envs 8
```

- `--mode`: `arm` or `hand` (required)
- `--num_envs`: parallel robot instances for metric averaging (default: 4)
- Runs 100 trials, 1000 physics steps each
- Results saved to `bo_results/best_gains_{mode}.json`
- Trial logs saved to `bo_results/bo_log_{mode}.jsonl`
