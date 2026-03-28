# Getting Started

## Prerequisites

- **OS**: Ubuntu 22.04+
- **GPU**: NVIDIA RTX 3070 or higher (CUDA 12+)
- **NVIDIA Isaac Sim 4.5+** — [Installation guide](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html)
- **Isaac Lab** — [Installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
- **Python 3.10+**

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd robotic_grasping_system

# 2. Activate Isaac Lab conda/venv environment
# (follow Isaac Lab installation guide above)

# 3. Install this package in editable mode
python -m pip install -e source/robotic_grasping_system

# 4. Verify the environment is registered
python scripts/list_envs.py
# Should show: Robotic-Grasping-System-v0
```

## Assets

The Vega upper body USD and YCB object USDs are included in the repository as sibling directories. After cloning, the layout should be:

```
parent_directory/
├── robotic_grasping_system/     # this repo
├── vega_upper_body-vega_1/      # Vega robot USD
│   └── vega_upper_body.usd
└── ycb_physics/                 # YCB grasp objects
    ├── 025_mug.usd
    ├── sugar_box.usd
    ├── tomato_soup_can.usd
    ├── 011_banana.usd
    └── mustard_bottle.usd
```

No manual path configuration is needed — the environment resolves asset paths relative to the repo root automatically.

## Quick Test

```bash
# Train with PPO (RSL-RL)
python scripts/rsl_rl/train.py --task Robotic-Grasping-System-v0 --num_envs 4096

# Evaluate a trained policy
python scripts/rsl_rl/play.py --task Robotic-Grasping-System-Play-v0 \
    --load_run <run_folder> --checkpoint <model.pt>
```

See [Usage](usage.md) for all training options, baselines, and Bayesian optimization.
