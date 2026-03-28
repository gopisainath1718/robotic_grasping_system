# Technical Overview

| Component | Details |
|---|---|
| **Platform** | NVIDIA Isaac Sim + Isaac Lab |
| **Robot** | Vega Upper Body (7-DOF arm + 11-DOF hand = 18 DOF) |
| **Objects** | YCB dataset: mug, sugar box, tomato soup can, banana, mustard bottle |
| **RL Algorithm** | PPO (RSL-RL) |
| **Parallel Envs** | 4096 (training), 32 (evaluation) |
| **Sim Rate** | 200 Hz physics, 100 Hz control (decimation = 2) |
| **Observation Space** | 79 dimensions |
| **Action Space** | 18 dimensions (joint position targets) |
| **Episode Length** | 5s (training), 10s (evaluation) |

## Task Description

The agent controls the right arm and hand of the Vega humanoid to:
1. **Reach** toward a randomly selected YCB object on a table
2. **Grasp** the object using all five fingertips
3. **Lift** the object off the table surface

Success is defined as maintaining no table-object contact for 50 consecutive simulation steps.

## Observation Space (79-dim)

| Observation | Dimensions | Purpose |
|---|---|---|
| Joint positions | 18 | Proprioception (arm + hand) |
| Joint velocities | 18 | Proprioception dynamics |
| Object pose | 7 | Task target (position + quaternion) |
| Object bounding box | 3 | Shape awareness for generalization across objects |
| Fingertip positions | 15 | World-frame fingertip poses (5 fingers x 3D) |
| Last action | 18 | Previous action for temporal context |

## Action Space (18-dim)

Joint position targets for:
- **Right arm** (7 DOF): `R_arm_j1` through `R_arm_j7`
- **Right hand** (11 DOF): 4 fingers x 2 joints + thumb x 3 joints
