# Design Decisions

## Hierarchical Reward Structure

The reward is designed as a **progression**: each phase unlocks only after the previous phase is sufficiently learned. This prevents the agent from receiving conflicting gradients early in training.

```
reach → fingertip contact → finger close → lift + lift velocity
```

**Phase 1 — Always active:**

| Reward Term | Weight | Purpose |
|---|---|---|
| `reach` | -2.0 | Minimize mean fingertip-to-object distance |
| `fingertip_contact` | +1.0 | Maximize number of fingers touching object (0-5) |
| `object_lateral_vel` | -0.1 | Penalize lateral object velocity (prevents knocking/sliding) |
| `action_rate` | -0.001 | Smooth actions (L2 penalty on action deltas) |
| `joint_vel_penalty` | -0.0001 | Regularize joint velocities |

**Phase 2 — Gated by contact running mean > 3:**

| Reward Term | Weight | Purpose |
|---|---|---|
| `finger_close` | +5.0 | Reward thumb-to-finger-centroid closeness: `1/(1 + dist)` |

**Phase 3 — Gated by contact running mean > 3.5:**

| Reward Term | Weight | Purpose |
|---|---|---|
| `lift` | +4.0 | Penalize table-object contact (returns -1 while touching, 0 when lifted) |
| `object_lift_vel` | +1.0 | Reward upward object velocity (clamped >= 0) |

**Running mean gating**: A circular buffer (window=2000 steps) tracks the mean fingertip contact count across all environments. `finger_close` activates at 3+ mean contacts, and `lift`/`object_lift_vel` activate at 3.5+. This acts as an implicit curriculum — the agent must first learn to reach and make contact, then close its grip, before being asked to lift.

**Why negative weight for reach?** The function returns a distance (>= 0) — the negative weight converts "minimize distance" into "maximize reward." Conversely, `finger_close` now returns `1/(1 + dist)` (higher when closer), so it uses a positive weight.

## Domain Randomization

Applied on every environment reset to improve policy robustness:

| Randomization | Range | Purpose |
|---|---|---|
| **Joint position offset** | +/-0.1 rad (~+/-5.7 deg) | Prevents the policy from memorizing a single start pose |
| **Table height** | +/-10 cm | Generalizes across varying table surfaces |
| **Object placement** | +/-10 cm (x and y) | Forces the policy to reach in different directions |

Object type is also randomized — each environment spawns one of 5 YCB objects at random, so the policy must generalize across different shapes and sizes.

## Observation Space Design

Each component serves a specific purpose:

- **Joint positions/velocities (36 dims)**: Proprioception — the agent needs to know its own state
- **Object pose (7 dims)**: Task-relevant — where to reach
- **Object bounding box (3 dims)**: Enables generalization across YCB objects with different shapes. The policy can adapt its grasp width based on object size
- **Fingertip positions (15 dims)**: World-frame positions of all 5 fingertips — combined with object pose, the policy can infer spatial relationships
- **Last action (18 dims)**: Previous action for temporal context, helping the policy produce smoother actions

## PD Gain Tuning via Bayesian Optimization

Manual tuning of actuator PD gains was insufficient — the hand has 11 joints with coupled dynamics, and small gain changes produce large behavioral differences. I used **Optuna's TPE sampler** to optimize stiffness and damping:

| Actuator | Stiffness (K) | Damping (D) | Method |
|---|---|---|---|
| Arm | 191.43 | 7.35 | Bayesian Optimization |
| Hand | 27.08 | 0.1 | Bayesian Optimization |

The optimization runs 1000 physics steps per trial, evaluating grasp stability across parallel environments. Search ranges: K in [1, 50] for hand, D in [0.1, 10].

## Network Architecture

```
Actor:  [obs_dim] -> 512 -> 256 -> 128 -> [18 actions]    (ELU activation)
Critic: [obs_dim] -> 512 -> 256 -> 128 -> [1 value]       (ELU activation)
```

- **3-layer MLP**: Large enough for the 79-dim observation space without being wasteful
- **ELU activation**: Avoids dead neurons (unlike ReLU) while being computationally efficient
- **Adaptive KL schedule**: Learning rate adapts to maintain KL divergence near 0.01 — prevents policy collapse during training

## PPO Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Learning rate | 1e-3 | Adaptive KL schedule adjusts this dynamically |
| Clip param | 0.2 | Standard PPO clipping |
| Entropy coef | 0.005 | Low — task has clear structure, don't over-explore |
| GAE lambda | 0.95 | Standard bias-variance tradeoff |
| Mini batches | 4 | With 4096 envs x 16 steps = 65536 samples per update |
| Empirical normalization | True | Stabilizes training with diverse observation scales |
