# Challenges and Iterations

## What Didn't Work (and Why)

**`approach_angle_reward`** (disabled — alignment bug):
Intended to reward the hand approaching perpendicular to the object's long axis. Computes alignment between the hand's X-axis and the object's principal axis. However, the body frame orientation didn't behave as expected — the reward was occasionally maximized in clearly wrong configurations. Needs debugging of the quaternion-to-rotation-matrix conversion for the hand base frame.

**`grasp_reward`** (disabled — premature):
A geometric grasp quality metric based on CrossDex (ICLR 2025): measures how well the object center lies between thumb and finger centroid. While mathematically sound, it introduced too many competing gradients when combined with the contact-based rewards. The simpler `fingertip_contact` + `finger_close` combination proved more trainable.

**`object_vel` -> split into `object_lateral_vel` + `object_lift_vel`**:
The original `object_vel` penalty penalized all object motion equally, causing the agent to avoid the object entirely. Splitting it into lateral (penalized, -0.1) and vertical (rewarded when gated, +1.0) components solved this — the agent is discouraged from knocking but encouraged to lift.

**`action_rate_l2` and `joint_vel_l2`** (re-enabled with low weights):
Initially disabled because they interfered with fast corrective movements. Re-enabled at very low weights (-0.001 and -0.0001 respectively) — enough to regularize without constraining the policy.

## Domain Randomization

Implemented but currently disabled for training stability:
- **Table height randomization** (+/-10 cm)
- **Object placement randomization** (+/-5 cm x/y)
- **Joint position randomization** (+/-45 deg)

These are ready to enable once the base policy converges — incremental randomization prevents catastrophic forgetting.

## Contact Sensor Tuning

Contact sensors on each fingertip filter specifically for the grasped object (not table or self-collisions). The force threshold (0.5 N) required careful tuning — too low triggered false positives from simulation noise, too high missed light touches during initial contact.
