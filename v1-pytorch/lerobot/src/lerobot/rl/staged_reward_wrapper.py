"""
Sparse Reward Wrapper for HIL-SERL Pick-and-Lift Task

Based on: "Catch-and-Lift Reward Design Checklist for HIL-SERL-Style Training"
Reference: HIL-SERL paper (Luo et al.), LeRobot HIL workflow, hil-serl-sim replication.

## Design Philosophy (from checklist §2)

HIL-SERL is built around offline demonstrations + online RL + human corrective
interventions. In this setup the reward does NOT need to encode every intermediate
behavior. It only needs to reflect the true task goal.

Dense shaping:
  - Increases tuning burden
  - Can reward near-success behavior instead of actual success
  - Partially defeats the HIL-SERL workflow (which relies on human guidance)

Sparse success reward:
  - Faithfully represents task completion
  - Consistent with the paper's binary success classifier design
  - The public hil-serl-sim replication reports 100% success after ~30K steps
    / ~1 hour with human intervention using this style of reward.

## Reward (Template 1 from checklist §8)

  reward = 1.0   if task succeeded (object lifted above threshold)
  reward = 0.0   otherwise
  reward -= 0.02 for unnecessary gripper toggles  ← handled by GripperPenaltyWrapper

The gripper penalty is applied separately by GripperPenaltyWrapper in
gym_manipulator.py (penalty configured in train_config_gym_hil_touch.json).
This wrapper only handles the success/failure signal.

## Success Detection (Option A — state-based, from checklist §5)

Uses `info["succeed"]` from PandaPickCubeGymEnv, which checks:
  - Object z-position above threshold (lifted off the table)
  - Object still in gripper contact
Deterministic, cheap, easy to debug.
"""

import gymnasium as gym


class StagedRewardWrapper(gym.Wrapper):
    """Sparse reward wrapper for pick-and-lift (HIL-SERL style).

    Gives:
      reward = 1.0  when info["succeed"] is True
      reward = 0.0  otherwise

    Exploration is driven by human demonstrations (offline buffer) and
    corrective interventions, NOT by dense shaping signals.

    Gripper penalty (-0.02 per unnecessary toggle) is handled separately
    by GripperPenaltyWrapper, consistent with the HIL-SERL paper.
    """

    def __init__(self, env: gym.Env, **kwargs):
        super().__init__(env)
        # No extra state needed — pure pass-through with reward override

    def step(self, action):
        obs, _reward, terminated, truncated, info = self.env.step(action)

        # Sparse binary reward: 1.0 on success, 0.0 otherwise
        # (checklist Template 1 §8)
        reward = 1.0 if info.get("succeed", False) else 0.0

        return obs, reward, terminated, truncated, info
