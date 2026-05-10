"""
Staged Hybrid Reward Wrapper v2.1 — JAX hil-serl-sim port.

Byte-for-byte mirror of the PyTorch / lerobot wrapper at
`hilserl-surrol-improved-v2/lerobot/src/lerobot/rl/staged_reward_wrapper.py`.
Math, weights, clips, and grasp-event predicate are identical so that
JAX-vs-PyTorch comparisons isolate the *framework*, not the reward.

Aligned with `hilserl_抓取任务_稀疏奖励替换为稠密奖励_完整实现流程.docx`
(2026-05-04 engineering scheme). See the PyTorch wrapper's docstring or
the V2 fork's `docs/CHANGES.md` §8b for the design rationale.

Schema (scheme §6):
  r_t = w_s · r_success + w_a · r_approach + w_g · r_grasp + w_l · r_lift − w_p · r_penalty
  defaults: w_s=1.00  w_a=0.05  w_g=0.10  w_l=0.05  w_p=0.01
  per-step total clipped to [-1.0, 2.0]  (scheme §9, prevents Q-value explosion)

Anti-farming guarantees:
  - Hovering near target: r_approach is differential, returns ~0 if not moving.
  - Light-touch / vibration: r_grasp is an event, fires once per episode.
  - Excessive lift: r_lift is clipped to 5 cm above z_pick_ref.

Usage in pick_cube_sim/config.py:
    env = PandaPickCubeGymEnv(...)
    env = StagedRewardWrapper(env)        # this module
    env = KeyBoardIntervention2(env)      # if not headless
    env = SERLObsWrapper(env, ...)
    env = ChunkingWrapper(env, ...)

Reads cube/TCP state directly from MuJoCo via `unwrapped._data.sensor(...)`,
so it is independent of obs-space shape.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class StagedRewardWrapper(gym.Wrapper):
    """Hybrid sparse + dense shaping reward (scheme-aligned v2.1)."""

    # Grasp-event detection (scheme §7.2 grasp predicate)
    GRASP_PROXIMITY_M: float = 0.04
    GRASP_LIFT_TRIGGER_M: float = 0.005
    GRASP_STREAK_REQUIRED: int = 2

    # Anti-farming clips (scheme §7.2)
    APPROACH_CLIP: float = 0.02
    LIFT_CLIP_M: float = 0.05

    # Per-step total safety bound (scheme §9 step 5)
    PER_STEP_CLIP_LOW: float = -1.0
    PER_STEP_CLIP_HIGH: float = 2.0

    # Default weights (scheme §6)
    DEFAULT_WEIGHTS: dict[str, float] = {
        "success": 1.00,
        "approach": 0.05,
        "grasp": 0.10,
        "lift": 0.05,
        "penalty": 0.01,
    }

    def __init__(
        self,
        env: gym.Env,
        weights: dict[str, float] | None = None,
        lift_target: float = 0.10,        # kept for signature parity (unused)
        max_episode_steps: int = 100,     # kept for signature parity (unused)
    ):
        super().__init__(env)
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights:
            for k, v in weights.items():
                if k not in self.DEFAULT_WEIGHTS:
                    raise ValueError(
                        f"Unknown reward weight '{k}'. "
                        f"Valid keys: {list(self.DEFAULT_WEIGHTS)}"
                    )
                self.weights[k] = float(v)

        self._z_init: float | None = None
        self._prev_dist: float | None = None
        self._z_pick_ref: float | None = None
        self._grasp_event_fired: bool = False
        self._grasp_streak: int = 0
        self._episode_sums: dict[str, float] = self._zero_breakdown()

    @staticmethod
    def _zero_breakdown() -> dict[str, float]:
        return {
            "success": 0.0,
            "approach": 0.0,
            "grasp": 0.0,
            "lift": 0.0,
            "penalty": 0.0,
            "total": 0.0,
        }

    def _read_state(self) -> tuple[np.ndarray, np.ndarray]:
        data = self.unwrapped._data
        block_pos = data.sensor("block_pos").data.copy()
        tcp_pos = data.sensor("2f85/pinch_pos").data.copy()
        return block_pos, tcp_pos

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        block_pos, tcp_pos = self._read_state()
        self._z_init = float(block_pos[2])
        self._prev_dist = float(np.linalg.norm(block_pos - tcp_pos))
        self._z_pick_ref = None
        self._grasp_event_fired = False
        self._grasp_streak = 0
        self._episode_sums = self._zero_breakdown()
        return obs, info

    def step(self, action):
        obs, _env_reward, terminated, truncated, info = self.env.step(action)

        block_pos, tcp_pos = self._read_state()
        curr_dist = float(np.linalg.norm(block_pos - tcp_pos))
        z_block = float(block_pos[2])

        # 1. r_approach: differential (potential-based shaping proxy)
        if self._prev_dist is None:
            r_approach = 0.0
        else:
            r_approach = float(np.clip(
                self._prev_dist - curr_dist,
                -self.APPROACH_CLIP, self.APPROACH_CLIP,
            ))
        self._prev_dist = curr_dist

        # 2. r_grasp: event-based, fires at most once per episode
        is_close = curr_dist < self.GRASP_PROXIMITY_M
        is_lifted = (
            self._z_init is not None
            and (z_block - self._z_init) > self.GRASP_LIFT_TRIGGER_M
        )
        if is_close and is_lifted:
            self._grasp_streak += 1
        else:
            self._grasp_streak = 0

        r_grasp = 0.0
        if not self._grasp_event_fired and self._grasp_streak >= self.GRASP_STREAK_REQUIRED:
            self._grasp_event_fired = True
            self._z_pick_ref = z_block
            r_grasp = 1.0

        # 3. r_lift: clipped, gated on grasp event
        if self._z_pick_ref is not None:
            r_lift = float(np.clip(
                z_block - self._z_pick_ref,
                0.0, self.LIFT_CLIP_M,
            ))
        else:
            r_lift = 0.0

        # 4. r_success: terminal sparse signal (scheme §5 principle 1)
        # The upstream PandaPickCubeGymEnv puts succeed flag in info dict.
        r_success = 1.0 if info.get("succeed", False) else 0.0

        # 5. r_penalty: placeholder (0.0 by default for sim)
        # Sim has no contact / collision sensor; EE bounds are upstream-clipped.
        r_penalty = 0.0

        # Combine (scheme §6 weighted sum)
        sub = {
            "success": r_success,
            "approach": r_approach,
            "grasp": r_grasp,
            "lift": r_lift,
            "penalty": r_penalty,
        }
        total_unclipped = (
            self.weights["success"] * sub["success"]
            + self.weights["approach"] * sub["approach"]
            + self.weights["grasp"] * sub["grasp"]
            + self.weights["lift"] * sub["lift"]
            - self.weights["penalty"] * sub["penalty"]
        )
        total = float(np.clip(
            total_unclipped, self.PER_STEP_CLIP_LOW, self.PER_STEP_CLIP_HIGH
        ))

        # Logging hooks
        info["reward_dict"] = {**sub, "total": total}
        for k, v in sub.items():
            self._episode_sums[k] += v
        self._episode_sums["total"] += total

        if terminated or truncated:
            info["episode_reward_breakdown"] = {
                **self._episode_sums,
                "grasp_event_fired": int(self._grasp_event_fired),
                "z_pick_ref": (
                    float(self._z_pick_ref)
                    if self._z_pick_ref is not None else float("nan")
                ),
            }

        return obs, total, terminated, truncated, info
