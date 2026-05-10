"""
Staged Hybrid Reward Wrapper v2.1 (scheme 2026-05-04).

Aligned with `hilserl_抓取任务_稀疏奖励替换为稠密奖励_完整实现流程.docx` —
the engineering spec that explicitly mitigates the three reward-hacking
failure modes the scheme calls out:

  - 靠近刷分 (hovering-near-target farming)
       Mitigated by: differential approach `clip(d_prev − d_now, ±0.02)`.
       Hovering produces ≈0 reward because distance stops decreasing;
       continuous proxies like 1/(1+5d) instead reward staying close,
       which is exactly the failure mode.

  - 轻触刷分 (light-touch farming)
       Mitigated by: grasp is an *event*, not a continuous ramp. It fires
       once per episode when (TCP-block 3D distance < 4 cm) AND (block z
       raised > 5 mm above z_init) holds for ≥2 consecutive steps. After
       firing it returns 0 forever — actors can't "vibrate the gripper"
       to milk it.

  - 抬高刷分 (excessive-lift farming)
       Mitigated by: r_lift = clip(z_block − z_pick_ref, 0, 0.05) and the
       env-level `terminate_on_success=True`. Lift saturates at 5 cm above
       the grasp point — well below the 10 cm success threshold — so any
       lift past 5 cm goes unrewarded, and success itself ends the
       episode immediately.

Reward schema
-------------
  r_t = w_s · r_success + w_a · r_approach + w_g · r_grasp + w_l · r_lift − w_p · r_penalty

Default weights (from scheme §6):
  w_s = 1.00   w_a = 0.05   w_g = 0.10   w_l = 0.05   w_p = 0.01

Per-step total is clipped to [−1.0, 2.0] (scheme §9 step 5: prevent Q
value explosion from sensor outliers).

Expected episode reward ranges (per scheme §12 / §11.B):
  Random / no-grasp:           0.0 – 0.2
  Reaching but no stable grasp: 0.1 – 0.3
  Stable grasp, partial lift:   0.3 – 0.6
  Full success:                 1.2 – 1.8

Note that the per-episode reward range (≈0–2) is intentionally an order of
magnitude smaller than the previous V2 design (which produced 15–60 per
episode). The previous design was an *ad-hoc* dense reward; this one is
the engineering-spec-compliant shaping signal that preserves the
HIL-SERL framework's sparse-success target as the dominant terminal
signal, with shaping providing only mild gradient assistance.

Logging
-------
Every step injects `info["reward_dict"]` with the per-stage breakdown so
downstream debug / visualisation can see exactly where each reward unit
came from. At episode termination, `info["episode_reward_breakdown"]`
holds the per-stage *sums* across the whole episode; `actor.py` forwards
these to the learner so they appear as columns in `episode_metrics.csv`
and as wandb panels.

Provenance
----------
Replaces the v2.0 design (`r ∈ [0, 10]` per step + `+10` success bonus)
which mirrored the published HIL-SERL paper's exploration-aggressive
shaping but proved vulnerable to all three farming modes above when run
end-to-end. v2.1 is a strict superset of v1's sparse reward (success
weight = 1.0 is identical) plus a small shaping term.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class StagedRewardWrapper(gym.Wrapper):
    """Hybrid sparse + dense shaping reward (scheme-aligned v2.1).

    Args:
        env: a gym env exposing `unwrapped._data.sensor("block_pos")` and
            `unwrapped._data.sensor("2f85/pinch_pos")` (gym_hil
            PandaPickCubeGymEnv satisfies this).
        weights: optional override for the five reward weights. Keys must
            be a subset of {"success","approach","grasp","lift","penalty"}.
            Missing keys keep their `DEFAULT_WEIGHTS` value.
        lift_target: kept for signature compatibility with v2.0 callers;
            UNUSED in v2.1 (lift is anchored to z_pick_ref, not to a fixed
            target).
        max_episode_steps: kept for signature compatibility; UNUSED.
    """

    # ── Grasp-event detection ──────────────────────────────────────
    # Three coupled conditions identify a stable grasp without needing a
    # contact / current sensor (which the sim env doesn't expose):
    #   1. TCP within 4 cm of block (3D)
    #   2. Block raised at least 5 mm above its initial z
    #   3. Both conditions held for ≥ 2 consecutive control steps
    GRASP_PROXIMITY_M: float = 0.04
    GRASP_LIFT_TRIGGER_M: float = 0.005
    GRASP_STREAK_REQUIRED: int = 2

    # ── Anti-farming clips (scheme §7.2) ───────────────────────────
    APPROACH_CLIP: float = 0.02     # ±2 cm/step max approach reward
    LIFT_CLIP_M: float = 0.05       # 5 cm cap above z_pick_ref

    # ── Per-step total safety bound (scheme §9 step 5) ─────────────
    # Asymmetric: lower bound −1.0 (penalty term × per-step rate),
    # upper bound 2.0 (success bonus + small shaping in same step).
    PER_STEP_CLIP_LOW: float = -1.0
    PER_STEP_CLIP_HIGH: float = 2.0

    # ── Default weights (scheme §6) ────────────────────────────────
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
        # Merge user-provided weights with defaults; warn on unknown keys.
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights:
            for k, v in weights.items():
                if k not in self.DEFAULT_WEIGHTS:
                    raise ValueError(
                        f"Unknown reward weight '{k}'. "
                        f"Valid keys: {list(self.DEFAULT_WEIGHTS)}"
                    )
                self.weights[k] = float(v)

        # Per-episode state (set in reset)
        self._z_init: float | None = None
        self._prev_dist: float | None = None
        self._z_pick_ref: float | None = None
        self._grasp_event_fired: bool = False
        self._grasp_streak: int = 0
        self._episode_sums: dict[str, float] = self._zero_breakdown()

    # ─────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────

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
        """Read block + TCP positions directly from MuJoCo sensors."""
        data = self.unwrapped._data
        block_pos = data.sensor("block_pos").data.copy()
        tcp_pos = data.sensor("2f85/pinch_pos").data.copy()
        return block_pos, tcp_pos

    # ─────────────────────────────────────────────────────────────────
    # Gym API
    # ─────────────────────────────────────────────────────────────────

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

        # ── 1. r_approach: differential (potential-based shaping proxy) ──
        # Equivalent to F(s,a,s') = γΦ(s') − Φ(s) with Φ(s) = −d(s) and
        # γ ≈ 1, then clipped to limit per-step magnitude. Hovering near
        # the target produces ~0 reward (no Δd), which kills the
        # "靠近刷分" failure mode.
        if self._prev_dist is None:
            r_approach = 0.0
        else:
            r_approach = float(np.clip(
                self._prev_dist - curr_dist,
                -self.APPROACH_CLIP, self.APPROACH_CLIP,
            ))
        self._prev_dist = curr_dist

        # ── 2. r_grasp: event-based, fires at most once per episode ────
        # Vibrating the gripper near the cube cannot farm this — the
        # event predicate either holds and fires, or never holds.
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

        # ── 3. r_lift: clipped, gated on grasp event ───────────────────
        # Returns 0 until grasp event fires; then linear in (z − z_pick_ref)
        # up to a 5 cm cap. Beyond 5 cm the policy is unrewarded for
        # lifting — combined with terminate_on_success=True this kills
        # "抬高刷分".
        if self._z_pick_ref is not None:
            r_lift = float(np.clip(
                z_block - self._z_pick_ref,
                0.0, self.LIFT_CLIP_M,
            ))
        else:
            r_lift = 0.0

        # ── 4. r_success: terminal sparse signal (scheme §5 principle 1) ──
        # Identical to V1 sparse — guarantees the optimisation target is
        # task completion, not shaping-sum maximisation.
        r_success = 1.0 if info.get("succeed", False) else 0.0

        # ── 5. r_penalty: placeholder (0.0 by default) ─────────────────
        # Sim PandaPickCubeGymEnv has no contact-force / collision sensor,
        # and EE workspace bounds are clipped by the controller upstream
        # so an explicit out-of-bounds detector would never fire.
        # Gripper-waste is handled separately by GripperPenaltyWrapper
        # via info["discrete_penalty"] consumed by SAC's discrete-critic
        # gradient channel — NOT the main reward — so we don't double
        # count it here.
        r_penalty = 0.0

        # ── Combine (scheme §6 weighted sum) ───────────────────────────
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

        # ── Logging hooks ──────────────────────────────────────────────
        # Per-step breakdown so debug code / wandb media panels can show
        # exactly where each unit of reward came from.
        info["reward_dict"] = {**sub, "total": total}

        # Accumulate per-episode sums, surfaced at termination.
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
