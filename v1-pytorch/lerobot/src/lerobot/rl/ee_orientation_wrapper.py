"""Gym wrapper that adds end-effector orientation control to gym_hil envs.

The stock ``EEActionWrapper`` in gym_hil only forwards xyz position and
gripper, hardcoding orientation deltas to zero.  This wrapper replaces it
with **absolute orientation offset** control:

* Action format: ``[x, y, z, rx, ry, rz, gripper]`` (7-D).
  - xyz in [-1, 1]: per-frame position deltas, scaled by ``ee_step_size``.
  - rx/ry/rz in [-π, π]: **total orientation offset** from the reference
    quaternion (captured at ``reset()``), in radians.  1:1 mapping, no
    extra scaling.
  - gripper in [0, 2].

* Each ``step()`` sets ``mocap_quat = euler2quat(rx,ry,rz) × ref_quat``.
  Because the orientation is *set* rather than *accumulated*, returning
  the input device to its reference pose makes the robot return to its
  exact initial orientation — no drift.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


def _euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Euler RPY (extrinsic XYZ) → quaternion [w, x, y, z]."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product  q1 × q2  with layout [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


class ActionSafetyWrapper(gym.Wrapper):
    """Limits action speed via per-step clipping and EMA smoothing.

    Placed **outside** ``EEOrientationActionWrapper`` so it sees the
    high-level action ``[x, y, z, rx, ry, rz, gripper]`` *before* it is
    converted into physical displacements.

    Parameters
    ----------
    env : gym.Env
    max_pos_action : float
        Maximum absolute value for each xyz position component per step.
        Default 0.4 means at most 40 % of the full range per tick.
    max_orn_action : float
        Maximum absolute *change* in the orientation offset per step
        (radians).  Prevents the policy from flipping the gripper
        upside-down in a single tick.
    smoothing_alpha : float
        EMA blending factor in (0, 1].  ``1.0`` = no smoothing;
        ``0.5`` = blend half of the previous action.
    """

    def __init__(
        self,
        env: gym.Env,
        max_pos_action: float = 0.4,
        max_orn_action: float = 0.25,
        smoothing_alpha: float = 0.6,
    ):
        super().__init__(env)
        self._max_pos = max_pos_action
        self._max_orn = max_orn_action
        self._alpha = smoothing_alpha
        self._prev_action: np.ndarray | None = None

        # Tighten action space so the policy samples within effective bounds
        lo = env.action_space.low.copy()
        hi = env.action_space.high.copy()
        lo[:3] = -max_pos_action
        hi[:3] = max_pos_action
        self.action_space = gym.spaces.Box(lo, hi, dtype=np.float32)

    def reset(self, **kwargs):
        self._prev_action = None
        return self.env.reset(**kwargs)

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).copy()

        # 1. Clip position magnitude
        action[:3] = np.clip(action[:3], -self._max_pos, self._max_pos)

        # 2. Rate-limit orientation change
        if self._prev_action is not None:
            orn_delta = action[3:6] - self._prev_action[3:6]
            orn_delta = np.clip(orn_delta, -self._max_orn, self._max_orn)
            action[3:6] = self._prev_action[3:6] + orn_delta

        # 3. EMA smoothing (skip gripper — it should be discrete/instant)
        if self._prev_action is not None:
            action[:6] = (
                self._alpha * action[:6]
                + (1.0 - self._alpha) * self._prev_action[:6]
            )

        self._prev_action = action.copy()
        return self.env.step(action)


class EEOrientationActionWrapper(gym.Wrapper):
    """Drop-in replacement for ``EEActionWrapper`` with orientation support.

    Parameters
    ----------
    env : gym.Env
        A gym_hil ``FrankaGymEnv`` (or wrapped variant).
    ee_step_size : dict
        ``{"x": float, "y": float, "z": float}`` — position step sizes (m).
    use_gripper : bool
        Whether the last action dim is the gripper command.
    """

    def __init__(
        self,
        env: gym.Env,
        ee_step_size: dict,
        use_gripper: bool = True,
    ):
        super().__init__(env)
        self._pos_scale = np.array(
            [ee_step_size["x"], ee_step_size["y"], ee_step_size["z"]]
        )
        self.use_gripper = use_gripper
        self._ref_quat: np.ndarray | None = None

        # Action space: [x(-1,1), y(-1,1), z(-1,1),
        #                rx(-π,π), ry(-π,π), rz(-π,π)]  + optional [grip(0,2)]
        lo = np.array([-1, -1, -1, -np.pi, -np.pi, -np.pi], dtype=np.float32)
        hi = np.array([1, 1, 1, np.pi, np.pi, np.pi], dtype=np.float32)
        if use_gripper:
            lo = np.concatenate([lo, [0.0]])
            hi = np.concatenate([hi, [2.0]])
        self.action_space = gym.spaces.Box(lo, hi, dtype=np.float32)

    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._ref_quat = self.unwrapped._data.mocap_quat[0].copy()
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)

        pos_delta = action[:3] * self._pos_scale
        orn_offset = action[3:6]  # radians, 1:1, no scaling

        # Absolute orientation set: ref_quat rotated by the offset
        if self._ref_quat is None:
            self._ref_quat = self.unwrapped._data.mocap_quat[0].copy()
        dq = _euler_to_quat(orn_offset[0], orn_offset[1], orn_offset[2])
        target_quat = _quat_mul(dq, self._ref_quat)
        target_quat /= np.linalg.norm(target_quat)
        self.unwrapped._data.mocap_quat[0] = target_quat

        # Build 7-D action for base env: [x, y, z, rx, ry, rz, gripper]
        gripper_cmd = 0.0
        if self.use_gripper:
            gripper_cmd = action[-1] - 1.0  # [0,2] → [-1,1]

        base_action = np.concatenate([
            pos_delta,
            np.zeros(3),       # base env ignores orn in its action vector
            [gripper_cmd],
        ]).astype(np.float32)

        return self.env.step(base_action)
