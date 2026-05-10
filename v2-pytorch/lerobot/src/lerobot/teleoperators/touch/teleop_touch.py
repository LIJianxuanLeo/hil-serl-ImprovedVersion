import logging
import sys
from enum import IntEnum
from typing import Any

import numpy as np

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .configuration_touch import TouchTeleopConfig

log = logging.getLogger(__name__)


class GripperAction(IntEnum):
    CLOSE = 0
    STAY = 1
    OPEN = 2


# Position mapping: Touch (x,y,z) -> Robot (z, x, y).
_TOUCH_TO_ROBOT_POS = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

# Orientation mapping: Touch (gimbal0=roll, gimbal1=pitch, gimbal2=yaw)
#   -> Robot (rx=gimbal2, ry=gimbal1, rz=gimbal0)
# so that Touch pitch controls gripper pitch, Touch yaw controls gripper yaw.
_TOUCH_TO_ROBOT_ORN = np.array(
    [
        [0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


class TouchTeleop(Teleoperator):
    """Teleoperator using the Geomagic Touch (Phantom Omni) haptic device.

    Reads 6-DOF absolute pose from the device and converts it into
    end-effector commands for the ``EEOrientationActionWrapper``.

    **Position** uses per-frame deltas (same as before).

    **Orientation** uses absolute offset tracking:  the total angular
    offset from a reference gimbal reading is output each frame.  This
    completely eliminates the cumulative drift that plagued the earlier
    frame-to-frame delta approach, because returning the stylus to the
    reference pose yields *exactly* zero offset.

    Two operating modes controlled by ``config.clutch_mode``:

    **clutch_mode = True** (default, for recording / manual control):
        - Stylus movement always drives the robot.
        - Button 1 (front) = **clutch**: hold to freeze the robot while
          repositioning the stylus.  Release to resume without jumps.
        - Button 2 (back)  = **gripper toggle**.
        - ``is_intervention`` is always ``True``.

    **clutch_mode = False** (for HIL-SERL training):
        - Button 1 (front) = **intervention trigger**: hold to take control.
        - Button 2 (back)  = **gripper toggle**.
        - ``is_intervention`` follows Button 1 state.
    """

    config_class = TouchTeleopConfig
    name = "touch"

    def __init__(self, config: TouchTeleopConfig):
        super().__init__(config)
        self.config = config

        self._connected = False
        self._touch_module = None
        self._get_pose_fn = None

        # Position tracking (per-frame delta)
        self._prev_pos: np.ndarray | None = None

        # Orientation tracking (absolute offset with clutch accumulator)
        self._ref_orn: np.ndarray | None = None
        self._orn_accumulator = np.zeros(3, dtype=np.float64)

        self._gripper_open = True
        self._button2_was_pressed = False
        self._last_button: int = 0

    # ------------------------------------------------------------------
    # Teleoperator interface
    # ------------------------------------------------------------------

    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (7,),
                "names": {
                    "delta_x": 0, "delta_y": 1, "delta_z": 2,
                    "delta_rx": 3, "delta_ry": 4, "delta_rz": 5,
                    "gripper": 6,
                },
            }
        return {
            "dtype": "float32",
            "shape": (6,),
            "names": {
                "delta_x": 0, "delta_y": 1, "delta_z": 2,
                "delta_rx": 3, "delta_ry": 4, "delta_rz": 5,
            },
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        if self._connected:
            return

        haptic_path = self.config.haptic_module_path
        if haptic_path and haptic_path not in sys.path:
            sys.path.insert(0, haptic_path)

        import touch_haptic  # type: ignore[import-untyped]

        self._touch_module = touch_haptic

        if self.config.device_name == "right":
            touch_haptic.initTouch_right()
            self._get_pose_fn = touch_haptic.getDevicePose_right
        else:
            touch_haptic.initTouch_left()
            self._get_pose_fn = touch_haptic.getDevicePose_left

        touch_haptic.startScheduler()
        self._connected = True

        mode = "clutch" if self.config.clutch_mode else "intervention"
        log.info(f"Touch device connected  (mode={mode})")
        if self.config.clutch_mode:
            log.info("  Move stylus to control robot.  Hold Button 1 (front) to clutch/pause.")
        else:
            log.info("  Hold Button 1 (front) to intervene.  Release to let policy act.")
        log.info("  Press Button 2 (back) to toggle gripper.")

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def reset_state(self) -> None:
        """Reset orientation accumulator and position anchor.

        Call this when a new episode begins so that the orientation
        offset starts from zero relative to the new ``_ref_quat`` that
        the wrapper captures at ``env.reset()``.
        """
        self._prev_pos = None
        self._ref_orn = None
        self._orn_accumulator = np.zeros(3, dtype=np.float64)

    # ------------------------------------------------------------------

    def _read_device(self) -> tuple[np.ndarray, np.ndarray, int]:
        """Return (position_mm, gimbal_angles_rad, button_bitmask).

        If the device is disconnected or an error occurs, returns zeros
        and logs a warning instead of crashing.
        """
        pose = np.zeros(7, dtype=np.float32)
        try:
            self._get_pose_fn(pose)
        except Exception as e:
            if self._connected:
                log.warning(f"Touch device read failed (disconnected?): {e}")
                self._connected = False
            return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64), 0
        pos = pose[:3].astype(np.float64)
        orn = pose[3:6].astype(np.float64)
        button = int(pose[6])
        self._last_button = button
        return pos, orn, button

    def _update_gripper(self, button2_pressed: bool) -> int:
        if button2_pressed and not self._button2_was_pressed:
            self._gripper_open = not self._gripper_open
        self._button2_was_pressed = button2_pressed
        return int(GripperAction.OPEN if self._gripper_open else GripperAction.CLOSE)

    def _compute_action(
        self, pos: np.ndarray, orn: np.ndarray, should_move: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (pos_delta [-1,1], orn_absolute_offset [rad]).

        Position is a per-frame delta (unchanged from before).
        Orientation is the *total* offset from the reference gimbal
        reading, accumulated across clutch segments.  This avoids any
        drift because we never round-trip through quaternion composition
        inside the teleoperator.
        """
        pos_delta = np.zeros(3, dtype=np.float64)

        orn_scale = np.array([self.config.roll_scale, 1.0, 1.0], dtype=np.float64)

        if should_move:
            # --- Position: per-frame delta ---
            if self._prev_pos is not None:
                raw_pos = pos - self._prev_pos
                pos_delta = _TOUCH_TO_ROBOT_POS @ raw_pos * self.config.position_scale
                pos_delta = np.clip(pos_delta, -1.0, 1.0)
            self._prev_pos = pos.copy()

            # --- Orientation: absolute offset from reference ---
            if self._ref_orn is None:
                self._ref_orn = orn.copy()
            segment_offset = _TOUCH_TO_ROBOT_ORN @ (orn - self._ref_orn) * orn_scale
            orn_total = self._orn_accumulator + segment_offset
        else:
            # Clutching / not intervening: freeze output, save state
            if self._ref_orn is not None:
                segment_offset = _TOUCH_TO_ROBOT_ORN @ (orn - self._ref_orn) * orn_scale
                self._orn_accumulator = self._orn_accumulator + segment_offset
                self._ref_orn = None
            orn_total = self._orn_accumulator
            self._prev_pos = None

        return pos_delta, orn_total

    # ------------------------------------------------------------------

    def get_action(self) -> dict[str, Any]:
        if not self._connected:
            # Device disconnected: return zero action (policy takes over)
            action_dict = {
                "delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0,
                "delta_rx": 0.0, "delta_ry": 0.0, "delta_rz": 0.0,
            }
            if self.config.use_gripper:
                action_dict["gripper"] = int(GripperAction.STAY)
            return action_dict

        pos, orn, button = self._read_device()
        button1_pressed = bool(button & 1)
        button2_pressed = bool(button & 2)

        gripper = self._update_gripper(button2_pressed)

        if self.config.clutch_mode:
            should_move = not button1_pressed
        else:
            should_move = button1_pressed

        pos_delta, orn_offset = self._compute_action(pos, orn, should_move)

        action_dict: dict[str, Any] = {
            "delta_x": float(pos_delta[0]),
            "delta_y": float(pos_delta[1]),
            "delta_z": float(pos_delta[2]),
            "delta_rx": float(orn_offset[0]),
            "delta_ry": float(orn_offset[1]),
            "delta_rz": float(orn_offset[2]),
        }
        if self.config.use_gripper:
            action_dict["gripper"] = gripper
        return action_dict

    def get_teleop_events(self) -> dict[str, Any]:
        if self.config.clutch_mode:
            is_intervention = True
        else:
            is_intervention = bool(self._last_button & 1)

        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: False,
            TeleopEvents.SUCCESS: False,
            TeleopEvents.RERECORD_EPISODE: False,
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        mod = self._touch_module
        if mod is not None:
            try:
                mod.stopScheduler()
                if self.config.device_name == "right":
                    mod.closeTouch_right()
                else:
                    mod.closeTouch_left()
            except Exception as e:
                log.warning(f"Touch disconnect error (device already removed?): {e}")
        self._connected = False
        log.info("Touch device disconnected")
