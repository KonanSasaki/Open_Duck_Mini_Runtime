"""SSH/terminal-friendly keyboard controller for Open Duck Mini.

Controls
--------
W/S : forward/backward
A/D : move left/right
Q/E : turn left/right
K   : stop immediately
Space: pause/unpause
R   : toggle sprint mode
+/- : increase/decrease phase-frequency offset
X   : toggle projector
B   : play a sound
Ctrl+C: exit
"""

from __future__ import annotations

import atexit
import select
import sys
import termios
import time
import tty

import numpy as np

from mini_bdx_runtime.buttons import Buttons


class KeyboardController:
    """Read commands from the terminal without requiring a joystick.

    This works in a normal local terminal and through ``ssh -t``. Movement is
    stopped automatically when no movement key has been received for
    ``command_timeout`` seconds.
    """

    X_SPEED = 0.15
    Y_SPEED = 0.20
    YAW_SPEED = 1.00

    def __init__(self, command_freq: float = 20, command_timeout: float = 0.60):
        del command_freq  # Kept for API compatibility with XBoxController.

        if not sys.stdin.isatty():
            raise RuntimeError(
                "Keyboard control requires an interactive terminal. "
                "Run from a terminal, or connect with: ssh -t <user>@<robot-ip>"
            )

        self.command_timeout = float(command_timeout)
        self.last_commands = np.zeros(7, dtype=float)
        self.last_movement_time = 0.0
        self.buttons = Buttons()
        self.sprint_enabled = False
        self._closed = False

        self._stdin_fd = sys.stdin.fileno()
        self._old_terminal_settings = termios.tcgetattr(self._stdin_fd)
        tty.setcbreak(self._stdin_fd)
        atexit.register(self.stop)

        print(
            "Keyboard control: W/S forward/back, A/D left/right, "
            "Q/E turn, K stop, Space pause, R sprint, +/- frequency, Ctrl+C exit"
        )

    def _read_keys(self) -> list[str]:
        keys: list[str] = []
        while select.select([sys.stdin], [], [], 0.0)[0]:
            keys.append(sys.stdin.read(1))
        return keys

    def _set_motion(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> None:
        self.last_commands[0] = x
        self.last_commands[1] = y
        self.last_commands[2] = yaw
        self.last_movement_time = time.monotonic()

    def _stop_motion(self) -> None:
        self.last_commands[:3] = 0.0
        self.last_movement_time = 0.0

    def get_last_command(self):
        # These emulate one-cycle Xbox button presses.
        pulse_a = False
        pulse_b = False
        pulse_x = False
        pulse_y = False
        pulse_dpad_up = False
        pulse_dpad_down = False

        for raw_key in self._read_keys():
            key = raw_key.lower()

            if key == "w":
                self._set_motion(x=self.X_SPEED)
            elif key == "s":
                self._set_motion(x=-self.X_SPEED)
            elif key == "a":
                self._set_motion(y=self.Y_SPEED)
            elif key == "d":
                self._set_motion(y=-self.Y_SPEED)
            elif key == "q":
                self._set_motion(yaw=self.YAW_SPEED)
            elif key == "e":
                self._set_motion(yaw=-self.YAW_SPEED)
            elif key == "k":
                self._stop_motion()
            elif raw_key == " ":
                pulse_a = True
                self._stop_motion()
            elif key == "r":
                self.sprint_enabled = not self.sprint_enabled
                print(f"Sprint: {'ON' if self.sprint_enabled else 'OFF'}")
            elif raw_key in ("+", "="):
                pulse_dpad_up = True
            elif raw_key in ("-", "_"):
                pulse_dpad_down = True
            elif key == "x":
                pulse_x = True
            elif key == "b":
                pulse_b = True
            elif key == "y":
                pulse_y = True

        if (
            self.last_movement_time > 0.0
            and time.monotonic() - self.last_movement_time > self.command_timeout
        ):
            self._stop_motion()

        self.buttons.update(
            pulse_a,
            pulse_b,
            pulse_x,
            pulse_y,
            self.sprint_enabled,  # Emulate holding LB.
            False,
            pulse_dpad_up,
            pulse_dpad_down,
        )

        return np.around(self.last_commands, 3), self.buttons, 0.0, 0.0

    def stop(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_motion()
        termios.tcsetattr(
            self._stdin_fd,
            termios.TCSADRAIN,
            self._old_terminal_settings,
        )
