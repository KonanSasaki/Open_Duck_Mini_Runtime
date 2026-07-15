import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


PACKAGE_ROOT = Path(__file__).parents[1] / "mini_bdx_runtime"
sys.path.insert(0, str(PACKAGE_ROOT))

# Keep the tests independent of a locally installed RustyPot wheel and hardware.
fake_rustypot = types.ModuleType("rustypot")
fake_rustypot.Sts3215PyController = MagicMock()
sys.modules["rustypot"] = fake_rustypot
sys.modules["onnxruntime"] = types.ModuleType("onnxruntime")

from mini_bdx_runtime.rustypot_position_hwi import HWI


JOINT_NAMES = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]


class DummyDuckConfig:
    def __init__(self):
        self.joints_offset = {name: 0.0 for name in JOINT_NAMES}


class HwiRustyPot15Tests(unittest.TestCase):
    def setUp(self):
        self.controller = MagicMock()
        fake_rustypot.Sts3215PyController.reset_mock()
        fake_rustypot.Sts3215PyController.return_value = self.controller
        self.hwi = HWI(
            DummyDuckConfig(),
            usb_port="/dev/test-motors",
            baudrate=500_000,
            timeout=0.25,
        )

    def test_constructs_sts3215_controller(self):
        fake_rustypot.Sts3215PyController.assert_called_once_with(
            serial_port="/dev/test-motors",
            baudrate=500_000,
            timeout=0.25,
        )

    def test_writes_pid_gains_with_v15_register_api(self):
        self.hwi.set_kps(np.ones(len(self.hwi.joints)) * 32)
        self.controller.sync_write_p_coefficient.assert_called_once_with(
            self.hwi.motor_ids, [32] * len(self.hwi.joints)
        )

        self.hwi.set_kds([1] * len(self.hwi.joints))
        self.controller.sync_write_d_coefficient.assert_called_once_with(
            self.hwi.motor_ids, [1] * len(self.hwi.joints)
        )

        self.hwi.set_kp(20, np.float64(8))
        self.assertEqual(
            self.controller.sync_write_p_coefficient.call_args_list[-1].args,
            ([20], [8]),
        )

    def test_rejects_gains_outside_u8_range(self):
        with self.assertRaises(ValueError):
            self.hwi.set_kps([256] * len(self.hwi.joints))

    def test_uses_sync_position_apis(self):
        self.hwi.joints_offsets["left_hip_yaw"] = 0.1
        self.hwi.set_position("left_hip_yaw", 0.2)
        first_call = self.controller.sync_write_goal_position.call_args_list[0]
        self.assertEqual(first_call.args[0], [20])
        self.assertAlmostEqual(first_call.args[1][0], 0.3)

        self.hwi.set_position_all({"right_ankle": 0.4, "left_hip_yaw": 0.2})
        self.assertEqual(
            self.controller.sync_write_goal_position.call_args_list[-1].args,
            ([14, 20], [0.4, 0.30000000000000004]),
        )

    def test_reads_positions_and_speeds_with_sync_api(self):
        positions = [float(i) for i in range(len(self.hwi.joints))]
        speeds = [float(i) / 10 for i in range(len(self.hwi.joints))]
        self.controller.sync_read_present_position.return_value = positions
        self.controller.sync_read_present_speed.return_value = speeds

        np.testing.assert_array_equal(
            self.hwi.get_present_positions(), np.around(positions, 3)
        )
        np.testing.assert_array_equal(
            self.hwi.get_present_velocities(), np.around(speeds, 3)
        )
        self.controller.sync_read_present_position.assert_called_once_with(
            self.hwi.motor_ids
        )
        self.controller.sync_read_present_speed.assert_called_once_with(
            self.hwi.motor_ids
        )

        self.controller.sync_read_present_position.return_value = [1.25]
        self.assertEqual(self.hwi.get_motor_position(20), 1.25)
        self.controller.sync_read_present_position.assert_called_with([20])

    def test_writes_torque_enable_register(self):
        self.hwi.set_torque_enabled(20, True)
        self.controller.sync_write_torque_enable.assert_called_once_with(
            [20], [True]
        )

        self.hwi.turn_off()
        self.assertEqual(
            self.controller.sync_write_torque_enable.call_args_list[-1].args,
            (self.hwi.motor_ids, [False] * len(self.hwi.joints)),
        )

    @patch("mini_bdx_runtime.rustypot_position_hwi.time.sleep")
    def test_turn_on_uses_sync_gain_writes(self, sleep):
        self.hwi.kps = [30] * len(self.hwi.joints)
        self.hwi.turn_on()

        self.assertEqual(
            self.controller.sync_write_p_coefficient.call_args_list[0].args,
            (self.hwi.motor_ids, [2] * len(self.hwi.joints)),
        )
        self.assertEqual(
            self.controller.sync_write_p_coefficient.call_args_list[-1].args,
            (self.hwi.motor_ids, [30] * len(self.hwi.joints)),
        )
        self.assertEqual(sleep.call_count, 2)


if __name__ == "__main__":
    unittest.main()
