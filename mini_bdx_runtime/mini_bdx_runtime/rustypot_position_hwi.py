import time

import numpy as np
import rustypot
from mini_bdx_runtime.duck_config import DuckConfig


class HWI:
    def __init__(
        self,
        duck_config: DuckConfig,
        usb_port: str = "/dev/ttyACM0",
        baudrate: int = 1_000_000,
        timeout: float = 1.0,
    ):

        self.duck_config = duck_config

        # Order matters here
        self.joints = {
            "left_hip_yaw": 20,
            "left_hip_roll": 21,
            "left_hip_pitch": 22,
            "left_knee": 23,
            "left_ankle": 24,
            "neck_pitch": 30,
            "head_pitch": 31,
            "head_yaw": 32,
            "head_roll": 33,
            # "left_antenna": None,
            # "right_antenna": None,
            "right_hip_yaw": 10,
            "right_hip_roll": 11,
            "right_hip_pitch": 12,
            "right_knee": 13,
            "right_ankle": 14,
        }

        self.zero_pos = {
            "left_hip_yaw": 0,
            "left_hip_roll": 0,
            "left_hip_pitch": 0,
            "left_knee": 0,
            "left_ankle": 0,
            "neck_pitch": 0,
            "head_pitch": 0,
            "head_yaw": 0,
            "head_roll": 0,
            # "left_antenna":0,
            # "right_antenna":0,
            "right_hip_yaw": 0,
            "right_hip_roll": 0,
            "right_hip_pitch": 0,
            "right_knee": 0,
            "right_ankle": 0,
        }

        self.init_pos = {
            "left_hip_yaw": 0.002,
            "left_hip_roll": 0.053,
            "left_hip_pitch": -0.63,
            "left_knee": 1.368,
            "left_ankle": -0.784,
            "neck_pitch": 0.0,
            "head_pitch": 0.0,
            "head_yaw": 0,
            "head_roll": 0,
            # "left_antenna": 0,
            # "right_antenna": 0,
            "right_hip_yaw": -0.003,
            "right_hip_roll": -0.065,
            "right_hip_pitch": 0.635,
            "right_knee": 1.379,
            "right_ankle": -0.796,
        }

        self.joints_offsets = self.duck_config.joints_offset

        self.kps = np.ones(len(self.joints)) * 32  # default kp
        self.kds = np.ones(len(self.joints)) * 0  # default kd
        self.low_torque_kps = np.ones(len(self.joints)) * 2

        self.io = rustypot.Sts3215PyController(
            serial_port=usb_port,
            baudrate=baudrate,
            timeout=timeout,
        )

    @property
    def motor_ids(self):
        return list(self.joints.values())

    @staticmethod
    def _gain_values(gains):
        """Convert NumPy/Python numeric values to RustyPot's u8 register values."""
        values = [int(gain) for gain in gains]
        if any(value < 0 or value > 255 for value in values):
            raise ValueError("Motor gains must be between 0 and 255")
        return values

    def set_kps(self, kps):
        self.kps = self._gain_values(kps)
        self.io.sync_write_p_coefficient(self.motor_ids, self.kps)

    def set_kds(self, kds):
        self.kds = self._gain_values(kds)
        self.io.sync_write_d_coefficient(self.motor_ids, self.kds)

    def set_kp(self, id, kp):
        self.io.sync_write_p_coefficient([id], self._gain_values([kp]))

    def set_torque_enabled(self, id, enabled):
        self.io.sync_write_torque_enable([id], [bool(enabled)])

    def get_motor_position(self, id):
        return self.io.sync_read_present_position([id])[0]

    def set_motor_position(self, id, position):
        self.io.sync_write_goal_position([id], [float(position)])

    def turn_on(self):
        self.io.sync_write_p_coefficient(
            self.motor_ids, self._gain_values(self.low_torque_kps)
        )
        print("turn on : low KPS set")
        time.sleep(1)

        self.set_position_all(self.init_pos)
        print("turn on : init pos set")

        time.sleep(1)

        self.io.sync_write_p_coefficient(
            self.motor_ids, self._gain_values(self.kps)
        )
        print("turn on : high kps")

    def turn_off(self):
        self.io.sync_write_torque_enable(
            self.motor_ids, [False] * len(self.motor_ids)
        )

    def set_position(self, joint_name, pos):
        """
        pos is in radians
        """
        id = self.joints[joint_name]
        pos = pos + self.joints_offsets[joint_name]
        self.io.sync_write_goal_position([id], [float(pos)])

    def set_position_all(self, joints_positions):
        """
        joints_positions is a dictionary with joint names as keys and joint positions as values
        Warning: expects radians
        """
        ids = [self.joints[joint] for joint in joints_positions]
        positions = [
            float(position + self.joints_offsets[joint])
            for joint, position in joints_positions.items()
        ]

        self.io.sync_write_goal_position(
            ids,
            positions,
        )

    def get_present_positions(self, ignore=[]):
        """
        Returns the present positions in radians
        """

        try:
            present_positions = self.io.sync_read_present_position(self.motor_ids)
        except Exception as e:
            print(e)
            return None

        present_positions = [
            pos - self.joints_offsets[joint]
            for joint, pos in zip(self.joints.keys(), present_positions)
            if joint not in ignore
        ]
        return np.array(np.around(present_positions, 3))

    def get_present_velocities(self, rad_s=True, ignore=[]):
        """
        Returns the present velocities in rad/s (default) or rev/min
        """
        try:
            present_velocities = self.io.sync_read_present_speed(self.motor_ids)
        except Exception as e:
            print(e)
            return None

        present_velocities = [
            vel
            for joint, vel in zip(self.joints.keys(), present_velocities)
            if joint not in ignore
        ]

        return np.array(np.around(present_velocities, 3))
