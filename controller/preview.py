import numpy as np
from typing import List, Tuple
from collections import deque

from utils.data import LatDebugInfo, VehicleState
from controller.control_utils import cal_error_point_fun

class Preview:
    """
    Trajectory-based dual-point preview control.

    Near preview point: the point on the desired trajectory at t seconds ahead,
    used for lateral error and heading error calculation.

    Far preview point: the point on the desired trajectory at t' seconds ahead,
    also used for lateral and heading error calculation.
    """
    def __init__(self, config):
        self._load_config(config)
        self._error_buffer = deque(maxlen=10)
        self._time = 0

    def _load_config(self, config):
        self._dt = config.dt
        self._near = config.near
        self._far = config.far
        self._delay = config.delay
        self._kp = config.K_P
        self._ki = config.K_I
        self._kd = config.K_D
    
    def control(self, path_list: List, vehicle_state: VehicleState, e_rr: Tuple, min_index: int):
        """
        Compute steering command using dual-point preview and PID control

        Args:
            path_list (List): reference path [(x, y, heading, curvature), ...]
            vehicle_state (VehicleState):current vehicle state
            e_rr (Tuple): tracking error
            min_index (int): index of the matched point on the path

        Returns:
            steer: steering command
            LatDebugInfo: lateral debug information
        """
        # Compute near and far preview point errors
        error_near, error_far = self._cal_near_far_error_fun(path_list, min_index, vehicle_state)

        # PID control on lateral error (near) and heading error (far)
        omega1 = self._pid_fun(error_near[0])
        omega2 = self._pid_fun(error_far[2])
        steer = -(omega1 + omega2) / 2

        # Implement control delay
        if self._time == 0:
            self._pre_steer = steer
            self._time = self._delay
        else:
            steer = self._pre_steer
        self._time -= 1
        return steer, LatDebugInfo()

    def _cal_near_far_error_fun(self, path_list, min_index, vehicle_state):
        """
        Calculate errors for near and far preview points based on predicted position
        """
        path_length = len(path_list)
        if path_length - (min_index + self._near) <= 0:
            near_point = path_list[-1]
        else:
            near_point = path_list[min_index + self._near]

        if path_length - (min_index + self._far) <= 0:
            far_point = path_list[-1]
        else:
            far_point = path_list[min_index + self._far]

        # Compute error between predicted vehicle position and preview points
        error_near = cal_error_point_fun(near_point, vehicle_state)
        error_far = cal_error_point_fun(far_point, vehicle_state)

        return error_near, error_far

    def _pid_fun(self, error):
        self._error_buffer.append(error)
        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._kp * error) + (self._kd * _de) + (self._ki * _ie), -1.0, 1.0)

    def set_delay(self, delay):
        self._delay = delay
