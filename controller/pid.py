import numpy as np
from collections import deque

from utils.data import LonDebugInfo

class PID:

    def __init__(self, config):
        self._load_config(config)
        self._error_buffer = deque(maxlen=60)  
        self._error_threshold = 1

    def _load_config(self, config):
        self._dt = config.dt
        self._kp = config.K_P
        self._ki = config.K_I
        self._kd = config.K_D

    def control(self, cur_speed, target_speed):
        error = target_speed - cur_speed 
        self._error_buffer.append(error)  

        if len(self._error_buffer) >= 2:
            integral_error = sum(self._error_buffer) * self._dt
            differential_error = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
        else:
            integral_error = 0.0
            differential_error = 0.0

        throttle = np.clip((self._kp * error) + (self._kd * differential_error) + (self._ki * integral_error), -1.0, 1.0)

        return throttle, LonDebugInfo(
            cur_speed=cur_speed,
            target_speed=target_speed
        )


