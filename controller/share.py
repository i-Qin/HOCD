import numpy as np

from rl.ppo import PPO
from rl.tvdc import TVDC
from rl.rl_utils import Normalization
from controller.g29 import G29
from controller.mpc import MPC
from controller.preview import Preview
from utils.data import LatDebugInfo


DRIVER_CONTROLLER_REGISTRY = {
    "Preview": Preview,
    "G29": G29,
}

VEHICLE_CONTROLLER_REGISTRY = {
    "MPC": MPC
}

AUTHORITY_MODEL_REGISTRY = {
    "PPO": PPO,
    "TVDC": TVDC
}

class SharedControl:
    def __init__(self, config):
        self._load_config(config)
        self._setup()

    def _load_config(self, config):
        self._cfg = config
        self._sc_cfg = config.controller.lateral.SharedControl
        self._lat_cfg = config.controller.lateral

        self._dt = config.controller.dt
        self._filter = config.controller.lateral.filter

        self._train = config.rl.train

        if not self._train:
            self._driver_state = self._sc_cfg.driver_state
            self._authority = self._sc_cfg.driver_state
    
    def _setup(self):
        self.driver_control = self._build_component(self._sc_cfg.driver_model, DRIVER_CONTROLLER_REGISTRY)
        self.vehicle_control = self._build_component(self._sc_cfg.vehicle_model, VEHICLE_CONTROLLER_REGISTRY)
        if not self._train:
            self._authority_model = self._build_component(self._sc_cfg.authority_model, AUTHORITY_MODEL_REGISTRY, use_rl_cfg=True)

        self._state_norm = Normalization(shape=6)

        self.steer_driver = 0
        self.steer_vehicle = 0
        self.authority = 0


    def _build_component(self, ctrl_type, registry, use_rl_cfg=False):

        assert ctrl_type in registry, f"Unknown controller: {ctrl_type}"

        ctrl_class = registry[ctrl_type]

        if use_rl_cfg:
            ctrl_param = self._cfg.rl
        else:
            ctrl_param = self._lat_cfg.get(ctrl_type, {})
            ctrl_param["dt"] = self._dt

        return ctrl_class(ctrl_param)


    def control(self, path_list, state, err, min_index, **kwargs):

        steer_vehicle, info_vehicle = self.vehicle_control.control(path_list, state, err, min_index)
        if self._filter:
            steer_vehicle = self._steer_filter_v(steer_vehicle, t=2)

        steer_driver, info_driver = self.driver_control.control(path_list, state, err, min_index)
        if self._filter:
            steer_driver = self._steer_filter_d(steer_driver, t=2)

        e_d = err[0]
        e_fi = err[2]
        ay = state.ay

        # Observation space
        state = np.array([self._driver_state, e_d, e_fi, ay, steer_driver, steer_vehicle])  # -ay

        action = kwargs.get("action", None)
        if self._train:
            if action is None:
                raise ValueError("SharedControl training mode requires 'action'")
        else:
            action = self._load_action(state)
        if self._filter:
            action = self._authority_filter(action, t=2)

        steer = action * steer_vehicle + (1-action) * steer_driver

        info = LatDebugInfo(
            authority=action,
            steer_driver=steer_driver,
            steer_vehicle=steer_vehicle
        )

        info.update(info_vehicle)
        info.update(info_driver)

        return steer, info
    
    def _load_action(self, state):
        if self._sc_cfg.authority_model == "PPO":
            if self._cfg.rl.use_state_norm:
                state = self._state_norm(state)
            action = self._authority_model.choose_action(state, train=False)  # 车辆权重
            action = float(action)
        elif self._sc_cfg.authority_model == "TVDC":
            action = self._authority_model.choose_action(state, train=False)  # 车辆权重
        else:
            action = 0.5
        return action
    
    def set_driver_state(self, state):
        self._driver_state = state

    def _authority_filter(self, x, t=2):
        out = np.exp(-1/t) * self.authority + (1 - np.exp(-1/t)) * x
        self.authority = out
        return out

    def _steer_filter_d(self, current_steering, t=2):
        steer = np.exp(-1/t)*self.steer_driver + (1 - np.exp(-1/t))*current_steering
        self.steer_driver = steer
        return steer

    def _steer_filter_v(self, current_steering, t=2):
        steer = np.exp(-1/t)*self.steer_vehicle + (1 - np.exp(-1/t))*current_steering
        self.steer_vehicle = steer
        return steer


