import numpy as np

from controller.g29 import G29
from controller.pid import PID
from controller.mpc import MPC
from controller.preview import Preview
from controller.share import SharedControl
from controller.control_utils import cal_error_fun, cal_pre_vehicle_info
from utils.data import PlanInfo, ControlInfo, SimulatorObservation, LatDebugInfo

LAT_CONTROLLER_REGISTRY = {
    "MPC": MPC,
    "Preview": Preview,
    "SharedControl": SharedControl,
    "G29": G29,
}

LON_CONTROLLER_REGISTRY = {
    "PID": PID
}

class VehicleControl:
    def __init__(self, config):
        self._load_config(config)
        self._setup()

    def _load_config(self, config):
        cfg = config.controller
        self._lat_control_type = cfg.lateral.type
        self._max_throttle = cfg.max_throttle
        self._max_brake = cfg.max_brake
        self._max_steer = cfg.max_steer
        self._min_steer = cfg.min_steer
        self._pre_ts = cfg.pre_ts
        self._filter = cfg.lateral.filter
        self._cfg = config
    
    def _setup(self):
        self.lat_control = self._build_controller(self._cfg.controller.lateral, LAT_CONTROLLER_REGISTRY)
        self.lon_control = self._build_controller(self._cfg.controller.longitudinal, LON_CONTROLLER_REGISTRY)
        self.steer = 0.0
        self.e_rr = None

    def _build_controller(self, ctrl_cfg, registry):
        ctrl_type = ctrl_cfg.type
        assert ctrl_type in registry, f"Unknown controller: {ctrl_type}"

        ctrl_class = registry[ctrl_type]

        if ctrl_type == "SharedControl":
            ctrl_param = self._cfg
        else:
            ctrl_param = ctrl_cfg.get(ctrl_type, {})
            ctrl_param["dt"] = self._cfg.controller.dt

        return ctrl_class(ctrl_param)

    def run_step(self, env: SimulatorObservation, plan: PlanInfo, **kwargs):

        # ======= Lateral control =======

        # 1. Predict future vehicle state
        pre_vehicle_state = cal_pre_vehicle_info(env.ego, self._pre_ts)

        # 2. Compute tracking error relative to reference path
        err, min_index = cal_error_fun(plan.local_path, pre_vehicle_state)

        # 3. Compute steering using lateral controller                      
        current_steer, lat_debug = self.lat_control.control(plan.local_path, pre_vehicle_state, err, min_index, **kwargs)

        # Add error info for debugging/visualization
        lat_debug.update(LatDebugInfo(e_d=err[0], e_d_dao=err[1], e_fi=err[2], e_fi_dao=err[3]))

        # Optional steering smoothing filter
        if self._filter:
            current_steer = self._steer_filter(current_steer, t=2)

        # ======= Longitudinal control =======

        # Compute acceleration command based on current speed and target speed
        current_acceleration, lon_debug = self.lon_control.control(env.ego.vx, plan.speed[min_index])

        # ======= Apply actuator limits =======
        if current_steer >= 0: 
            steer = min(self._max_steer, current_steer) 
        else: 
            steer = max(self._min_steer, current_steer) 

        if current_acceleration >= 0: 
            throttle = min(self._max_throttle, current_acceleration) 
            brake = 0 
        else: 
            throttle = 0 
            brake = min(self._max_brake, abs(current_acceleration))

        return ControlInfo(
            steer=steer,
            throttle=throttle,
            brake=brake,
            lat=lat_debug,
            lon=lon_debug
        )
           
    def _steer_filter(self, current_steering, t=2):
        steer = np.exp(-1/t)*self.steer + (1 - np.exp(-1/t))*current_steering
        return steer




