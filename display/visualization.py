import carla
from dataclasses import asdict
from collections import defaultdict
from matplotlib import pyplot as plt

from utils.data import SimulatorObservation, PlanInfo, ControlInfo

class Visualization:
    def __init__(self, config, sim):
        self._sim = sim
        self._load_config(config)

        self._data = defaultdict(list)
    
    def _load_config(self, config):
        cfg = config.controller.lateral
        self._lat_control_type = cfg.type
        self._authority_type = cfg.SharedControl.authority_model
        self._debug = config.planner.debug


    def run_step(self, env: SimulatorObservation, plan: PlanInfo, ctrl: ControlInfo):
        ego_dict = asdict(env.ego)
        lon_dict = asdict(ctrl.lon)
        lat_dict = asdict(ctrl.lat)

        self._data["steer"].append(ctrl.steer)

        for k, v in ego_dict.items():
            self._data[f"{k}"].append(v)

        for k, v in lon_dict.items():
            self._data[f"{k}"].append(v)

        for k, v in lat_dict.items():
            self._data[f"{k}"].append(v)

        # === Debug visualization in simulator ===
        if self._debug:
            global_path = plan.global_path
            local_path = plan.local_path
            if global_path:
                self._debug_path(global_path, size=0.04, color=carla.Color(0, 0, 0), life_time=0.1)
            if local_path:
                self._debug_path(local_path, size=0.04, color=carla.Color(0, 255, 0), life_time=0.1)

    def _debug_path(self, path, size, color, life_time):
        """Render trajectory points in the CARLA world"""
        for point in path:
            x, y, z, _ = point
            location = carla.Location(x, y, z+2)
            self._sim.world.debug.draw_point(location, size, color, life_time)

    def show_result(self):
        # ===== Reaction time =====
        if self._data.get("reaction_time"):
            valid_times = [t for t in self._data["reaction_time"] if t is not None]
            if valid_times:
                reaction_time = round(sum(valid_times) / len(valid_times), 3)
                print('Average reaction time:', reaction_time)

        # ===== Plot vehicle state variables =====
        plt.figure(figsize=(12, 6))
        n = 2
        m = 4
        states = ['vy', 'ay', 'yaw', 'steer', 'e_d', 'e_fi', 'e_d_dao', 'e_fi_dao']
        for i in range(len(states)):
            plt.subplot(n, m, i+1)
            keyword = states[i]
            plt.plot(self._data[keyword])
            plt.title(keyword)
        plt.show()

        # ===== Shared control analysis =====
        if self._lat_control_type == 'SharedControl':
            plt.plot(self._data['steer_driver'], label='steer_driver')
            plt.plot(self._data['steer_vehicle'], label='steer_vehicle')
            plt.plot(self._data['steer'], label='steer')
            plt.legend()
            plt.show()
            plt.plot(self._data['authority'])
            plt.title('Vehicle Authority')
            plt.show()

        plt.plot(self._data['x'], self._data['y'])
        plt.show()

