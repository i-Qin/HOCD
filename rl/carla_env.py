import gym
import carla
import pickle
import random
import numpy as np

from simulator.simulator import CarlaSimulator
from simulator.simulator_utils import get_vehicle_state
from planner.global_plan import GlobalPlan
from planner.plan_utils import distance_destination
from controller.pid import PID
from controller.share import SharedControl
from controller.control_utils import cal_error_fun, cal_pre_vehicle_info


class CarlaEnv:
    def __init__(self, config):
        self._load_config(config)
        self._setup(config)

    def _load_config(self, config):
        self.max_episode_steps = config.rl.max_episode_steps
        
        self._target_speed = config.planner.TARGET_SPEED / 3.6
        self._random = config.planner.random
        
        self._dt = config.controller.dt
        self._pre_ts = config.controller.pre_ts
        self._max_throttle = config.controller.max_throttle
        self._max_brake = config.controller.max_brake
        self._max_steer = config.controller.max_steer
        self._min_steer = config.controller.min_steer

        lon_cfg = config.controller.longitudinal.PID
        lon_cfg.dt = self._dt

        self.cfg = config
        self.lon_cfg = lon_cfg
    
    def _setup(self, config):

         # Initialize CARLA simulator and connect
        self._sim = CarlaSimulator(config)

        # Load feasible routes or generate a fixed route
        if self._random:
            with open("utils/feasible_routes.pkl", "rb") as f:
                self._routes = pickle.load(f)
        else:
            town_map, topology, start, end = self._sim.get_map_info()
            self._destination = end
            self._path, _, _, _, _ = GlobalPlan(config, town_map, topology).plan(start, end)
            self.start_idx = self._sim.start_point

        # Initialize controllers
        self._lat_control = SharedControl(self.cfg)
        self._lon_control = PID(self.lon_cfg)
        
        # Define action and observation space
        self.action_space = gym.spaces.Box(np.float32(0.0), np.float32(1.0), (1,))
        self.observation_space = gym.spaces.Box(np.float32(np.array([0.0, -5.0, -1.0, -20.0, -1.0, -1.0])),
                                                np.float32(np.array([1.0, 5.0, 1.0, 20.0, 1.0, 1.0])),
                                                (6,))
        # Initialize variables for comfort calculation
        self.last_ay = 0
        self.last_action = 0

    def reset(self):
        # Reset time step counter
        self.time_step = 0

        # Randomly select a route if enabled
        if self._random:
            route = random.choice(self._routes)

            path, start_idx, end_idx = route["x_y_heading_k_list"], route["start_idx"], route["end_idx"]
            end = self._sim.all_default_spawn[end_idx].location
            self._destination = end
            self.start_idx = start_idx
            self._path = path

        # Respawn ego vehicle
        self._sim._spawn_ego_vehicle(self.start_idx)

        # Create collision sensor
        self._sim.generate_collision_sensor()

        # Randomly assign driver state
        self.ability = np.random.choice(np.array([0.0, 0.5, 1.0]))
        state_index = {0.0: 1, 0.5: 60, 1.0: 120}
        self._lat_control.driver_control.set_delay(state_index[self.ability])
   
        # Update state
        state = self._get_state()
        
        return state

    def step(self, action):

        self.time_step += 1

        self.action = float(action)

         # ====== Lateral control ======
        steer_driver, steer_vehicle = self.state[4], self.state[5]
        current_steer = action * steer_vehicle + (1-action) * steer_driver

        control = carla.VehicleControl()
        control.hand_brake = False
        control.manual_gear_shift = False
        control.gear = 1

        if current_steer >= 0:
            steer = min(self._max_steer, current_steer)
        else:
            steer = max(self._min_steer, current_steer)
        control.steer = steer

        self.steer_driver = steer_driver
        self.steer_vehicle = steer_vehicle
        self.steer = steer

        # ====== Longitudinal control ======
        current_acceleration, _ = self._lon_control.control(self.ego_state.vx, self._target_speed)

        if current_acceleration >= 0:
            control.throttle = min(self._max_throttle, current_acceleration)
            control.brake = 0
        else:
            control.throttle = 0
            control.brake = min(self._max_brake, abs(current_acceleration)) 

        # ====== Apply control and update ======
        self._sim.ego_vehicle.apply_control(control)
        self._sim.world.tick()

        next_state = self._get_state()
        reward = self._get_reward()
        done = self._terminal()
        truncated = self._truncated()

        return next_state, reward, done, truncated, {}

    def render(self):
        # Top-down view using spectator camera
        spectator = self._sim.world.get_spectator()
        transform = carla.Transform(self._sim.ego_vehicle.get_location() + carla.Location(z=80), carla.Rotation(pitch=-90))
        spectator.set_transform(transform)

    def close(self):
        pass

    def _get_state(self):
        """
        Get current state: [ability, lateral error, heading error, lateral acceleration, driver steering, vehicle steering]
        """
        # Get current environment state
        self.ego_state = get_vehicle_state(self._sim.ego_vehicle)
        ay = self.ego_state.ay

        # Predict future vehicle state
        pre_vehicle_state = cal_pre_vehicle_info(self.ego_state, self._pre_ts)

        # Compute tracking errors based on predicted state
        err, min_index = cal_error_fun(self._path, pre_vehicle_state)
        e_d = err[0]
        e_fi = err[2]
        self.err = err

        # Compute steering from vehicle and driver controllers
        steer_vehicle, _ = self._lat_control.vehicle_control.control(self._path, pre_vehicle_state, err, min_index)
        steer_driver, _ = self._lat_control.driver_control.control(self._path, pre_vehicle_state, err, min_index)

        state = np.array([self.ability, e_d, e_fi, ay, steer_driver, steer_vehicle])

        self.state = state

        return state
    
    def _get_reward(self):

        # ========== 1 Safety ==========
        collision = len(self._sim.collision_hist) > 0
        r_safe = -200 if collision else 0

        # ========== 2 Tracking  ==========
        err_d = abs(self.err[0])
        err_yaw = abs(self.err[2])
        r_tracking = -2 * (err_d + err_yaw)

        # ========== 3 Comfort  ==========
        ay = abs(self.state[3])
        jerk = abs((ay-self.last_ay) / self._dt)
        action_ratio = abs(self.action - self.last_action)
        self.last_ay = ay
        self.last_action = self.action
        r_comfort = -(ay + 0.01*jerk + action_ratio)

        # ========== 4 Human-machine conflict ==========
        conflict_1 = abs(self.steer_driver - self.steer) 
        conflict_2 = abs(self.ability-self.action)
        r_conflict = -(conflict_1 + conflict_2)

        reward = r_safe + 2*r_tracking + r_comfort + r_conflict

        return reward

    def _terminal(self):
        """
        Episode termination: Reaching destination / Collision
        """
        # Reach destination
        dis = distance_destination(self.ego_state, self._destination)
        if dis < 2:
            self._sim.ego_vehicle.apply_control(carla.VehicleControl(steer=0, throttle=0, brake=1))
            self._sim.destroy_actors()
            return True
        
        # Collision
        elif len(self._sim.collision_hist) > 0:
            print('Collision occurred!')
            self._sim.destroy_actors()
            return True
        else:
            return False

    def _truncated(self):
        """
        Terminate if exceeding max episode length
        """
        if self.time_step > self.max_episode_steps:
            print('Max episode steps exceeded!')
            self._sim.destroy_actors()
            return True
        else:
            return False






