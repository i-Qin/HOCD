import copy
import math
import numpy as np

from planner.global_plan import GlobalPlan
from planner.plan_utils import FrenetPath, QuarticPolynomial, QuinticPolynomial, distance_destination
from utils.data import PlanInfo, SimulatorObservation


class VehiclePlan:
    def __init__(self, config, map_info):
        self._load_config(config)
        self._setup(config, map_info)

    def _load_config(self, config):
        cfg = config.planner
        self._debug = cfg.debug

        self._MAX_SPEED = cfg.MAX_SPEED / 3.6
        self._MAX_ACCEL = cfg.MAX_ACCEL
        self._MAX_CURVATURE = cfg.MAX_CURVATURE

        self._MAX_ROAD_WIDTH = cfg.MAX_ROAD_WIDTH
        self._D_ROAD_W = cfg.D_ROAD_W

        self._DT = cfg.DT
        self._MAX_T = cfg.MAX_T
        self._MIN_T = cfg.MIN_T

        self._TARGET_SPEED = cfg.TARGET_SPEED / 3.6
        self._D_T_S = cfg.D_T_S / 3.6
        self._N_S_SAMPLE = cfg.N_S_SAMPLE

        self._ROBOT_RADIUS = cfg.ROBOT_RADIUS

        self._K_J = cfg.K_J
        self._K_T = cfg.K_T
        self._K_D = cfg.K_D
        self._K_LAT = cfg.K_LAT
        self._K_LON = cfg.K_LON
    
    def _setup(self, config, map_info):
        town_map, topology, origin, destination = map_info
        self._global_planner = GlobalPlan(config, town_map, topology)
        self._global_path, self._csp, _, _, _ = self._global_planner.plan(origin, destination)
        self._destination = destination
        self._plan_start_point()

    def _plan_start_point(self):
        """
        Initialize planning state in Frenet coordinates.
        """
        self._s0 = 0        # longitudinal position
        self._c_d = 0       # lateral offset
        self._c_d_d = 0     # lateral velocity
        self._c_d_dd = 0    # lateral acceleration
        self._c_speed = 0   # longitudinal speed
        self._c_accel = 0   # longitudinal acceleration

    def run_step(self, env: SimulatorObservation, human_intention=''):
        # 1. Detect nearby obstacles
        obs_list = self.detect_obstacle(env)

        # 2. Compute optimal local trajectory using Frenet planner
        path = self._frenet_optimal_planning(obs_list, human_intention)

        # 3. Extract trajectory (x, y, yaw, curvature) and speed profile
        path_list = list(zip(path.x, path.y, path.yaw, path.c))
        speed_list = path.s_d

        # 4. Update planning start state for next cycle
        self._update_plan_start_point(path)

        return PlanInfo(
            global_path=self._global_path,
            local_path=path_list,
            speed=speed_list,
            distance=distance_destination(env.ego, self._destination)
        )

    def _update_plan_start_point(self, path):
        self._s0 = path.s[1]
        self._c_d = path.d[1]
        self._c_d_d = path.d_d[1]
        self._c_d_dd = path.d_dd[1]
        self._c_speed = path.s_d[1]
        self._c_accel = path.s_dd[1]

    def _frenet_optimal_planning(self, ob, human_intention):
        # 1. Generate candidate Frenet trajectories
        fplist = self._calc_frenet_paths(human_intention)

        # 2. Transform trajectories into Cartesian coordinates
        all_fplist = self._calc_global_paths(fplist)

        # 3. Remove trajectories that collide with obstacles
        fplist_without_obs = self._check_paths(all_fplist, ob)

        # 4. Select minimum cost trajectory
        min_cost = float("inf")
        best_path = None
        for fp in fplist_without_obs:
            if min_cost >= fp.cf:
                min_cost = fp.cf
                best_path = fp

        return best_path

    def _calc_frenet_paths(self, human_intention):
        """
        Generate candidate trajectories in Frenet space.
        """
        frenet_paths = []
        
        # Encode human lane-change intention
        if human_intention:
            if human_intention == 'LEFT':
                human_intention_change = 3
            elif human_intention == 'RIGHT':
                human_intention_change = -3
        else:
            human_intention_change = 0
        print("Human intention:",human_intention, human_intention_change)

        # ===== Lateral sampling =====
        for di in np.arange(-self._MAX_ROAD_WIDTH, self._MAX_ROAD_WIDTH, self._D_ROAD_W): 
            for Ti in np.arange(self._MIN_T, self._MAX_T, self._DT): 
                fp = FrenetPath()

                lat_qp = QuinticPolynomial(self._c_d, self._c_d_d, self._c_d_dd, di, 0.0, 0.0, Ti)

                fp.t = [t for t in np.arange(0.0, Ti, self._DT)]
                fp.d = [lat_qp.calc_point(t) for t in fp.t]
                fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
                fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
                fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

                 # ===== Longitudinal sampling =====
                for tv in np.arange(
                    self._TARGET_SPEED - self._D_T_S * self._N_S_SAMPLE,
                    self._TARGET_SPEED + self._D_T_S * self._N_S_SAMPLE,
                    self._D_T_S,
                ):
                    tfp = copy.deepcopy(fp)

                    lon_qp = QuarticPolynomial(
                        self._s0, self._c_speed, self._c_accel, tv, 0.0, Ti
                    )

                    tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                    tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                    tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                    tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                    # === Cost computation ===
                    Jp = sum(np.power(tfp.d_ddd, 2)) 
                    Js = sum(np.power(tfp.s_ddd, 2)) 

                    # Speed tracking error
                    ds = (self._TARGET_SPEED - tfp.s_d[-1]) ** 2
                    
                    # Lateral and longitudinal cost
                    tfp.cd = self._K_J * Jp + self._K_T * Ti + self._K_D * (tfp.d[-1] + human_intention_change) ** 2 
                    tfp.cv = self._K_J * Js + self._K_T * Ti + self._K_D * ds
                    tfp.cf = self._K_LAT * tfp.cd + self._K_LON * tfp.cv

                    frenet_paths.append(tfp)

        return frenet_paths

    def _calc_global_paths(self, fplist):
        """
        Convert Frenet trajectories into Cartesian coordinates.
        """
        for fp in fplist:
            for i in range(len(fp.s)):
                ix, iy = self._csp.calc_position(fp.s[i])
                if ix is None:
                    break
                i_yaw = self._csp.calc_yaw(fp.s[i])
                di = fp.d[i]
                fx = ix + di * math.cos(i_yaw + math.pi / 2.0)
                fy = iy + di * math.sin(i_yaw + math.pi / 2.0)
                fp.x.append(fx)
                fp.y.append(fy)

            # Compute yaw and arc length
            for i in range(len(fp.x) - 1):
                dx = fp.x[i + 1] - fp.x[i]
                dy = fp.y[i + 1] - fp.y[i]
                fp.yaw.append(math.atan2(dy, dx))
                fp.ds.append(math.hypot(dx, dy))

            fp.yaw.append(fp.yaw[-1])
            fp.ds.append(fp.ds[-1])

            # Compute curvature using yaw difference
            d_theta_ = np.diff(fp.yaw) 
            d_theta_pre = np.insert(d_theta_, 0, d_theta_[0])
            d_theta_aft = np.insert(d_theta_, -1, d_theta_[-1])
            d_theta = np.sin((d_theta_pre + d_theta_aft) / 2)
            fp.c = d_theta / fp.ds

        return fplist

    def _check_paths(self, fplist, ob):
        """
        Filter out invalid trajectories (collision check).
        """
        ok_ind = []
        for i, _ in enumerate(fplist):
            if not self._check_collision(fplist[i], ob):
                continue
            ok_ind.append(i)
        return [fplist[i] for i in ok_ind]

    def _check_collision(self, fp, ob):
        """
        Collision checking using circular robot model.
        """
        for i in range(len(ob)):
            d = [((ix - ob[i][0]) ** 2 + (iy - ob[i][1]) ** 2) for (ix, iy) in zip(fp.x, fp.y)]
            collision = any([di <= self._ROBOT_RADIUS**2 for di in d])
            if collision:
                return False
        return True

    def detect_obstacle(self, env: SimulatorObservation):
        """
        Detect nearby obstacles based on distance and relative position.
        """
        ego = env.ego
        neighbors = env.neighbors

        obs_info = []

        for neighbor in neighbors:
            distance = np.sqrt((neighbor.x - ego.x) ** 2 + (neighbor.y - ego.y) ** 2 + (neighbor.z - ego.z) ** 2)
            vec_ego_obs = np.array([neighbor.x - ego.x, neighbor.y - ego.y, neighbor.z - ego.z])

            long_dis = np.dot(ego.vec_forward, vec_ego_obs)

            lat_dis = np.dot(ego.vec_right, vec_ego_obs)

            if long_dis > 0 and distance < 70 and abs(lat_dis) < 3:
                obs_info.append((neighbor.x, neighbor.y))

        return obs_info
