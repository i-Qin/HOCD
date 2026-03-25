import cvxopt
import numpy as np
from typing import Tuple, List

from utils.data import LatDebugInfo, VehicleState

class MPC:
    def __init__(self, config):
        self._load_config(config)
        self._N = 6  
        self._P = 2  
        self._n = 4 
    
    def _load_config(self, config):
        self._dt = config.dt
        self._Q = config.Q
        self._R = config.R
        self._F = config.F

    def control(self, path_list: List, vehicle_state: VehicleState, e_rr: Tuple, min_index: int):
        """
        MPC-based lateral control
        Args:
            path_list (List): reference path [(x, y, heading, curvature), ...]
            vehicle_state (VehicleState):current vehicle state
            e_rr (Tuple): tracking error
            min_index (int): index of the matched point on the path

        Returns:
            steer: steering command
            LatDebugInfo: lateral debug information
        """
        # 1. Compute system matrices A, B, C based on vehicle state
        A, B, C = self._cal_A_B_C_fun(vehicle_state)

        # 2. Extract reference curvature at the projection point
        k_r = path_list[min_index][3]

        # 3. Solve K using the Riccati equation
        A_bar, B_bar, C_bar = self._cal_coefficient_of_discretion_fun(A, B, C, k_r, vehicle_state.vx)

        # 4. Solve MPC optimization problem
        steer = self._cal_control_para_fun(A_bar, B_bar, C_bar, e_rr)

        return steer, LatDebugInfo()

    def _cal_A_B_C_fun(self, state: VehicleState):
        """
        Compute continuous-time state-space matrices A, B, C
        based on vehicle parameters and longitudinal velocity Vx
        """
        vx = max(state.vx, 0.1)
        a = state.param.lf
        b = state.param.lr
        Cf = state.param.cf
        Cr = state.param.cr
        m = state.param.mass
        Iz = state.param.iz

        A = np.zeros(shape=(self._n, self._n), dtype="float64")
        B = np.zeros(shape=(self._n, 1), dtype="float64")
        C = np.zeros(shape=(self._n, 1), dtype="float64")

        A[0][1] = 1

        A[1][1] = (Cf + Cr) / (m * vx)
        A[1][2] = -(Cf + Cr) / m
        A[1][3] = (a * Cf - b * Cr) / (m * vx)

        A[2][3] = 1

        A[3][1] = (a * Cf - b * Cr) / (Iz * vx)
        A[3][2] = -(a * Cf - b * Cr) / Iz
        A[3][3] = (a * a * Cf + b * b * Cr) / (Iz * vx)

        B[1][0] = -Cf / m
        B[3][0] = -a * Cf / Iz

        C[1][0] = (a*Cf + b*Cr)/(m*vx) - vx
        C[3][0] = (a**2*Cf + b**2*Cr)/(Iz*vx)
        return A, B, C

    def _cal_coefficient_of_discretion_fun(self, A, B, C, k_r, vx):
        """
        Discretize continuous system
        """
        dt = self._dt 
        temp = np.linalg.inv(np.eye(4) - (dt * A) / 2)
        A_bar = temp @ (np.eye(4) + (dt * A) / 2)
        B_bar = temp @ B * dt
        C_bar = temp @ C * dt * k_r * vx
        return A_bar, B_bar, C_bar

    def _cal_control_para_fun(self, A_bar, B_bar, C_bar, e_rr):
        """
        Solve MPC optimization problem using quadratic programming
        """
        Q = np.eye(4)
        Q[0][0] = self._Q[0]
        Q[1][1] = self._Q[1]
        Q[2][2] = self._Q[2]
        Q[3][3] = self._Q[3]
        F = np.eye(4) * self._F
        R = self._R

        # Build prediction matrices M, C, Cc
        M = np.zeros(shape=((self._N+1)*self._n, self._n))
        M[0:self._n, :] = np.eye(self._n)
        for i in range(1, self._N + 1):
            M[i*self._n:(i+1)*self._n, :] = A_bar @ M[(i-1)*self._n:i*self._n, :]

        C = np.zeros(shape=((self._N + 1) * self._n, self._N * self._P))
        C[self._n:2*self._n, 0:self._P] = B_bar 
        for i in range(2, self._N + 1):
            C[i * self._n:(i + 1) * self._n, (i-1) * self._P:i * self._P] = B_bar
            for j in range(i-2, -1, -1):
                C[i*self._n:(i+1)*self._n, j*self._P:(j+1)*self._P] = \
                    A_bar @ C[i*self._n:(i+1)*self._n, (j+1)*self._P:(j+2)*self._P]

        Cc = np.zeros(shape=((self._N+1)*self._n, 1))
        for i in range(1, self._N+1):
            Cc[self._n*i:self._n*(i+1), 0:1] = A_bar @ Cc[self._n*(i-1):self._n*i, 0:1] + C_bar

        # Construct block diagonal Q_bar and R_bar
        Q_bar = np.zeros(shape=((self._N+1)*self._n, (self._N+1)*self._n))
        for i in range(self._N):
            Q_bar[i*self._n:(i+1)*self._n, i*self._n:(i+1)*self._n] = Q
        Q_bar[self._N*self._n:, self._N*self._n:] = F

        R_bar = np.zeros(shape=(self._N*self._P, self._N*self._P))
        for i in range(self._N):
            R_bar[i*self._P:(i+1)*self._P, i*self._P:(i+1)*self._P] = np.eye(self._P)*R

        # Quadratic cost: 0.5 x'Hx + f'x
        H = C.T @ Q_bar @ C + R_bar
        E = C.T @ Q_bar @ Cc + C.T @ Q_bar @ M @ (np.array(e_rr).reshape(self._n, 1))

        H = 2 * H
        f = 2 * E

        # Input constraints
        lb = np.ones(shape=(self._N*self._P, 1))*(-1)
        ub = np.ones(shape=(self._N*self._P, 1))
        G = np.concatenate((np.identity(self._N*self._P), -np.identity(self._N*self._P)))  # （4n, 2n）
        h = np.concatenate((ub, -lb))  # (4n, 1)
        
        # Solve QP using cvxopt
        cvxopt.solvers.options['show_progress'] = False  
        res = cvxopt.solvers.qp(cvxopt.matrix(H), cvxopt.matrix(f),
                                G=cvxopt.matrix(G), h=cvxopt.matrix(h))
        return res['x'][0]

