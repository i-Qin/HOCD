import math
import numpy as np
from typing import List
from dataclasses import replace

from utils.data import VehicleState


def cal_pre_vehicle_info(ego: VehicleState, pre_ts: float) -> VehicleState:
    """
    Vehicle Motion Prediction
    """
    x_pre = ego.x + ego.vx * pre_ts * math.cos(ego.yaw) - ego.vy * pre_ts * math.sin(ego.yaw)
    y_pre = ego.y + ego.vy * pre_ts * math.cos(ego.yaw) + ego.vx * pre_ts * math.sin(ego.yaw)
    yaw_pre = ego.yaw + ego.yaw_rate * pre_ts

    pre_vehicle_state = replace(
        ego,
        x=x_pre,
        y=y_pre,
        yaw=yaw_pre
    )

    return pre_vehicle_state

def cal_error_fun(path_list: List, vehicle_state: VehicleState):
    """
    Compute the error between a predicted point on the trajectory 
    and the vehicle state at a given time.
    """
    x_pre, y_pre = vehicle_state.x, vehicle_state.y
    vx, vy = vehicle_state.vx, vehicle_state.vy
    fi, fi_dao = vehicle_state.yaw, vehicle_state.yaw_rate

    # 1. Find the nearest (matching) point on the reference path
    path_length = len(path_list)
    min_d = 10000
    min_index = 0
    for i in range(0, path_length):
        d = (path_list[i][0] - x_pre) ** 2 + (path_list[i][1] - y_pre) ** 2
        if d < min_d:
            min_d = d
            min_index = i

    # 2. Compute the tangent vector and normal vector in the path coordinate frame
    tor_v = np.array([math.cos(path_list[min_index][2]), math.sin(path_list[min_index][2])])
    n_v = np.array([-math.sin(path_list[min_index][2]), math.cos(path_list[min_index][2])])

    # 3. Compute the vector from the matching point to the current vehicle position
    d_v = np.array([x_pre - path_list[min_index][0], y_pre - path_list[min_index][1]])

    # 4. Compute lateral error (e_d) and longitudinal error (e_s)
    e_d = np.dot(n_v, d_v)
    e_s = np.dot(tor_v, d_v)

    # 5. Compute reference heading angle theta_r
    theta_r = path_list[min_index][2] + path_list[min_index][3] * e_s

    # 6. Compute derivative of lateral error e_d
    e_d_dao = vy * math.cos(fi - theta_r) + vx * math.sin(fi - theta_r)

    # 7. Compute heading error e_fi
    e_fi = math.sin(fi - theta_r)

    # 8. Compute derivative of longitudinal position s
    s_dao = (vx * math.cos(fi - theta_r) - vy * math.sin(fi - theta_r)) / (1 - path_list[min_index][3] * e_d)

    # 10. Assume curvature at projection point ≈ curvature at matching point
    e_fi_dao = fi_dao - path_list[min_index][3] * s_dao

    e_rr = (e_d, e_d_dao, e_fi, e_fi_dao)

    return e_rr, min_index

def cal_error_point_fun(point, vehicle_state):

    x, y = vehicle_state.x, vehicle_state.y
    vx, vy = vehicle_state.vx, vehicle_state.vy
    fi, fi_dao = vehicle_state.yaw, vehicle_state.yaw_rate

    tor_v = np.array([math.cos(point[2]), math.sin(point[2])])
    n_v = np.array([-math.sin(point[2]), math.cos(point[2])])

    d_v = np.array([x - point[0], y - point[1]])

    e_d = np.dot(n_v, d_v)
    e_s = np.dot(tor_v, d_v)

    theta_r = point[2] + point[3] * e_s 

    e_d_dao = vy * math.cos(fi - theta_r) + vx * math.sin(fi - theta_r)

    e_fi = math.sin(fi - theta_r) 

    S_dao = (vx * math.cos(fi - theta_r) - vy * math.sin(fi - theta_r)) / (1 - point[3] * e_d)

    e_fi_dao = fi_dao - point[3] * S_dao

    e_rr = (e_d, e_d_dao, e_fi, e_fi_dao)

    return e_rr