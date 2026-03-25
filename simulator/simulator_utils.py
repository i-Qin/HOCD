import math
import numpy as np

from utils.data import VehicleParam, VehicleState


def get_vehicle_parameters(vehicle):
    physics_control = vehicle.get_physics_control()

    wheel0 = physics_control.wheels[0].position  # left front wheel
    wheel1 = physics_control.wheels[1].position  # right front wheel
    wheel2 = physics_control.wheels[2].position  # left rear wheel
    wheel3 = physics_control.wheels[3].position  # right rear wheel

    # wheelbase
    dis13 = math.sqrt((wheel1.x - wheel3.x) ** 2 + (wheel1.y - wheel3.y) ** 2)
    wheelbase = dis13 / 100

    lf = wheelbase / 2
    lr = wheelbase / 2

    mass = physics_control.mass
    mass_front = mass / 2
    mass_rear = mass / 2
    iz = lf ** 2 * mass_front + lr ** 2 * mass_rear

    cf = -110000
    cr = -110000

    return VehicleParam(lf, lr, cf, cr, mass, iz)

def get_vehicle_state(vehicle):
    param = get_vehicle_parameters(vehicle)

    transform = vehicle.get_transform()
    location = transform.location
    x = location.x
    y = location.y
    z = location.z

    vx, vy, vz = get_velocity_vcs(vehicle)

    ax, ay = get_acceleration_vcs(vehicle)

    yaw = get_yaw(vehicle)
    yaw_rate = get_yaw_rate(vehicle)

    vec_forward = get_forward_vector(vehicle)
    vec_right = get_right_vector(vehicle)
    vec_up = get_up_vector(vehicle)
    
    return VehicleState(param, x, y, z, vx, vy, vz, ax, ay, yaw, yaw_rate, vec_forward, vec_right, vec_up)

def get_forward_vector(vehicle):
    forward = vehicle.get_transform().get_forward_vector()
    vec_forward = [forward.x, forward.y, forward.z]
    return vec_forward

def get_right_vector(vehicle):
    right = vehicle.get_transform().get_right_vector()
    vec_right = [right.x, right.y, right.z]
    return vec_right

def get_up_vector(vehicle):
    up = vehicle.get_transform().get_up_vector()
    vec_up = [up.x, up.y, up.z]
    return vec_up

def get_yaw(vehicle):
    yaw = vehicle.get_transform().rotation.yaw * (math.pi / 180)
    return yaw

def get_yaw_rate(vehicle):
    yaw_rate = vehicle.get_angular_velocity().z * (math.pi / 180)
    return yaw_rate

def get_velocity_vcs(vehicle):
    """
    Vehicle lateral and longitudinal velocities (m/s) in the vehicle coordinate system 
    """
    vec_right = vehicle.get_transform().get_right_vector()
    vec_right = [vec_right.x, vec_right.y, vec_right.z]
    vec_forward = vehicle.get_transform().get_forward_vector()
    vec_forward = [vec_forward.x, vec_forward.y, vec_forward.z]
    vec_up = vehicle.get_transform().get_up_vector()
    vec_up = [vec_up.x, vec_up.y, vec_up.z]
    v = vehicle.get_velocity()
    vec_v = [v.x, v.y, v.z]
    vy = np.dot(vec_v, vec_right)
    vx = np.dot(vec_v, vec_forward)
    vz = np.dot(vec_v, vec_up)
    return vx, vy, vz


def get_acceleration_vcs(vehicle):
    """
    Vehicle lateral and longitudinal acceleration (m/s^2) in the vehicle coordinate system 
    """
    vec_right = vehicle.get_transform().get_right_vector()
    vec_right = [vec_right.x, vec_right.y, vec_right.z]
    vec_forward = vehicle.get_transform().get_forward_vector()
    vec_forward = [vec_forward.x, vec_forward.y, vec_forward.z]
    acc = vehicle.get_acceleration()
    vec_acc = [acc.x, acc.y, acc.z]
    ax = np.dot(vec_acc, vec_forward)
    ay = np.dot(vec_acc, vec_right)
    return ax, ay







