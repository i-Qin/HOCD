import carla
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, fields

# ===================simulator=========================
@dataclass
class VehicleParam:
    lf: float
    lr: float
    cf: float
    cr: float
    mass: float
    iz: float

@dataclass
class VehicleState:
    param: VehicleParam
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    ax: float
    ay: float
    yaw: float
    yaw_rate: float
    vec_forward: float
    vec_right: float
    vec_up: float

@dataclass
class SimulatorObservation:
    ego: VehicleState
    neighbors: List[VehicleState]
    image: Optional[np.ndarray]

@dataclass
class MapInfo:
    town_map: carla.Map
    topology: list
    start: carla.Location
    end: carla.Location

# ===================planner=========================
@dataclass
class PlanInfo:
    global_path: List[Tuple[float, float, float, float]]  # x,y,yaw,kappa
    local_path: List[Tuple[float, float, float, float]]
    speed: list
    distance: float

# ===================controller=========================
@dataclass
class LatDebugInfo:

    e_d: Optional[float] = None
    e_d_dao: Optional[float] = None
    e_fi: Optional[float] = None
    e_fi_dao: Optional[float] = None

    # Shared control
    steer_driver: Optional[float] = None
    steer_vehicle: Optional[float] = None
    authority: Optional[float] = None

    # G29
    g29_connected: Optional[bool] = None
    triangle: Optional[bool] = None
    rect: Optional[bool] = None
    circle: Optional[bool] = None
    times: Optional[bool] = None
    intention: Optional[str] = None
    reaction_time: Optional[float] = None

    def update(self, other):
        if other is None:
            return

        for f in fields(self):
            v = getattr(other, f.name)
            if v is not None and getattr(self, f.name) is None:
                setattr(self, f.name, v)

@dataclass
class LonDebugInfo:
    cur_speed: Optional[float] = None
    target_speed: Optional[float] = None


@dataclass
class ControlInfo:
    lat: LatDebugInfo
    lon: LonDebugInfo
    steer: Optional[float] = None
    throttle: Optional[float] = None
    brake: Optional[float] = None



    

