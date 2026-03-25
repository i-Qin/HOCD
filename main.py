from omegaconf import OmegaConf

from simulator.simulator import CarlaSimulator
from planner.vehicle_planner import VehiclePlan
from controller.vehicle_controller import VehicleControl
from display.visualization import Visualization
from display.pygame_display import PygameDisplay
from utils.data import ControlInfo, LatDebugInfo, LonDebugInfo


config = OmegaConf.load("utils/config.yaml")

# Initialize simulator
sim = CarlaSimulator(config, init_level="full")

# Visualization module for logging and plotting results
display = Visualization(config, sim)

# Vehicle controller (lateral + longitudinal)
controller = VehicleControl(config)
ctrl = ControlInfo(lat=LatDebugInfo(), lon=LonDebugInfo())

# Initialize planner with map information
map_info = sim.get_map_info()
planner = VehiclePlan(config, map_info)

# Pygame-based real-time visualization
pygame_display = PygameDisplay(config)

# Planning frequency settings
planning_time = 0.2 
planning_count = int(0.2 / 0.01)

count = 0

try:
    while True:
        # Step the simulator and get current observation
        env = sim.tick()

         # Planning
        if count % planning_count == 0:
            plan = planner.run_step(env, ctrl.lat.intention)

        # Compute control commands based on current plan
        ctrl = controller.run_step(env, plan)

        # Update visualization
        pygame_display.run_step(env, ctrl)
        display.run_step(env, plan, ctrl)
        
        # Apply control to the vehicle
        sim.control_vehicle(throttle=ctrl.throttle, steer=ctrl.steer, brake=ctrl.brake, gear=1)

        # Stop the vehicle when close to destination
        if plan.distance < 7:
            sim.control_vehicle(steer=0, throttle=0, brake=2)
            print("last waypoint reached")
            break

        count += 1
        
    # Show final results and  plots
    display.show_result()

finally:
    # Clean up all actors
    sim.ego_vehicle.destroy()
    for veh in sim.other_vehicles:
        veh.destroy()
