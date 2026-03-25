import carla
import pickle
from tqdm import tqdm
import networkx as nx

from simulator.simulator import CarlaSimulator
from planner.global_plan import GlobalPlan
from omegaconf import OmegaConf

config = OmegaConf.load("utils/config.yaml")

sim = CarlaSimulator(config)

town_map, topology, _, _ = sim.get_map_info()

global_planner = GlobalPlan(config, town_map, topology)

spawn_points = sim.all_default_spawn[:50]

routes = []

print("Spawn points:", len(spawn_points))

for i in tqdm(range(len(spawn_points))):
    for j in range(len(spawn_points)):

        if i == j:
            continue

        start = spawn_points[i].location
        end = spawn_points[j].location

        try:
            x_y_heading_k_list, csp, x, y, z = global_planner.plan(start, end)

            if len(x_y_heading_k_list) < 50:
                continue

            routes.append({
                "start_idx": i,
                "end_idx": j,
                "x_y_heading_k_list": x_y_heading_k_list,
                "x": x,
                "y": y,
                "z": z
            })

        except nx.NetworkXNoPath:
            continue

print("Feasible routes:", len(routes))

# 保存
with open("feasible_routes.pkl", "wb") as f:
    pickle.dump(routes, f)

print("Saved to feasible_routes.pkl")