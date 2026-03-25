import math
import carla
import pickle
import numpy as np
from enum import Enum
import networkx as nx

from planner.plan_utils import waypoint_list_2_target_path

class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to others.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANE_FOLLOW = 4
    CHANGE_LANE_LEFT = 5
    CHANGE_LANE_RIGHT = 6


class GlobalPlan:
    def __init__(self, config, town_map, topology):
        self._map = town_map
        self._topology = topology
        self._load_config(config)
        self._build_graph()

    def _load_config(self, config):
        self._ds = config.planner.ds

    def plan(self, origin: carla.Location, destination: carla.Location):
        """
        Generate a full waypoint-based path between origin and destination.
        """
        # Obtain preliminary planning results using A *
        route = self._route_search(origin, destination) 
        origin_wp = self._map.get_waypoint(origin)
        destination_wp = self._map.get_waypoint(destination) 
        path_way = []

        # First segment
        edge = self._graph.get_edge_data(route[0], route[1])
        path = [edge["entry_waypoint"]] + edge["path"] + [edge["exit_waypoint"]]
        clos_index = self._closest_index(origin_wp, path)
        for wp in path[clos_index:]:
            path_way.append((wp, edge["type"]))

        # Middle segments
        if len(route) > 3: 
            for index in range(1, len(route) - 2):
                edge = self._graph.get_edge_data(route[index], route[index + 1])
                path = edge["path"] + [edge["exit_waypoint"]]
                for wp in path:
                    path_way.append((wp, edge["type"]))

        # Last segment
        edge = self._graph.get_edge_data(route[-2], route[-1])
        path = edge["path"] + [edge["exit_waypoint"]]
        clos_index = self._closest_index(destination_wp, path)
        if clos_index != 0: 
            for wp in path[:clos_index + 1]:
                path_way.append((wp, edge["type"]))
        else: 
            pass

        path_xyz = [(wp.transform.location.x, wp.transform.location.y, wp.transform.location.z) for wp, _ in path_way]

        x_y_heading_k_list, csp, x, y, z = waypoint_list_2_target_path(path_xyz, self._ds)

        return x_y_heading_k_list, csp, x, y, z

    def _build_graph(self):
        """"
        Build a directed graph from map topology for global path planning.
        """
        self._graph = nx.DiGraph() 
        self._id_map = dict()  
        self._road_to_edge = dict() 

        for seg in self._topology:
            entry_waypoint = seg["entry"] 
            exit_waypoint = seg["exit"] 
            path = seg["path"]  
            intersection = entry_waypoint.is_intersection
            road_id, section_id, lane_id = entry_waypoint.road_id, entry_waypoint.section_id, entry_waypoint.lane_id
            entry_xyz = entry_waypoint.transform.location
            entry_xyz = (np.round(entry_xyz.x, 2), np.round(entry_xyz.y, 2), np.round(entry_xyz.z, 2)) 
            exit_xyz = exit_waypoint.transform.location
            exit_xyz = (np.round(exit_xyz.x, 2), np.round(exit_xyz.y, 2), np.round(exit_xyz.z, 2))
            for xyz in entry_xyz, exit_xyz:
                if xyz not in self._id_map:
                    New_ID = len(self._id_map)
                    self._id_map[xyz] = New_ID
                    self._graph.add_node(New_ID, vertex=xyz)

            n1 = self._id_map[entry_xyz]
            n2 = self._id_map[exit_xyz]

            if road_id not in self._road_to_edge:
                self._road_to_edge[road_id] = dict()
            if section_id not in self._road_to_edge[road_id]:
                self._road_to_edge[road_id][section_id] = dict()

            self._road_to_edge[road_id][section_id][lane_id] = (n1, n2)

            entry_forward_vector = entry_waypoint.transform.rotation.get_forward_vector()  
            exit_forward_vector = exit_waypoint.transform.rotation.get_forward_vector() 

            self._graph.add_edge(u_of_edge=n1, v_of_edge=n2,
                                 length=len(path) + 1, path=path,
                                 entry_waypoint=entry_waypoint, exit_waypoint=exit_waypoint,
                                 entry_vector=entry_forward_vector, exit_vector=exit_forward_vector,
                                 net_vector=self.vector_fun(entry_waypoint.transform.location,
                                                                      exit_waypoint.transform.location),
                                 intersection=intersection, type=RoadOption.LANE_FOLLOW)

    def _route_search(self, origin, destination):
        """
        Compute route using A* search.
        """
        start_edge = self._find_location_edge(origin)  
        end_edge = self._find_location_edge(destination)  
        route = self._A_star(start_edge[0], end_edge[0])
        if route is None:  
            raise nx.NetworkXNoPath(f"Node {start_edge[0]} not reachable from {end_edge[0]}")
        route.append(end_edge[1]) 
        return route

    def _find_location_edge(self, loc: carla.Location):
        """
        Find the graph edge corresponding to a given location.
        """
        nearest_wp = self._map.get_waypoint(loc)
        edge = None
        try:
            edge = self._road_to_edge[nearest_wp.road_id][nearest_wp.section_id][nearest_wp.lane_id]
        except KeyError:
            pass
        return edge

    def _A_star(self, n_begin, n_end):
        """
        A* search for shortest path between two nodes.
        """
        route = []
        open_set = dict() 
        closed_set = dict()
        open_set[n_begin] = (0, -1) 
        def cal_heuristic(n):
            return math.hypot(self._graph.nodes[n]['vertex'][0] - self._graph.nodes[n_end]['vertex'][0],
                              self._graph.nodes[n]['vertex'][1] - self._graph.nodes[n_end]['vertex'][1])

        while 1:
            if len(open_set) == 0:
                return None
            c_node = min(open_set, key=lambda n: open_set[n][0] + cal_heuristic(n))
            if c_node == n_end:
                closed_set[c_node] = open_set[c_node]
                del open_set[c_node] 
                break
            for suc in self._graph.successors(c_node): 
                new_cost = self._graph.get_edge_data(c_node, suc)["length"]   
                if suc in closed_set: 
                    continue
                elif suc in open_set: 
                    if open_set[c_node][0] + new_cost < open_set[suc][0]:
                        open_set[suc] = (open_set[c_node][0] + new_cost, c_node)
                else:  
                    open_set[suc] = (open_set[c_node][0] + new_cost, c_node)
            closed_set[c_node] = open_set[c_node]
            del open_set[c_node] 

        route.append(n_end)
        while 1:
            if closed_set[route[-1]][1] != -1:
                route.append(closed_set[route[-1]][1])  
            else:
                break
        return list(reversed(route))

    @staticmethod
    def _closest_index(current_waypoint, waypoint_list):
        """
        Find index of closest waypoint in a list.
        """
        min_distance = float('inf') 
        closest_index = -1
        for i, waypoint in enumerate(waypoint_list):
            distance = waypoint.transform.location.distance(current_waypoint.transform.location)
            if distance < min_distance:
                min_distance = distance
                closest_index = i

        return closest_index

    @staticmethod
    def vector_fun(loc_1: carla.Location, loc_2: carla.Location):
        """
        Compute normalized direction vector from loc_1 to loc_2.
        """

        delt_x = loc_2.x - loc_1.x
        delt_y = loc_2.y - loc_1.y
        delt_z = loc_2.z - loc_1.z
        norm = np.linalg.norm([delt_x, delt_y, delt_z]) + np.finfo(float).eps 
        return np.round([delt_x / norm, delt_y / norm, delt_z / norm], 4) 
    

    def build_all_feasible_pairs(self, min_length=50):
        """
        Compute all feasible (start, end) node pairs with valid paths.
        """

        feasible_pairs = []
        nodes = list(self._graph.nodes)

        for i, start in enumerate(nodes):
            for end in nodes:
                if start == end:
                    continue
                route = self._A_star(start, end)
                if route is None:
                    continue
                if len(route) < min_length:
                    continue
                feasible_pairs.append((start, end))

        with open("feasible_routes.pkl", "wb") as f:
            pickle.dump(feasible_pairs, f)

        print("Total feasible routes:", len(feasible_pairs))