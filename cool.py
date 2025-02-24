import networkx as nx
import osmnx as ox
import random


def find_route_with_target_distance(graph, start_node, target_distance):
    """
    Finds a route with the specified target distance using the A* algorithm.

    Parameters:
        graph (networkx.MultiDiGraph): The graph to search.
        start_node (int): The starting node of the route.
        target_distance (float): The target distance for the route in meters.

    Returns:
        list: A list of nodes representing the route.
    """

    def calculate_path_distance(graph, path):
        """Calculate the total distance of a path based on edge lengths."""
        return sum(
            nx.get_edge_attributes(graph, "length").get((u, v, k), 0)
            for u, v, k in zip(path[:-1], path[1:], [0] * (len(path) - 1))
        )

    current_distance = 0
    route = [start_node]
    current_node = start_node

    while current_distance < target_distance:
        # Choose a random distance target (approximately 1/4 of the remaining distance)
        remaining_distance = target_distance - current_distance
        random_distance = remaining_distance / 4

        # Find a random node within a random_distance radius
        random_point = ox.utils_geo.sample_points_within_polygon(
            ox.utils_geo.graph_to_gdfs(graph, nodes=True, edges=False)[0].unary_union, 1
        )[0]
        random_node = ox.distance.nearest_nodes(graph, random_point.x, random_point.y)

        # Find A* path to the random node
        try:
            path_to_random = nx.astar_path(
                graph, current_node, random_node, weight="length"
            )
            distance_to_random = calculate_path_distance(graph, path_to_random)

            # Find A* path back to the start node
            path_back = nx.astar_path(
                graph, random_node, start_node, weight="length"
            )
            distance_back = calculate_path_distance(graph, path_back)

            # Check if the round trip fits within the remaining target distance
            if current_distance + distance_to_random + distance_back <= target_distance:
                # Add the path to the route and update distances
                route.extend(path_to_random[1:])
                route.extend(path_back[1:])
                current_distance += distance_to_random + distance_back
                break
            else:
                # Add the path to the route and update current node and distance
                route.extend(path_to_random[1:])
                current_node = random_node
                current_distance += distance_to_random

        except nx.NetworkXNoPath:
            # If no path is found, skip this random point
            continue

    return route

# Load a graph for a specific location
graph = ox.graph_from_point((53.36486137451511, -1.8160056925378616), dist=5000, network_type='walk')
start_node = ox.distance.nearest_nodes(graph, X=-1.8160056925378616, Y=53.36486137451511)

# Choose a starting node
# start_node = list(graph.nodes())[0]

# Find a route with a target distance of 5000 meters
target_distance = 5000  # in meters
route = find_route_with_target_distance(graph, start_node, target_distance)

# Plot the route
fig, ax = ox.plot_graph_route(graph, route, route_linewidth=6, node_size=0, bgcolor='k')
