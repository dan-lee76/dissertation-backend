import random
import time

t= time.time()
import networkx as nx
import osmnx as ox
print(time.time() - t)
t= time.time()
G = ox.graph_from_point((53.36486137451511, -1.8160056925378616), dist=5000, network_type='walk', dist_type="network")
print (time.time() - t)

start_node = ox.distance.nearest_nodes(G, X=-1.8160056925378616, Y=53.36486137451511)
end_node = ox.distance.nearest_nodes(G, Y=53.35510304745989, X=-1.8055002497162305)

# G = ox.simplification.simplify_graph(G)
fig, ax = ox.plot_graph(G,  node_size=0, bgcolor='k')



# route = nx.astar_path(G, start_node, end_node, weight='length')
#
# fig, ax = ox.plot_graph_route(G, route, route_linewidth=6, node_size=0, bgcolor='k')

def find_random_point(start_node, distance, G):
    """
    Finds a random point approximately at a given distance from the start node in a graph.

    Parameters:
    - start_node: The starting node ID
    - distance: Distance in meters
    - G: The OSMNx graph

    Returns:
    - A tuple (lat, lon) of the random point's coordinates
    """
    # Get all nodes within the specified distance from the start_node
    subgraph = nx.ego_graph(G, start_node, radius=distance, distance='length')

    # Get nodes within the subgraph
    nodes = list(subgraph.nodes)

    # Select a random node
    random_node = random.choice(nodes)

    # Get the coordinates of the random node
    point = (G.nodes[random_node]['y'], G.nodes[random_node]['x'])

    return point

yen = ox.shortest_path(G, start_node, end_node, weight='length')
fig, ax = ox.plot_graph_route(G, yen, route_linewidth=6, node_size=0, bgcolor='k')

distance = 10
distance_km = distance * 1000
# other_point = find_random_point(start_node, distance_km, G)
# other_point = ox.distance.nearest_nodes(G, X=other_point[1], Y=other_point[0])
# yen2 = ox.shortest_path(G, start_node, other_point, weight='length')
# fig, ax = ox.plot_graph_route(G, yen2, route_linewidth=6, node_size=0, bgcolor='k')

def calculate_path_distance(G, path):
    x=[]
    for u, v in zip(path[:-1], path[1:]):
       x.append(G[u][v][0]['length'])
    return sum(x)


# def generate_path(G, start_node, end_node, target_distance):
#     """
#     Generates a path in a graph with a target distance using random points.
#
#     Parameters:
#     - G: The OSMNx graph
#     - start_node: The starting node ID
#     - end_node: The ending node ID
#     - target_distance: The target distance in meters
#
#     Returns:
#     - A list of node IDs representing the path
#     """
#     current_distance = 0
#     route = [start_node]
#     current_node = start_node
#
#     while current_distance < target_distance:
#         # Choose a random distance target (approximately 1/4 of the remaining distance)
#         remaining_distance = target_distance - current_distance
#         random_distance = remaining_distance / 4
#
#         # Find a random node within a random_distance radius
#         random_point = find_random_point(start_node, random_distance, G)
#         random_node = ox.distance.nearest_nodes(G, X=random_point[1], Y=random_point[0])
#
#         # Find A* path to the random node
#         try:
#             path_to_random = nx.astar_path(
#                 G, current_node, random_node, weight="length"
#             )
#             distance_to_random = calculate_path_distance(G, path_to_random)
#
#             # Find A* path back to the start node
#             path_back = nx.astar_path(
#                 G, random_node, start_node, weight="length"
#             )
#             distance_back = ox.utils_graph.calculate_path_length(G, path_back)
#
#             # Check if the round trip fits within the remaining target distance
#             if current_distance + distance_to_random + distance_back <= target_distance:
#                 # Add the path to the route and update distances
#                 route.extend(path_to_random[1:])
#                 route.extend(path_back[1:])
#                 current_distance += distance_to_random + distance_back
#                 break
#             else:
#                 # Add the path to the route and update current node and distance
#                 route.extend(path_to_random[1:])
#                 current_node = random_node
#                 current_distance += distance_to_random
#
#         except nx.NetworkXNoPath:
#             # If no path is found, skip this random point
#             continue
#     return route

def generate_path(graph, start_node, end_node, target_distance, tollerance = 0):
    current_distance = 0
    current_node = start_node
    route = [start_node]
    visited = set()
    temp_visited = set()
    start = True

    while current_distance < target_distance:
        remaining_distance = target_distance - current_distance
        random_distance = remaining_distance / 2
        random_point = find_random_point(current_node, random_distance, graph)
        random_node = ox.distance.nearest_nodes(graph, X=random_point[1], Y=random_point[0])

        # print("Random Node", random_node)
        path_to_random = nx.astar_path(graph, current_node, random_node, weight='length')
        distance_to_random = calculate_path_distance(graph, path_to_random)

        # ox.plot_graph_route(G, path_to_random, route_linewidth=6, node_size=0, bgcolor='k')
        # for node in path_to_random:
        #     if node == random_node:
        #         continue
        #     elif node not in visited:
        #         visited.add(node)
        #         graph.remove_node(node)
        # ox.plot_graph(graph, node_size=0, bgcolor='k')
        # print("Route Back start")
        path_back = nx.astar_path(graph, random_node, end_node, weight='length')
        distance_back = calculate_path_distance(graph, path_back)
        # print("Route Back generated")

        if (current_distance + distance_to_random + distance_back >= target_distance - tollerance):
            route.extend(path_to_random[1:])
            route.extend(path_back[1:])
            current_distance += distance_to_random + distance_back
            print(f"Distance: {current_distance}m")
            print(f"Distance: {current_distance/1000}km")
            print("once")
            break
        else:
            # if any(node in temp_visited for node in path_back):
            #     continue
            # else:
            start = False
            # visited.update(temp_visited)
            route.extend(path_to_random[1:])
            current_node = random_node
            current_distance += distance_to_random
            # fig, ax = ox.plot_graph_route(G, route, route_linewidth=6, node_size=0, bgcolor='k')
    return route

t = time.time()
route = generate_path(G, start_node, end_node, 15000, 1000)
print(time.time() - t)
G = ox.graph_from_point((53.36486137451511, -1.8160056925378616), dist=5000, network_type='walk', dist_type="network")
fig, ax = ox.plot_graph_route(G, route, route_linewidth=6, node_size=0, bgcolor='k')

def generate_path_alt(graph, start_node, end_node, target_distance, tollerance = 0):
    current_distance = 0
    current_node = start_node
    route = [start_node]
    visited = set()
    temp_visited = set()
    start = True

    while current_distance < target_distance:
        remaining_distance = target_distance - current_distance
        random_distance = remaining_distance / 2
        random_point = find_random_point(current_node, random_distance, graph)
        random_node = ox.distance.nearest_nodes(graph, X=random_point[1], Y=random_point[0])

        # print("Random Node", random_node)
        path_to_random = nx.astar_path(graph, current_node, random_node, weight='length')
        distance_to_random = calculate_path_distance(graph, path_to_random)

        # ox.plot_graph_route(G, path_to_random, route_linewidth=6, node_size=0, bgcolor='k')
        # for node in path_to_random:
        #     if node == random_node:
        #         continue
        #     elif node not in visited:
        #         visited.add(node)
        #         graph.remove_node(node)
        # ox.plot_graph(graph, node_size=0, bgcolor='k')
        # print("Route Back start")
        path_back = nx.astar_path(graph, random_node, end_node, weight='length')
        distance_back = calculate_path_distance(graph, path_back)
        # print("Route Back generated")

        if (current_distance + distance_to_random + distance_back >= target_distance - tollerance):
            route.extend(path_to_random[1:])
            route.extend(path_back[1:])
            current_distance += distance_to_random + distance_back
            print(f"Distance: {current_distance}m")
            print(f"Distance: {current_distance/1000}km")
            print("once")
            break
        else:
            # if any(node in temp_visited for node in path_back):
            #     continue
            # else:
            start = False
            # visited.update(temp_visited)
            route.extend(path_to_random[1:])
            current_node = random_node
            current_distance += distance_to_random
            # fig, ax = ox.plot_graph_route(G, route, route_linewidth=6, node_size=0, bgcolor='k')
    return route