import random
import time
t= time.time()
import networkx as nx
import osmnx as ox
from shapely.geometry import LineString, Point, Polygon
import geopandas as gpd
from heapq import heappop, heappush
from itertools import count
from networkx.algorithms.shortest_paths.weighted import _weight_function
print(time.time() - t)
t= time.time()
G = ox.graph_from_point((53.36486137451511, -1.8160056925378616), dist=8000, network_type='walk', dist_type="network")
print (time.time() - t)

start_node = ox.nearest_nodes(G, X=-1.8160056925378616, Y=53.36486137451511)
end_node = ox.nearest_nodes(G, Y=53.34344386440596, X=-1.778107050662822)

# G = ox.simplification.simplify_graph(G)
# fig, ax = ox.plot_graph(G,  node_size=0, bgcolor='k')

route = nx.astar_path(G, start_node, end_node, weight='length')

route_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in route]
print(route_coords)
route_line = LineString(route_coords)

target_distance = 15

buffer_distance = target_distance*0.0025
route_buffer = route_line.buffer(buffer_distance, quad_segs=1)

print(route_buffer)

gdf = gpd.GeoDataFrame({'geometry': [route_buffer]}, crs="EPSG:4326")
print(f"gdf:{gdf}")
gdf.to_file("route_buffer.geojson", driver="GeoJSON")
random_polygon = gdf.sample(n=1).iloc[0].geometry

# Get the exterior (boundary) of the polygon
boundary = random_polygon.exterior

# Extract coordinates of the boundary as a list of tuples
coords = list(boundary.coords)

# Select a random coordinate
random_coord = random.choice(coords)

# Convert to a Point geometry
random_node = Point(random_coord)

print("Random node:", random_node)
print(f"Random node coords: {random_node.x}, {random_node.y}")
random_node2 = ox.distance.nearest_nodes(G, X=random_node.y, Y=random_node.x)
print("Random node:", G[random_node2])

route1 = nx.astar_path(G, start_node, random_node2, weight='length')
route2 = nx.astar_path(G, random_node2, end_node, weight='length')

def merge_route(route1, route2):
    return route1[:-1] + route2

merged_route = merge_route(route1, route2)
# fig, ax = ox.plot_graph_route(G, merged_route, route_linewidth=6, node_size=0, bgcolor='k')
# fig, ax = ox.plot_graph_route(G, route1, route_linewidth=6, node_size=0, bgcolor='k')
# fig, ax = ox.plot_graph_route(G, route2, route_linewidth=6, node_size=0, bgcolor='k')
def calculate_path_distance(G, path):
    x=[]
    for u, v in zip(path[:-1], path[1:]):
       x.append(G[u][v][0]['length'])
    return sum(x)


def remove_edges_from_route(G, route):
    graph = G.copy()
    edges_to_remove = [(route[i], route[i + 1]) for i in range(len(route) - 1)]
    print(edges_to_remove)
    nodes_to_remove = [route[0], route[-2]]
    # for node in route:
    #     G.remove_node(node)
    fig, ax = ox.plot_graph_route(graph, route, route_linewidth=6, node_size=0, bgcolor='k')
    # print(edges_to_remove)
    # G.remove_nodes_from(nodes_to_remove)
    # fig, ax = ox.plot_graph(G, node_size=0, bgcolor='k')
    graph.remove_edges_from(edges_to_remove)
    # edges_to_remove = [(route[i], route[i + 1], 0) for i in range(len(route) - 1)]  # Assuming key 0
    # G.remove_edges_from(edges_to_remove)
    # edges_to_remove = [(route[i], route[i + 1], 1) for i in range(len(route) - 1)]  # Assuming key 0
    # G.remove_edges_from(edges_to_remove)
    fig, ax = ox.plot_graph(graph, node_size=0, bgcolor='k')
    return graph

def extend_route_dfs(G, start_node, end_node, route_to_expand):
    visited_outer = set()
    for node in route_to_expand:
        visited_outer.add(node)
    cutoff = 50
    def dfs(current_node, target, path, visited, depth, cumulative_distance):
        if depth > cutoff:
            return
        # if cumulative_distance > target_distance + 2:
        #     return
        if len(path) >= 1:
            cumulative_distance += G[current_node][path[-1]][0]['length']
        # ox.plot_graph_route(G, path, route_linewidth=6, node_size=0, bgcolor='k')
        path.append(current_node)
        visited.add(current_node)
        if current_node == target:
            yield list(path)
        else:
            for neighbor in G.successors(current_node):
                if neighbor not in visited:
                    # cumulative_distance+=G[current_node][neighbor][0]['length']
                    yield from dfs(neighbor, target, path, visited, depth + 1, cumulative_distance)

        # backtrack as not viable node
        path.pop()
        visited.remove(current_node)

    yield from dfs(start_node, end_node, [], visited_outer, 0, 0)


# def astar_with_route(graph, start, goal, heuristic, base_route):
#     # Priority queue for open set (f, g, node, path)
#     weight = "weight"
#     open_set = []
#     heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start, base_route[:]))
#     weight = _weight_function(G, weight)
#     # Closed set to keep track of visited nodes
#     closed_set = set()
#
#     while open_set:
#         # Get the node with the lowest f-value
#         f, g, current, path = heapq.heappop(open_set)
#
#         # If the goal is reached, return the path and cost
#         if current == goal:
#             return path
#
#         # Add current node to the closed set
#         closed_set.add(current)
#
#         # Explore neighbors
#         for neighbor, w in graph[current].items():
#             cost = weight(current, neighbor, w)
#             if neighbor in closed_set:
#                 continue
#
#             # Avoid backtracking to nodes in the base route unless necessary
#             if neighbor in path:
#                 continue
#
#             # Calculate tentative g-score
#             tentative_g = g + cost
#
#             # Create a new path including this neighbor
#             new_path = path + [neighbor]
#
#             # Push to open set with updated f-value
#             heapq.heappush(open_set, (tentative_g + heuristic(neighbor, goal), tentative_g, neighbor, new_path))
#
#     # If the goal is not reachable
#     return None
#
# # Example usage
# def heuristic(node, goal):
#     """Example heuristic function."""
#     return abs(node - goal)

def astar_path(G, source, target, route1, heuristic=None, weight="weight", *, cutoff=None):
    if source not in G or target not in G:
        msg = f"Either source {source} or target {target} is not in G"
        raise nx.NodeNotFound(msg)

    if heuristic is None:
        # The default heuristic is h=0 - same as Dijkstra's algorithm
        def heuristic(u, v):
            return 0

    push = heappush
    pop = heappop
    weight = _weight_function(G, weight)

    G_succ = G._adj  # For speed-up (and works for both directed and undirected graphs)

    # The queue stores priority, node, cost to reach, and parent.
    # Uses Python heapq to keep in priority order.
    # Add a counter to the queue to prevent the underlying heap from
    # attempting to compare the nodes themselves. The hash breaks ties in the
    # priority and is guaranteed unique for all nodes in the graph.
    c = count()
    queue = [(0, next(c), source, 0, None)]

    # Maps enqueued nodes to distance of discovered paths and the
    # computed heuristics to target. We avoid computing the heuristics
    # more than once and inserting the node into the queue too many times.
    enqueued = {}
    # Maps explored nodes to parent closest to the source.
    explored = {}

    while queue:
        # Pop the smallest item from queue.
        _, __, curnode, dist, parent = pop(queue)

        if curnode == target:
            path = [curnode]
            node = parent
            while node is not None:
                path.append(node)
                node = explored[node]
            path.reverse()
            return path

        if curnode in explored:
            # Do not override the parent of starting node
            if explored[curnode] is None:
                continue

            # Skip bad paths that were enqueued before finding a better one
            qcost, h = enqueued[curnode]
            if qcost < dist:
                continue

        explored[curnode] = parent

        for neighbor, w in G_succ[curnode].items():
            cost = weight(curnode, neighbor, w)
            if cost is None:
                continue
            if neighbor in route1:
                # print("Backtracking Detected")
                continue
            ncost = dist + cost
            if neighbor in enqueued:
                qcost, h = enqueued[neighbor]
                # if qcost <= ncost, a less costly path from the
                # neighbor to the source was already determined.
                # Therefore, we won't attempt to push this neighbor
                # to the queue
                if qcost <= ncost:
                    continue
            else:
                h = heuristic(neighbor, target)

            if cutoff and ncost + h > cutoff:
                continue

            enqueued[neighbor] = ncost, h
            push(queue, (ncost + h, next(c), neighbor, ncost, curnode))

    raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}")

n = ox.distance.nearest_nodes(G, Y=53.383737141983225, X=-1.848105996648731)
n2 = ox.distance.nearest_nodes(G, Y=53.35339694450694, X=-1.8666549539010102)
route1 = nx.astar_path(G, start_node, n, weight='length')
route3 = astar_path(G, n, n2, route1, weight='length')
route4 = astar_path(G, n2, end_node, route1, weight='length')
merged_route = merge_route(route1, route3)
merged_route = merge_route(merged_route, route4)
distance = calculate_path_distance(G, merged_route) / 1000
print(f"Distance: {distance}")
fig, ax = ox.plot_graph_route(G, merged_route, route_linewidth=6, node_size=0, bgcolor='k')
# for node in coords:
#     point = Point(node)
#     converted_node = ox.distance.nearest_nodes(G, X=point.y, Y=point.x)
#     route1 = nx.astar_path(G, start_node, converted_node, weight='length')
#     try:
#         route3 = astar_path(G, converted_node, end_node, route1, weight='length')
#     except nx.NetworkXNoPath:
#         continue
#     merged_route = merge_route(route1, route3)
#     distance = calculate_path_distance(G, merged_route) / 1000
#     print(f"Distance: {distance}")
#     if target_distance -1 < distance < target_distance + 1:
#         print(f"Distance Achieved: {distance}")
#         # fig, ax = ox.plot_graph_route(G, route1, route_linewidth=6, node_size=0, bgcolor='k')
#         # fig, ax = ox.plot_graph_route(G, route3, route_linewidth=6, node_size=0, bgcolor='k')
#         fig, ax = ox.plot_graph_route(G, merged_route, route_linewidth=6, node_size=0, bgcolor='k')
        # print("New")


# for node in coords:
#     point = Point(node)
#     converted_node = ox.distance.nearest_nodes(G, X=point.y, Y=point.x)
#     route1 = nx.astar_path(G, start_node, converted_node, weight='length')
#     # G.remove_nodes_from(route1)
#     # fig, ax = ox.plot_graph_route(G, merged_route, route_linewidth=6, node_size=0, bgcolor='k')
#
#     # G_modified = remove_edges_from_route(G, route1)
#     route2 = nx.astar_path(G, converted_node, end_node, weight='length')
#     merged_route = merge_route(route1, route2)
#     # fig, ax = ox.plot_graph_route(G, merged_route, route_linewidth=6, node_size=0, bgcolor='k')
#     distance = calculate_path_distance(G, merged_route) / 1000
#     print(f"Distance: {distance}")
#     print(f"Distance Achieved: {distance}")
#     fig, ax = ox.plot_graph_route(G, route1, route_linewidth=6, node_size=0, bgcolor='k')
#     fig, ax = ox.plot_graph_route(G, route2, route_linewidth=6, node_size=0, bgcolor='k')
#     fig, ax = ox.plot_graph_route(G, merged_route, route_linewidth=6, node_size=0, bgcolor='k')
#     print("New")
    # ox.plot_graph_route(G, route1, route_linewidth=6, node_size=0, bgcolor='k')
    # l = extend_route_dfs(G, converted_node, end_node, route1)
    # route_1_distance = calculate_path_distance(G, route1) / 1000
    # for path in l:
    #     distance = calculate_path_distance(G, path) / 1000
    #     print(f"DFS Distance: {distance}\tRoute 1 Distance: {route_1_distance} \tTotal Distance: {distance+route_1_distance}")
    #     if target_distance - 1 < distance+route_1_distance < target_distance + 1:
    #         print(f"Distance Achieved: {distance}")
    #         fig, ax = ox.plot_graph_route(G, path, route_linewidth=6, node_size=0, bgcolor='k')
    #         break
    # route_3 = astar_path(G, converted_node, end_node, route1)
    # route_3_dis = calculate_path_distance(G, route_3) / 1000
    # print(f"Route 3 Distance: {route_3_dis}")
    # fig, ax = ox.plot_graph_route(G, route_3, route_linewidth=6, node_size=0, bgcolor='k')
    # if target_distance -1 < distance < target_distance + 1:
    #     print(f"Distance Achieved: {distance}")
    #     fig, ax = ox.plot_graph_route(G, merged_route, route_linewidth=6, node_size=0, bgcolor='k')
    #     print("New")
    #     ox.plot_graph_route(G, route1, route_linewidth=6, node_size=0, bgcolor='k')
    #     # l = extend_route_dfs(G, converted_node, end_node, route1)
    #     route_1_distance = calculate_path_distance(G, route1) / 1000
    #     # for path in l:
    #     #     distance = calculate_path_distance(G, path) / 1000
    #     #     print(f"DFS Distance: {distance}\tRoute 1 Distance: {route_1_distance} \tTotal Distance: {distance+route_1_distance}")
    #     #     if target_distance - 1 < distance+route_1_distance < target_distance + 1:
    #     #         print(f"Distance Achieved: {distance}")
    #     #         fig, ax = ox.plot_graph_route(G, path, route_linewidth=6, node_size=0, bgcolor='k')
    #     #         break
    #     route_3 = astar_with_route(G, converted_node, end_node, heuristic, route1)
    #     # route_3_dis = calculate_path_distance(G, route_3) / 1000
    #     # print(f"Route 3 Distance: {route_3_dis}")
    #     fig, ax = ox.plot_graph_route(G, route_3, route_linewidth=6, node_size=0, bgcolor='k')


