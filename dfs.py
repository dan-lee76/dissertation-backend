import heapq
import time
t= time.time()
import networkx as nx
import osmnx as ox
print(time.time() - t)
ox.settings.use_cache = True
ox.settings.cache_folder = "/_cache"
# print("Cache enabled:", ox.settings.use_cache)
# print("Cache folder:", ox.settings.cache_folder)
# ox.settings.cache_only_mode = True
# G = ox.graph_from_place('Edale, Derbyshire, England', network_type='walk')
t= time.time()
# Downloads all of the walkable streets
G = ox.graph_from_point((53.36486137451511, -1.8160056925378616), dist=5000, network_type='walk')
print (time.time() - t)
# fig, ax = ox.plot_graph(G)

start_node = ox.distance.nearest_nodes(G, X=-1.8160056925378616, Y=53.36486137451511)
end_node = ox.distance.nearest_nodes(G, Y=53.35510304745989, X=-1.8055002497162305)

# # Plot nodes and edges on a map
# ax = edges.plot(figsize=(6,6), color="gray")
# ax = nodes.plot(ax=ax, color="red", markersize=2.5)

# ox.plot_graph(G, figsize=(6,6), edge_color="grey", node_color="red", node_size=2)
# astar_path = nx.astar_path(G, start_node, end_node, weight='length')

# t = time.time()
# route = ox.shortest_path(G, start_node, end_node, weight='length')
# print ("A Star", time.time() - t)
#
# fig, ax = ox.plot_graph_route(G, route, route_linewidth=6, node_size=0, bgcolor='k')

# t = time.time()
# astar_path = nx.astar_path(G, start_node, end_node, weight='length')
# print (time.time() - t)

# fig, ax = ox.plot_graph_route(G, astar_path, route_linewidth=6, node_size=0, bgcolor='k')
# print(route)

## plot a figure with the nodes data type

# all_paths = nx.all_simple_paths(G, source=start_node, target=end_node, cutoff=50)
# # fig, ax = ox.plot_graph_route(G, all_paths, route_linewidth=6, node_size=0, bgcolor='k')
# # print(G.nodes.values())
# c=0
# for path in all_paths:
#     c+=1
#     # print(path)
#     # fig, ax = ox.plot_graph_route(G, path, route_linewidth=6, node_size=0, bgcolor='k')
#
# print(c)

# def get_path_from_distance(G, source, target, distance_km, cutoff, tollerance):
#     t = time.time()
#     all_paths = nx.all_simple_paths(G, source, target, cutoff)
#     print(time.time() - t)
#     t= time.time()
#     for path in all_paths:
#         distance = calculate_path_distance(G,path) / 1000
#         if distance_km-tollerance <= distance <= distance_km+tollerance:
#             print(time.time() - t)
#             print(f"Route Found\nDistance: {distance}")
#             # fig, ax = ox.plot_graph_route(G, path, route_linewidth=6, node_size=0, bgcolor='k')
#             break
#         # print(osmnx.stats.edge_length_total(path))

def calculate_path_distance(G, path):
    x=[]
    for u, v in zip(path[:-1], path[1:]):
       x.append(G[u][v][0]['length'])
    return sum(x)


# get_path_from_distance(G, start_node, end_node, 10, 50, 1)

def get_all_paths(G, start_node, end_node, cutoff = 25, target_distance = 10, tollerance = 1):
    target_distance = target_distance * 1000
    tollerance = tollerance * 1000
    def dfs(current_node, target, path, visited, depth, cumulative_distance):
        if depth > cutoff:
            return
        if cumulative_distance > target_distance + tollerance:
            return
        if len(path) >= 1:
            cumulative_distance += G[current_node][path[-1]][0]['length']
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


    yield from dfs(start_node, end_node, [], set(), 0, 0)

def process_graph_weightings(graph):
    for u, v, key, data in graph.edges(data=True, keys=True):
        data["weight"] = data["length"]
        if data["highway"] == "footway":
            data["weight"] *= 1
        elif data["highway"] == "bridleway":
            data["weight"] *= 1
        elif data["highway"] == "steps":
            data["weight"] *= 1
        elif data["highway"] == "path":
            data["weight"] *= 1
        elif data["highway"] == "track":
            data["weight"] *= 1
        elif data["highway"] == "pedestrian":
            data["weight"] *= 1
        elif data["highway"] == "tertiary":
            data["weight"] *= 1.2
        else:
            data["weight"] *= 1.4
    return graph

process_graph_weightings(G)

distance_km = 15
tollerance = 1

print("Generating Route")
t = time.time()
x = get_all_paths(G, start_node, end_node, 50, 15)
print(time.time() - t)

t = time.time()
for path in get_all_paths(G, start_node, end_node, 50, target_distance=distance_km):
    distance = calculate_path_distance(G, path) / 1000
    # fig, ax = ox.plot_graph_route(G, path, route_linewidth=6, node_size=0, bgcolor='k')
    if distance_km - tollerance <= distance <= distance_km + tollerance:
        print(time.time() - t)
        print(f"Route Found\nDistance: {distance}")
        fig, ax = ox.plot_graph_route(G, path, route_linewidth=6, node_size=0, bgcolor='k')
        t = time.time()

print("done")

## highway
# - path
# - footway
# - bridleway
# G = process_graph_weightings(G)
# x = nx.astar_path(G, start_node, end_node, weight="weight")
# fig, ax = ox.plot_graph_route(G, x, route_linewidth=6, node_size=0, bgcolor='k')

def find_closest_path(G, start_node, end_node, cutoff = 10, target_distance = 10, tollerance = 1):
    target_distance = target_distance * 1000
    tollerance = tollerance * 1000
    closest_path = []
    closest_diff = float('inf')

    def dfs(current_node, target, path, visited, depth, cumulative_distance, cum_weight = 0):
        nonlocal closest_path, closest_diff
        # Check if this path is closer to the target distance


        # if cumulative_distance > target_distance + closest_diff:
        #     return
        if depth > cutoff:
            return
        # if len(path) >= 2:
        #     # print(path, visited, depth, cumulative_distance, cum_weight)
        #     print(G[path[-1]][current_node][0])
        #     cumulative_distance += G[current_node][path[-1]][0]['length']
        visited.add(current_node)
        # print(running_distance)
        if current_node == target:
            if cum_weight < target_distance + tollerance:
                return
            diff = abs(target_distance - cum_weight)
            if diff < closest_diff:
                closest_diff = diff
                print(f"RETURNING WITH A DISTANCE OF: {cumulative_distance/1000}")
                yield list(path), target_distance - closest_diff  # Make a copy
            return
        # else:
        for neighbor in G.successors(current_node):
            if neighbor not in path:
                # cumulative_distance+=G[current_node][neighbor][0]['length']
                path.append(neighbor)
                weight = G[current_node][neighbor][0]['length']
                cumulative_distance += G[current_node][neighbor][0]['length']
                yield from dfs(neighbor, target, path, visited, depth + 1, cumulative_distance, cum_weight + weight)
                path.pop()



    yield from dfs(start_node, end_node, [], set(), 0, 0)

# print("starting")
# t = time.time()
# x = find_closest_path(G, start_node, end_ node, 30, target_distance=10)
# print("done")
# print(time.time() - t)
# print(x, x[1]/1000)
# fig, ax = ox.plot_graph_route(G, x[0], route_linewidth=6, node_size=0, bgcolor='k')
