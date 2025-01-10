import heapq
import time
t= time.time()
import networkx as nx
import osmnx as ox
print(time.time() - t)
ox.settings.use_cache = True
ox.settings.cache_folder = "/_cache"
t= time.time()
G = ox.graph_from_point((53.36486137451511, -1.8160056925378616), dist=5000, network_type='walk', simplify=True)
print (time.time() - t)
# fig, ax = ox.plot_graph(G)

start_node = ox.distance.nearest_nodes(G, X=-1.8160056925378616, Y=53.36486137451511)
end_node = ox.distance.nearest_nodes(G, Y=53.35510304745989, X=-1.8055002497162305)

print(f"Amount of Nodes: {len(G.nodes)}")
print(f"Amount of Edges: {len(G.edges)}")


def calculate_path_distance(G, path):
    x=[]
    for u, v in zip(path[:-1], path[1:]):
       x.append(G[u][v][0]['length'])
    return sum(x)


# get_path_from_distance(G, start_node, end_node, 10, 50, 1)

def get_all_paths(G, start_node, end_node, cutoff = 25, target_distance = 10, tolerance = 1):
    target_distance *= 1000
    tolerance *= 1000
    def dfs(current_node, target, path, visited, depth, cumulative_distance):
        if depth >= cutoff or cumulative_distance > target_distance + tolerance:
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

distance_km = 15
tollerance = 1

print("Generating Route")
t = time.time()
print(time.time() - t)

def iddfs(graph, start_node, end_node, maxDepth, target_distance):
    t = time.time()
    for depth in range(maxDepth):
        print(f"Depth: {depth}")
        # for path in nx.all_simple_paths(G, start_node, end_node, depth):
        for path in get_all_paths(graph, start_node, end_node, depth, target_distance):
            distance = calculate_path_distance(G, path) / 1000
            # print(f"Distance: {distance}")
            # fig, ax = ox.plot_graph_route(G, path, route_linewidth=6, node_size=0, bgcolor='k')
            if distance_km - tollerance <= distance <= distance_km + tollerance:
                print(time.time() - t)
                print(f"Route Found\nDistance: {distance}")
                fig, ax = ox.plot_graph_route(G, path, route_linewidth=6, node_size=0, bgcolor='k')
                t = time.time()

iddfs(G, start_node, end_node, 75, 15)

print("done")
