import heapq
import time
import networkx as nx
import osmnx as ox
import igraph as ig

# Timing and Graph Creation
t = time.time()
ox.settings.use_cache = True
ox.settings.cache_folder = "/_cache"

t = time.time()
G = ox.graph_from_point((53.36486137451511, -1.8160056925378616), dist=5000, network_type='walk', simplify=True)
print("Graph generation time:", time.time() - t)

start_node = ox.distance.nearest_nodes(G, X=-1.8160056925378616, Y=53.36486137451511)
end_node = ox.distance.nearest_nodes(G, Y=53.35510304745989, X=-1.8055002497162305)

print(f"Amount of Nodes: {len(G.nodes)}")
print(f"Amount of Edges: {len(G.edges)}")

# Convert NetworkX Graph to igraph
fast_graph = ig.Graph.from_networkx(G)

def calculate_path_distance(graph, path):
    """Calculate the total distance of a path in igraph."""
    distance = 0
    for u, v in zip(path[:-1], path[1:]):
        eid = graph.get_eid(u, v)
        distance += graph.es[eid]['length']
    return distance

def get_all_paths(graph, start_node, end_node, cutoff=25, target_distance=10, tolerance=1):
    """Generate all paths using a depth-limited search."""
    target_distance *= 1000
    tolerance *= 1000

    def dfs(graph, current_node, target, path, visited, depth, cumulative_distance):
        if depth >= cutoff or cumulative_distance > target_distance + tolerance:
            return
        if len(path) >= 1:
            eid = graph.get_eid(path[-1], current_node)
            cumulative_distance += graph.es[eid]['length']
        path.append(current_node)
        visited.add(current_node)
        if current_node == target:
            yield list(path)
        else:
            for neighbor in graph.neighbors(current_node, mode="out"):
                if neighbor not in visited:
                    yield from dfs(graph, neighbor, target, path, visited, depth + 1, cumulative_distance)
        # Backtrack
        path.pop()
        visited.remove(current_node)

    yield from dfs(graph, start_node, end_node, [], set(), 0, 0)

def iddfs(graph, start_node, end_node, maxDepth, target_distance):
    """Iterative Deepening Depth-First Search (IDDFS) with igraph."""
    t = time.time()
    for depth in range(maxDepth):
        print(f"Depth: {depth}")
        for path in get_all_paths(graph, start_node, end_node, depth, target_distance):
            distance = calculate_path_distance(graph, path) / 1000
            if distance_km - tollerance <= distance <= distance_km + tollerance:
                print(time.time() - t)
                print(f"Route Found\nDistance: {distance}")
                fig, ax = ox.plot_graph_route(G, path, route_linewidth=6, node_size=0, bgcolor='k')
                t = time.time()

# Parameters
distance_km = 15
tollerance = 1

# Execute IDDFS
print("Generating Route")
iddfs(fast_graph, start_node, end_node, 75, 15)
print("done")
