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

def iterative_dfs(G, start_node, end_node, cutoff, target_distance, tolerance):
    target_distance *= 1000
    tolerance *= 1000
    stack = [(start_node, [], 0, 0)]  # (current_node, path, depth, cumulative_distance)
    visited = set()

    while stack:
        current_node, path, depth, cumulative_distance = stack.pop()

        if depth > cutoff or cumulative_distance > target_distance + tolerance:
            continue

        new_path = path + [current_node]
        if depth > 0:
            cumulative_distance += G[current_node][path[-1]][0]['length']

        if current_node == end_node:
            yield new_path
            continue

        if current_node in visited:
            continue

        # visited.add(current_node)

        for neighbor in G.successors(current_node):
            if neighbor not in new_path:
                stack.append((neighbor, new_path, depth + 1, cumulative_distance))



def iddfs_with_reuse(G, start_node, end_node, maxDepth, target_distance, tolerance):
    target_distance *= 1000
    tolerance *= 1000

    for depth in range(30, maxDepth + 1):
        found = False
        print("Depth: ", depth)
        for path in iterative_dfs(G, start_node, end_node, depth, target_distance, tolerance):
            # ox.plot_graph_route(G, path, route_linewidth=6, node_size=0, bgcolor='k')
            distance = calculate_path_distance(G, path) / 1000
            # print(f"Distance: {distance}")
            if target_distance - tolerance <= distance <= target_distance + tolerance:
                print(f"Route Found at Depth {depth}\nDistance: {distance}")
                fig, ax = ox.plot_graph_route(G, path, route_linewidth=6, node_size=0, bgcolor='k')
                found = True



# iddfs_with_reuse(G, start_node, end_node, 50, 15, 1)
def route_to_coords(route):
    return [(G.nodes[node]['y'], G.nodes[node]['x']) for node in route]

def generate_iddfs_route(distance):
    route = iddfs_with_reuse(G, start_node, end_node, 50, distance, 1)
    return route_to_coords(route)

if __name__ == "__main__":
    route = iddfs_with_reuse(G, start_node, end_node, 50, 10, 1)
    ox.plot_graph_route(G, route, route_linewidth=6, node_size=0, bgcolor='k')
    print("done")
# print("done")
