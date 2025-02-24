import heapq
import random
import time

from graph_preperation import Graph_Builder

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
G = ox.graph_from_point((53.36486137451511, -1.8160056925378616), dist=5000, network_type='walk', simplify=True)
print (time.time() - t)
# fig, ax = ox.plot_graph(G)

start_node = ox.nearest_nodes(G, X=-1.8160056925378616, Y=53.36486137451511)
end_node = ox.nearest_nodes(G, Y=53.34344386440596, X=-1.778107050662822)

# # Plot nodes and edges on a map
# ax = edges.plot(figsize=(6,6), color="gray")
# ax = nodes.plot(ax=ax, color="red", markersize=2.5)

# ox.plot_graph(G, figsize=(6,6), edge_color="grey", node_color="red", node_size=2)
# t= time.time()
# astar_path = nx.astar_path(G, start_node, end_node, weight='length')
# print(f"A*:{time.time() - t}")

# t = time.time()
# route = ox.shortest_path(G, start_node, end_node, weight='length')
# print ("A Star", time.time() - t)
#
# fig, ax = ox.plot_graph_route(G, astar_path, route_linewidth=6, node_size=0, bgcolor='k')

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

def get_all_paths_optimized(G, start_node, end_node, cutoff=25, target_distance=10, tolerance=1):
    target_distance *= 1000
    tolerance *= 1000

    # Precompute edge lengths for quick access
    edge_lengths = {}
    for u, v, data in G.edges(data=True):
        edge_lengths.setdefault(u, {})[v] = data['length']
    print("Edge Lengths Computed")
    # Iterative DFS using a stack to avoid recursion overhead
    stack = [(start_node, iter(G.successors(start_node)), None, 0, 0)]
    path = [start_node]
    visited = {start_node}
    print("Starting DFS")
    while stack:
        current_node, neighbors, prev_node, depth, cumulative = stack[-1]
        try:
            neighbor = next(neighbors)
            if neighbor in visited:
                continue

            # Calculate new cumulative distance
            new_cumulative = cumulative
            if prev_node is not None:
                new_cumulative += edge_lengths[current_node][neighbor]

            # Prune paths exceeding constraints early
            new_depth = depth + 1
            if (new_depth > cutoff) or (new_cumulative > target_distance + tolerance):
                continue

            # Update path and visited
            path.append(neighbor)
            visited.add(neighbor)

            # Yield valid paths
            if neighbor == end_node:
                if target_distance - tolerance <= new_cumulative <= target_distance + tolerance:
                    yield list(path)

            # Push new state to stack
            stack.append((neighbor, iter(G.successors(neighbor)), current_node, new_depth, new_cumulative))

        except StopIteration:
            # Backtrack: remove current node from path and visited
            stack.pop()
            if path:
                removed = path.pop()
                visited.remove(removed)

def get_all_paths_different(G, start_node, end_node, cutoff = 25, target_distance = 10, tolerance = 1):
    target_distance *= 1000
    tolerance *= 1000
    edge_lengths = {}
    for u, v, data in G.edges(data=True):
        edge_lengths.setdefault(u, {})[v] = data['length']

    stack = [(start_node, iter(G.successors(start_node)), None, 0, 0)]
    path = [start_node]
    visited = {start_node}
    while stack:
        current_node, neighbors, prev_node, depth, cumulative = stack[-1]
        try:
            neighbor = next(neighbors)
        except StopIteration:
            # Backtrack: remove current node from path and visited
            stack.pop()
            if path:
                removed = path.pop()
                visited.remove(removed)
        else:
            if neighbor in visited:
                continue

            new_cumulative = cumulative
            if len(path) >= 1:
                new_cumulative += G[current_node][neighbor][0]['length']

            visited.add(neighbor)
            path.append(neighbor)

            if depth >= cutoff or new_cumulative > target_distance + tolerance:
                path.pop()
                visited.remove(neighbor)
                continue

            if neighbor == end_node:
                if target_distance - tolerance <= new_cumulative <= target_distance + tolerance:
                    yield list(path)

            stack.append((neighbor, iter(G.successors(neighbor)), current_node, depth + 1, new_cumulative))



def get_all_paths_bidirectional(G, start_node, end_node, cutoff=25, target_distance=10, tolerance=1):
    target_distance *= 1000
    tolerance *= 1000

    # Precompute edge lengths for quick access
    edge_lengths = {}
    for u, v, data in G.edges(data=True):
        edge_lengths.setdefault(u, {})[v] = data['length']
    reverse_edge_lengths = {}
    for u, v, data in G.edges(data=True):
        reverse_edge_lengths.setdefault(v, {})[u] = data['length']

    # Track visited nodes and their paths from both directions
    forward_visited = {}  # {node: (path, cumulative_distance)}
    backward_visited = {}  # {node: (path, cumulative_distance)}

    # Initialize stacks for forward and backward DFS
    forward_stack = [(start_node, [start_node], 0, 0, {start_node})]
    backward_stack = [(end_node, [end_node], 0, 0, {end_node})]

    while forward_stack or backward_stack:
        # Process forward DFS
        if forward_stack:
            current, path, depth, dist, visited = forward_stack.pop()
            forward_visited[current] = (path, dist)

            # Check if current node exists in backward search results
            if current in backward_visited:
                back_path, back_dist = backward_visited[current]
                total_dist = dist + back_dist
                if (target_distance - tolerance <= total_dist <= target_distance + tolerance):
                    full_path = path[:-1] + back_path[::-1]
                    yield full_path

            # Expand forward neighbors
            if depth < cutoff and dist <= target_distance + tolerance:
                for neighbor in G.successors(current):
                    if neighbor not in visited:
                        new_dist = dist + edge_lengths[current][neighbor]
                        new_visited = visited.copy()
                        new_visited.add(neighbor)
                        forward_stack.append((
                            neighbor,
                            path + [neighbor],
                            depth + 1,
                            new_dist,
                            new_visited
                        ))

        # Process backward DFS
        if backward_stack:
            current, path, depth, dist, visited = backward_stack.pop()
            backward_visited[current] = (path, dist)

            # Check if current node exists in forward search results
            if current in forward_visited:
                forw_path, forw_dist = forward_visited[current]
                total_dist = forw_dist + dist
                if (target_distance - tolerance <= total_dist <= target_distance + tolerance):
                    full_path = forw_path[:-1] + path[::-1]
                    yield full_path

            # Expand backward neighbors (using predecessors)
            if depth < cutoff and dist <= target_distance + tolerance:
                for predecessor in G.predecessors(current):
                    if predecessor not in visited:
                        new_dist = dist + reverse_edge_lengths[current][predecessor]
                        new_visited = visited.copy()
                        new_visited.add(predecessor)
                        backward_stack.append((
                            predecessor,
                            path + [predecessor],
                            depth + 1,
                            new_dist,
                            new_visited
                        ))


def get_all_paths_fitness(G, start_node, end_node, cutoff=25, target_distance=10, tolerance=1):
    target_distance *= 1000
    tolerance *= 1000
    edge_lengths = {}
    for u, v, data in G.edges(data=True):
        edge_lengths.setdefault(u, {})[v] = data['length']

    # Initialize stack with sorted successors by fitness in descending order
    start_successors = sorted(G.successors(start_node),
                          key=lambda v: G[start_node][v][0]['fitness'], reverse=True)
    stack = [(start_node, iter(start_successors), None, 0, 0, 0)]
    path = [start_node]
    visited = {start_node}
    while stack:
        current_node, neighbors, prev_node, depth, cumulative, cum_fitness = stack[-1]
        try:
            neighbor = next(neighbors)
        except StopIteration:
            # Backtrack: remove current node from path and visited
            stack.pop()
            if path:
                removed = path.pop()
                visited.remove(removed)
        else:
            if neighbor in visited:
                continue

            new_cumulative = cumulative
            new_fitness = cum_fitness
            if len(path) >= 1:
                new_cumulative += G[current_node][neighbor][0]['length']
                new_fitness += G[current_node][neighbor][0]['fitness']

            visited.add(neighbor)
            path.append(neighbor)
            if depth >= cutoff or new_cumulative > target_distance + tolerance:
                path.pop()
                visited.remove(neighbor)
                continue

            if neighbor == end_node:
                if target_distance - tolerance <= new_cumulative <= target_distance + tolerance:
                    yield list(path)

            # Sort the next node's successors by fitness before adding to stack
            neighbor_successors = sorted(G.successors(neighbor),
                                   key=lambda v: G[neighbor][v][0]['fitness'], reverse=True)
            stack.append((neighbor, iter(neighbor_successors), current_node, depth + 1, new_cumulative, new_fitness))


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

# process_graph_weightings(G)



# print("Generating Route")
# t = time.time()
# x = get_all_paths(G, start_node, end_node, 50, 15)
# print(time.time() - t)

class DFS_Generator:
    def __init__(self, distance_km, tolerance):
        self.graph_builder = Graph_Builder((53.36486137451511, -1.8160056925378616), (53.34344386440596, -1.778107050662822))
        self.G = self.graph_builder.get_graph()
        self.generator = get_all_paths_bidirectional(self.G, start_node, end_node, 50, target_distance=distance_km)
        self.target_distance = distance_km
        self.tolerance = tolerance
        print("setup done")
    
    def get_generated_route(self):
        t = time.time()
        print("time to generate")
        for path in self.generator:
            distance = calculate_path_distance(G, path) / 1000
            if self.target_distance - self.tolerance <= distance <= self.target_distance + self.tolerance:
                print(time.time() - t)
                print(f"Route Found\nDistance: {distance}")
                ox.plot_graph_route(G, path, route_linewidth=6, node_size=0, bgcolor='k')
                # return path
            
    def route_to_coords(self, route):
        return [(G.nodes[node]['y'], G.nodes[node]['x']) for node in route]
            
    def generate_route(self):
        route = self.get_generated_route()
        return self.route_to_coords(route)
    

def dfs_gen(distance_km, tolerance=1):
    t = time.time()
    for path in get_all_paths_bidirectional(G, start_node, end_node, 50, target_distance=distance_km):
        distance = calculate_path_distance(G, path) / 1000
        # print(distance)
        # fig, ax = ox.plot_graph_route(G, path, route_linewidth=6, node_size=0, bgcolor='k')
        if distance_km - tolerance <= distance <= distance_km + tolerance:
            print(time.time() - t)
            print(f"Route Found\nDistance: {distance}")
            return path
            # print(len(path))
            # fig, ax = ox.plot_graph_route(G, path, route_linewidth=6, node_size=0, bgcolor='k')
            # t = time.time()


# Old one
def route_to_coords(route):
    return [(G.nodes[node]['y'], G.nodes[node]['x']) for node in route]

def generate_route(distance):
    route = dfs_gen(distance)
    return route_to_coords(route)


if __name__ == "__main__":
    t = time.time()
    # route = dfs_gen(10)
    gen = DFS_Generator(10, 1)
    route = gen.get_generated_route()
    print(time.time() - t)
    print(route)
    ox.plot_graph_route(G, route, route_linewidth=6, node_size=0, bgcolor='k')
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
