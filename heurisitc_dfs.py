import heapq
import random
import time

from graph_preperation import Graph_Builder

t = time.time()
import networkx as nx
import osmnx as ox

print(time.time() - t)
ox.settings.use_cache = True
ox.settings.cache_folder = "/_cache"
# print("Cache enabled:", ox.settings.use_cache)
# print("Cache folder:", ox.settings.cache_folder)
# ox.settings.cache_only_mode = True
# G = ox.graph_from_place('Edale, Derbyshire, England', network_type='walk')
t = time.time()
# Downloads all of the walkable streets
G = ox.graph_from_point((53.36486137451511, -1.8160056925378616), dist=5000, network_type='walk', simplify=True)
print(time.time() - t)
# fig, ax = ox.plot_graph(G)

start_node = ox.nearest_nodes(G, X=-1.8160056925378616, Y=53.36486137451511)
end_node = ox.nearest_nodes(G, Y=53.34344386440596, X=-1.778107050662822)


def calculate_path_distance(G, path):
    x = []
    for u, v in zip(path[:-1], path[1:]):
        x.append(G[u][v][0]['length'])
    return sum(x)



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

    print("Starting DFS")
    # Initialize stacks for forward and backward DFS
    forward_stack = [(start_node, [start_node], 0, 0, {start_node}, 0)]
    backward_stack = [(end_node, [end_node], 0, 0, {end_node}, 0)]
    print(forward_stack)
    print(backward_stack)
    while forward_stack or backward_stack:
        # Process forward DFS
        # print("In the stack")
        if forward_stack:
            current, path, depth, dist, visited, cum_fitness = forward_stack.pop()
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
                neighbor_successors = sorted(G.successors(current), key=lambda v: G[current][v][0]['fitness'],
                                             reverse=False)
                for neighbor in neighbor_successors:
                    if neighbor not in visited:
                        new_dist = dist + edge_lengths[current][neighbor]
                        new_visited = visited.copy()
                        new_visited.add(neighbor)
                        new_fitness = G[current][neighbor][0]['fitness'] + cum_fitness
                        forward_stack.append((
                            neighbor,
                            path + [neighbor],
                            depth + 1,
                            new_dist,
                            new_visited,
                            new_fitness
                        ))

        # Process backward DFS
        if backward_stack:
            current, path, depth, dist, visited, cum_fitness = backward_stack.pop()
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
                neighbor_predecessors = sorted(G.predecessors(current), key=lambda v: G[v][current][0]['fitness'], reverse=False)
                for predecessor in neighbor_predecessors:
                    if predecessor not in visited:
                        new_dist = dist + reverse_edge_lengths[current][predecessor]
                        new_visited = visited.copy()
                        new_visited.add(predecessor)
                        new_fitness = G[predecessor][current][0]['fitness'] + cum_fitness
                        backward_stack.append((
                            predecessor,
                            path + [predecessor],
                            depth + 1,
                            new_dist,
                            new_visited,
                            new_fitness
                        ))

def compute_lookahead_fitness(G, node, depth):
    total = 0
    current = node
    for _ in range(depth):
        next_nodes = list(G.successors(current))
        if not next_nodes:
            break
        next_node = max(next_nodes, key=lambda v: G[current][v][0]['fitness'])
        total += G[current][next_node][0]['fitness']
        current = next_node
    return total

def get_all_paths_fitness_peak(G, start_node, end_node, cutoff=25, target_distance=10, tolerance=1):
    target_distance *= 1000
    tolerance *= 1000
    edge_lengths = {}
    for u, v, data in G.edges(data=True):
        edge_lengths.setdefault(u, {})[v] = data['length']

    # Initialize stack with sorted successors by 3-step lookahead fitness
    start_successors = sorted(
        G.successors(start_node),
        key=lambda v: (G[start_node][v][0]['fitness'] + compute_lookahead_fitness(G, v, 2)),
        reverse=True
    )
    stack = [(start_node, iter(start_successors), None, 0, 0, 0)]
    path = [start_node]
    visited = {start_node}
    print("Starting")
    while stack:
        current_node, neighbors, prev_node, depth, cumulative, cum_fitness = stack[-1]
        try:
            neighbor = next(neighbors)
        except StopIteration:
            # Backtrack
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

            # Sort the next node's successors by 3-step lookahead
            neighbor_successors = sorted(
                G.successors(neighbor),
                key=lambda v: (G[neighbor][v][0]['fitness'] + compute_lookahead_fitness(G, v, 2)),
                reverse=True
            )
            stack.append((neighbor, iter(neighbor_successors), current_node, depth + 1, new_cumulative, new_fitness))

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

class DFS_Generator:
    def __init__(self, distance_km, tolerance):
        self.graph_builder = Graph_Builder((53.36486137451511, -1.8160056925378616),
                                           (53.34344386440596, -1.778107050662822))
        self.G = self.graph_builder.get_graph()
        self.generator = get_all_paths_fitness_peak(self.G, start_node, end_node, 30, target_distance=distance_km)
        self.target_distance = distance_km
        self.tolerance = tolerance
        print("setup done")

    def get_generated_route(self):
        t = time.time()
        print("time to generate")
        routes = []
        c = 0
        for path in self.generator:
            distance = calculate_path_distance(G, path) / 1000
            print(distance)
            if self.target_distance - self.tolerance <= distance <= self.target_distance + self.tolerance:
                # print(time.time() - t)
                # print(f"Route Found\nDistance: {distance}")
                # ox.plot_graph_route(G, path, route_linewidth=6, node_size=0, bgcolor='k')
                routes.append([path, self.graph_builder.get_route_fitness(path), len(self.graph_builder.get_peak_nodes(path))])
                c+=1
                print(c)
                if c > 100:
                    break
                # return path
            # if distance > self.target_distance + self.tolerance:
            #     break
        print(len(routes))
        route1 = min(routes, key=lambda x: x[1])[0]
        route2 = max(routes, key=lambda x: x[1])[0]
        ox.plot_graph_route(self.G, route1, route_linewidth=6, node_size=0, bgcolor='k')
        ox.plot_graph_route(self.G, route2, route_linewidth=6, node_size=0, bgcolor='k')
        route1 = min(routes, key=lambda x: x[2])[0]
        route2 = max(routes, key=lambda x: x[2])[0]
        ox.plot_graph_route(self.G, route1, route_linewidth=6, node_size=0, bgcolor='k')
        ox.plot_graph_route(self.G, route2, route_linewidth=6, node_size=0, bgcolor='k')

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


