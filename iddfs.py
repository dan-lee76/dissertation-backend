import heapq
import time
import networkx as nx
import osmnx as ox
ox.settings.use_cache = True
ox.settings.cache_folder = "/_cache"
t= time.time()
G = ox.graph_from_point((53.36486137451511, -1.8160056925378616), dist=5000, network_type='walk', simplify=True)
print (time.time() - t)
# fig, ax = ox.plot_graph(G)

start_node = ox.nearest_nodes(G, X=-1.8160056925378616, Y=53.36486137451511)
end_node = ox.nearest_nodes(G, Y=53.34344386440596, X=-1.778107050662822)

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

distance_km = 10
tollerance = 0.25
print("Generating Route")
t = time.time()
print(time.time() - t)


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

def iddfs(graph, start_node, end_node, maxDepth, target_distance):
    t = time.time()
    for depth in range(1, maxDepth):
        print(f"Depth: {depth}")
        # for path in nx.all_simple_paths(G, start_node, end_node, depth):
        for path in get_all_paths_bidirectional(graph, start_node, end_node, depth, target_distance):
            distance = calculate_path_distance(G, path) / 1000
            # print(f"Distance: {distance}")
            # fig, ax = ox.plot_graph_route(G, path, route_linewidth=6, node_size=0, bgcolor='k')
            if target_distance - tollerance <= distance <= target_distance + tollerance:
                print(time.time() - t)
                print(f"Route Found\nDistance: {distance}")
                print(len(path))
                # fig, ax = ox.plot_graph_route(G, path, route_linewidth=6, node_size=0, bgcolor='k')
                t = time.time()
                return path



def route_to_coords(route):
    return [(G.nodes[node]['y'], G.nodes[node]['x']) for node in route]

def generate_route(distance):
    route = iddfs(G, start_node, end_node, 50, distance)
    return route_to_coords(route)
    
    
if __name__ == "__main__":
    route = iddfs(G, start_node, end_node, 50, 10)
    ox.plot_graph_route(G, route, route_linewidth=6, node_size=0, bgcolor='k')
    print("done")