import osmnx as ox
import networkx as nx
import random
import copy

# For reproducibility (optional)
random.seed(42)


def route_length(G, route):
    """
    Compute the total length (in meters) of a given route.
    """
    x = []
    for u, v in zip(route[:-1], route[1:]):
        x.append(G[u][v][0]['length'])
    return sum(x)


def generate_initial_route(G, start, goal):
    """
    Generate an initial route from start to goal using the shortest path.
    """
    try:
        route = nx.shortest_path(G, start, goal, weight='length')
    except nx.NetworkXNoPath:
        raise Exception("No path exists between the start and goal nodes!")
    return route


def generate_neighbor_route(G, route, start, goal, max_attempts=10):
    """
    Generate a neighbor candidate route by replacing one segment of the current route
    with an alternative detour. This method preserves the fixed endpoints (start and goal).

    The idea is to pick an edge (u,v) in the route (except for the first and last edges)
    and then try to find an alternative sub-path from u to v that is different from the
    direct edge. The candidate route is formed by splicing this alternative path in.
    """
    new_route = None
    for attempt in range(max_attempts):
        if len(route) < 3:
            break  # Not enough nodes to modify without altering endpoints.
        # Select an edge index, avoiding the very first or very last edge so endpoints remain fixed.
        i = random.randint(1, len(route) - 3)
        u = route[i]
        v = route[i + 1]

        # Make a copy of the graph and remove the direct edge (u,v) to force a detour.
        G_removed = G.copy()
        if G_removed.has_edge(u, v):
            G_removed.remove_edge(u, v)
        # In case of directed graphs, you might want to also remove the reverse edge.
        if G_removed.is_directed() and G_removed.has_edge(v, u):
            G_removed.remove_edge(v, u)

        try:
            alt_path = nx.shortest_path(G_removed, u, v, weight='length')
            # Ensure that the alternative path is indeed a detour (has at least one extra node).
            if len(alt_path) > 2:
                # Construct the new route by splicing the alternative segment into the route.
                new_route = route[:i + 1] + alt_path[1:] + route[i + 2:]
                # Validate that new_route still connects start and goal.
                if new_route[0] == start and new_route[-1] == goal:
                    break
        except nx.NetworkXNoPath:
            continue  # Try another edge if no detour was found.

    if new_route is None:
        # If no modification is found, return the original route.
        new_route = route
    return new_route


def tabu_search(G, start, goal, target_length, iterations=100, tabu_tenure=5, neighbors_per_iter=10):
    """
    Run a tabu search to generate a point-to-point hiking route from start to goal that
    has a total length as close as possible to target_length (in meters).

    Parameters:
      G                - the graph (from OSMnx)
      start            - the starting node
      goal             - the destination node
      target_length    - desired route length (in meters)
      iterations       - number of iterations to run
      tabu_tenure      - number of candidate routes to remember in the tabu list
      neighbors_per_iter - number of neighbor routes to generate at each iteration

    Returns:
      best_route       - the route (list of nodes) found with an objective value closest to target_length.
    """
    current_route = generate_initial_route(G, start, goal)
    best_route = current_route
    best_obj = abs(route_length(G, current_route) - target_length)

    # Use a list of route tuples as the tabu list.
    tabu_list = []

    for it in range(iterations):
        neighbor_candidates = []
        for _ in range(neighbors_per_iter):
            candidate_route = generate_neighbor_route(G, current_route, start, goal)
            neighbor_candidates.append(candidate_route)

        # Select the best candidate not in the tabu list.
        candidate = None
        candidate_obj = float('inf')
        for cand in neighbor_candidates:
            if tuple(cand) in tabu_list:
                continue
            obj = abs(route_length(G, cand) - target_length)
            if obj < candidate_obj:
                candidate_obj = obj
                candidate = cand

        # If all candidates are in the tabu list, choose a random one.
        if candidate is None:
            candidate = random.choice(neighbor_candidates)
            candidate_obj = abs(route_length(G, candidate) - target_length)

        current_route = candidate
        tabu_list.append(tuple(candidate))
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)

        # Update the best route if the current candidate is closer to the target length.
        if candidate_obj < best_obj:
            best_obj = candidate_obj
            best_route = candidate

        print(f"Iteration {it + 1}: Candidate length = {route_length(G, candidate):.1f} m, "
              f"Difference from target = {candidate_obj:.1f} m")
        fig, ax = ox.plot_graph_route(G, candidate, route_linewidth=4, node_size=0)
    return best_route


def main():
    # Define the area of interest.
    # You can change this to any area or place name.
    place = "Peak District, United Kingdom"
    print(f"Downloading graph data for {place} …")
    G = ox.graph_from_point((53.36486137451511, -1.8160056925378616), dist=5000, network_type='walk', simplify=True)

    # For a point-to-point route, choose two nodes: one for start and one for goal.
    nodes = list(G.nodes)
    start_node = ox.nearest_nodes(G, X=-1.8160056925378616, Y=53.36486137451511)
    goal_node = ox.nearest_nodes(G, Y=53.34344386440596, X=-1.778107050662822)
    # Ensure the goal node is different from the start.
    # goal_node = start_node
    # while goal_node == start_node:
    #     goal_node = random.choice(nodes)

    print(f"Using start node: {start_node}")
    print(f"Using goal node: {goal_node}")

    # Set your target route length in meters (e.g., 7000 m for a 7 km hike).
    target_length = 10000

    print(f"Searching for a route from start to goal near {target_length} m …")
    best_route = tabu_search(G, start_node, goal_node, target_length,
                             iterations=50, tabu_tenure=5, neighbors_per_iter=10)

    best_route_length = route_length(G, best_route)
    print(f"Best route found: {len(best_route)} nodes, total length = {best_route_length:.1f} m")

    # Plot the best route.
    fig, ax = ox.plot_graph_route(G, best_route, route_linewidth=4, node_size=0)


if __name__ == '__main__':
    main()
