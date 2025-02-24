import osmnx as ox
import networkx as nx
import random
from collections import deque
import time

# ox.config(use_cache=True, log_console=True)

TARGET_DISTANCE = 10000  # 10km in meters


def get_graph(area_name):
    """Download and prepare the graph for a given area."""
    G = ox.graph_from_point((53.36486137451511, -1.8160056925378616), dist=5000, network_type='walk', simplify=True)
    return G


def calculate_fitness(path, G):
    """Calculate fitness based on deviation from target distance."""
    if not path or len(path) < 2:
        return float('inf'), 0

    total_distance = sum(ox.utils_graph.get_route_edge_attributes(G, path, 'length'))
    deviation = abs(total_distance - TARGET_DISTANCE)
    return deviation, total_distance


def generate_neighbor(current_path, G, tabu_list):
    """Generate a neighbor route with path perturbation."""
    if len(current_path) < 3:
        return current_path, None

    # Randomly select a perturbation point
    split_point = random.randint(1, len(current_path) - 2)
    current_node = current_path[split_point]

    if current_node in tabu_list:
        return current_path, None

    # Find alternative routes around the perturbation point
    try:
        # Create temporary graph with slightly modified weights
        temp_G = G.copy()
        for u, v, data in temp_G.edges(data=True):
            data['temp_weight'] = data['length'] * random.uniform(0.8, 1.2)

        # Find new path segment with perturbed weights
        new_segment = nx.shortest_path(temp_G, current_path[0], current_path[-1], weight='temp_weight')
    except nx.NetworkXNoPath:
        return current_path, None

    return new_segment, current_node


def tabu_search(start_point, end_point, G, max_iterations=1000, tabu_tenure=10):
    """Tabu Search implementation for target distance route finding."""
    start_node = ox.nearest_nodes(G, X=-1.8160056925378616, Y=53.36486137451511)
    end_node = ox.nearest_nodes(G, Y=53.34344386440596, X=-1.778107050662822)

    # Initial solution (shortest path)
    try:
        initial_path = nx.shortest_path(G, start_node, end_node, weight='length')
    except nx.NetworkXNoPath:
        raise ValueError("No initial path found between the points")

    tabu_list = deque(maxlen=tabu_tenure)
    current_path = initial_path
    current_deviation, current_distance = calculate_fitness(current_path, G)

    best_path = current_path.copy()
    best_deviation = current_deviation

    for iteration in range(max_iterations):
        neighbor_path, tabu_candidate = generate_neighbor(current_path, G, tabu_list)

        if neighbor_path == current_path:
            continue

        neighbor_deviation, neighbor_distance = calculate_fitness(neighbor_path, G)

        # Acceptance criteria
        if (neighbor_deviation < current_deviation) or \
                (neighbor_deviation == current_deviation and len(neighbor_path) > len(current_path)):

            current_path = neighbor_path
            current_deviation = neighbor_deviation
            current_distance = neighbor_distance
            tabu_list.append(tabu_candidate)

            if neighbor_deviation < best_deviation:
                best_path = neighbor_path.copy()
                best_deviation = neighbor_deviation

        # Occasionally allow worse solutions to escape local optima
        elif random.random() < 0.1:
            current_path = neighbor_path
            current_deviation = neighbor_deviation
            current_distance = neighbor_distance
            tabu_list.append(tabu_candidate)

    return best_path


# Example usage
if __name__ == "__main__":
    # Define area and coordinates (Snowdonia National Park example)
    AREA = "Snowdonia National Park, UK"
    START_POINT = (53.0685, -4.0763)  # Llanberis coordinates
    END_POINT = (53.1396, -3.8031)  # Capel Curig coordinates

    # Get graph data
    G = get_graph(AREA)

    # Run Tabu Search
    start_time = time.time()
    best_route = tabu_search(START_POINT, END_POINT, G)
    elapsed_time = time.time() - start_time

    # Calculate results
    deviation, distance = calculate_fitness(best_route, G)

    print(f"\nOptimized Route Found:")
    print(f"Target distance: {TARGET_DISTANCE / 1000} km")
    print(f"Achieved distance: {distance / 1000:.2f} km")
    print(f"Deviation: {abs(distance - TARGET_DISTANCE) / 1000:.2f} km")
    print(f"Computation time: {elapsed_time:.2f} seconds")

    # Plot the route
    m = ox.plot_route_folium(G, best_route, route_color='#009688')
    m.save('ideal_hiking_route.html')