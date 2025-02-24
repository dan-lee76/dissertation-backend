import osmnx as ox
from collections import defaultdict, deque

from graph_preperation import Graph_Builder


class HikingRouteFinder:
    def __init__(self, graph, graph_builder):
        self.G = graph
        self.graph_builder = graph_builder
        self.best_fitness = float('inf')
        self.best_path = None
        self.target_distance = 10000  # 10km
        self.tolerance = 500  # ±500m
        self.min_dist = self.target_distance - self.tolerance
        self.max_dist = self.target_distance + self.tolerance

    def calculate_node_fitness(self, node):
        nature = self.G.nodes[node].get('nature', {})
        green_score = self.G.nodes[node].get('green_score', 0)
        elevation = self.G.nodes[node].get('elevation', 0)
        if self.graph_builder.is_peak(node) or nature.get('natural') == 'peak':
            fitness = abs(10000 - (green_score * elevation * 2))
        else:
            fitness = abs(10000 - (green_score * elevation * 1))
        return fitness

    def find_best_route(self, start, end):
        forward_stack = deque()
        start_fitness = self.calculate_node_fitness(start)
        forward_stack.append((start, [start], 0, start_fitness))

        backward_stack = deque()
        end_fitness = self.calculate_node_fitness(end)
        backward_stack.append((end, [end], 0, end_fitness))

        forward_visited = defaultdict(list)
        backward_visited = defaultdict(list)

        forward_visited[start].append(([start], 0, start_fitness))
        backward_visited[end].append(([end], 0, end_fitness))

        while forward_stack and backward_stack:
            # Process forward direction
            if forward_stack:
                current_node, path, dist, fit = forward_stack.pop()
                # Check against backward_visited
                if current_node in backward_visited:
                    for (back_path, back_dist, back_fit) in backward_visited[current_node]:
                        total_dist = dist + back_dist
                        if self.min_dist <= total_dist <= self.max_dist:
                            node_fitness = self.calculate_node_fitness(current_node)
                            total_fit = fit + back_fit - node_fitness
                            if total_fit < self.best_fitness:
                                self.best_fitness = total_fit
                                combined_path = path + back_path[:-1][::-1]
                                self.best_path = combined_path
                # Expand neighbors (forward direction uses outgoing edges)
                for neighbor in self.G.neighbors(current_node):
                    if neighbor in path:
                        continue
                    edge_data = self.G.get_edge_data(current_node, neighbor)
                    if not edge_data:
                        continue
                    edge_length = edge_data[0]['length']
                    new_dist = dist + edge_length
                    if new_dist > self.max_dist:
                        continue
                    new_fit = fit + self.calculate_node_fitness(neighbor)
                    new_path = path + [neighbor]
                    # Check if this path is worth adding
                    add = True
                    for (existing_path, existing_dist, existing_fit) in forward_visited.get(neighbor, []):
                        if existing_dist <= new_dist and existing_fit <= new_fit:
                            add = False
                            break
                    if add:
                        forward_stack.append((neighbor, new_path, new_dist, new_fit))
                        forward_visited[neighbor].append((new_path, new_dist, new_fit))

            # Process backward direction
            if backward_stack:
                current_node, path, dist, fit = backward_stack.pop()
                # Check against forward_visited
                if current_node in forward_visited:
                    for (forward_path, forward_dist, forward_fit) in forward_visited[current_node]:
                        total_dist = forward_dist + dist
                        if self.min_dist <= total_dist <= self.max_dist:
                            node_fitness = self.calculate_node_fitness(current_node)
                            total_fit = forward_fit + fit - node_fitness
                            if total_fit < self.best_fitness:
                                self.best_fitness = total_fit
                                combined_path = forward_path + path[:-1][::-1]
                                self.best_path = combined_path
                # Expand predecessors (backward direction uses incoming edges)
                for predecessor in self.G.predecessors(current_node):
                    if predecessor in path:
                        continue
                    edge_data = self.G.get_edge_data(predecessor, current_node)
                    if not edge_data:
                        continue
                    edge_length = edge_data[0]['length']
                    new_dist = dist + edge_length
                    if new_dist > self.max_dist:
                        continue
                    new_fit = fit + self.calculate_node_fitness(predecessor)
                    new_path = path + [predecessor]
                    # Check if this path is worth adding
                    add = True
                    for (existing_path, existing_dist, existing_fit) in backward_visited.get(predecessor, []):
                        if existing_dist <= new_dist and existing_fit <= new_fit:
                            add = False
                            break
                    if add:
                        backward_stack.append((predecessor, new_path, new_dist, new_fit))
                        backward_visited[predecessor].append((new_path, new_dist, new_fit))

        return self.best_path, self.best_fitness

graph_builder = Graph_Builder((53.36486137451511, -1.8160056925378616), (53.34344386440596, -1.778107050662822))
graph = graph_builder.get_graph()
HRF = HikingRouteFinder(graph, graph_builder)
start_node = ox.nearest_nodes(graph, X=-1.8160056925378616, Y=53.36486137451511)
end_node = ox.nearest_nodes(graph, Y=53.34344386440596, X=-1.778107050662822)
best_route = HRF.find_best_route(start_node, end_node)
print(best_route)
ox.plot_graph_route(graph, best_route[0], route_linewidth=4, node_size=0)
