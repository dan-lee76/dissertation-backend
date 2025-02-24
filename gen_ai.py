import osmnx as ox
import networkx as nx
import random
import numpy as np
from graph_preperation import Graph_Builder


class StandaloneMemeticRouter:
    def __init__(self,
                 north_east_coord=(53.36486137451511, -1.8160056925378616),
                 south_west_coord=(53.34344386440596, -1.778107050662822),
                 population_size=50,
                 generations=100,
                 mutation_rate=0.3,
                 target_distance=10000,
                 peak_search_radius=500):

        # Initialize graph builder with coordinates
        self.graph_builder = Graph_Builder(north_east_coord, south_west_coord)
        self.G = self.graph_builder.get_graph()

        # Get start/end nodes from graph builder
        self.start_node = self.graph_builder.start_node
        self.end_node = self.graph_builder.end_node

        # Algorithm parameters
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.target_distance = target_distance
        self.peak_search_radius = peak_search_radius

        # Precompute important nodes
        self.peak_nodes = [n for n in self.G.nodes
                           if self.graph_builder.is_peak(n)]

        # Cache for local search
        self._distance_cache = {}

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = self._generate_biased_random_walk()
            population.append(individual)
        return population

    def _generate_biased_random_walk(self):
        individual = [self.start_node]
        current_distance = 0
        current_node = self.start_node

        while current_distance < self.target_distance * 1.2:
            neighbors = list(self.G.neighbors(current_node))

            # Bias selection towards nodes with higher fitness
            weights = []
            for n in neighbors:
                node_data = self.G.nodes[n]
                weight = node_data.get('fitness', 1) + 0.1  # Ensure non-zero
                # Penalize revisiting nodes
                if n in individual:
                    weight *= 0.1
                weights.append(weight)

            # Normalize weights
            total_weight = sum(weights)
            probabilities = [w / total_weight for w in weights]

            next_node = np.random.choice(neighbors, p=probabilities)
            edge_length = self.G[current_node][next_node][0]['length']

            if current_distance + edge_length > self.target_distance * 1.2:
                break

            individual.append(next_node)
            current_distance += edge_length
            current_node = next_node

        return individual

    def route_distance(self, route):
        """Memoized route distance calculation"""
        route_tuple = tuple(route)
        if route_tuple not in self._distance_cache:
            distance = 0
            for u, v in zip(route[:-1], route[1:]):
                distance += self.G[u][v][0]['length']
            self._distance_cache[route_tuple] = distance
        return self._distance_cache[route_tuple]

    def fitness(self, route):
        try:
            # Base fitness from graph attributes
            base_fitness = self.graph_builder.get_route_fitness(route)

            # Distance penalty
            distance = self.route_distance(route)
            distance_penalty = -abs(self.target_distance - distance) * 2

            # Peak bonus
            peak_count = sum(1 for n in route if self.graph_builder.is_peak(n))
            peak_bonus = peak_count * 5000

            # End proximity bonus
            end_dist = self._distance_to_end(route[-1])
            end_penalty = -end_dist * 0.1

            return (base_fitness + distance_penalty + peak_bonus + end_penalty) / 1000
        except:
            return -np.inf

    def _distance_to_end(self, node):
        """Euclidean distance to end node in meters"""
        y1, x1 = self.G.nodes[node]['y'], self.G.nodes[node]['x']
        y2, x2 = self.G.nodes[self.end_node]['y'], self.G.nodes[self.end_node]['x']
        return ox.distance.great_circle_vec(y1, x1, y2, x2) * 1000

    def selection(self, population):
        scores = np.array([self.fitness(ind) for ind in population])
        scaled_scores = np.exp(scores - np.max(scores))  # For numerical stability
        probabilities = scaled_scores / np.sum(scaled_scores)
        return random.choices(population, weights=probabilities, k=2)

    def crossover(self, parent1, parent2):
        # Find common nodes that are peaks or high-fitness
        common_nodes = set(parent1) & set(parent2)
        if not common_nodes:
            return parent1 if self.fitness(parent1) > self.fitness(parent2) else parent2

        # Prefer crossover points at peaks
        crossover_candidates = [n for n in common_nodes
                                if self.graph_builder.is_peak(n)]
        if not crossover_candidates:
            crossover_candidates = list(common_nodes)

        crossover_point = random.choice(crossover_candidates)
        idx1 = parent1.index(crossover_point)
        idx2 = parent2.index(crossover_point)

        # Create hybrid child
        child = parent1[:idx1] + parent2[idx2:]
        return self._repair_route(child)

    def _repair_route(self, route):
        """Ensure route connectivity using A*"""
        repaired = [route[0]]
        for i in range(1, len(route)):
            try:
                path = nx.astar_path(self.G, repaired[-1], route[i], weight='length')
                repaired += path[1:]
            except nx.NetworkXNoPath:
                continue
        return repaired

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            # 70% chance of peak-focused mutation
            if random.random() < 0.7 and self.peak_nodes:
                return self._peak_based_mutation(individual)
            else:
                return self._random_mutation(individual)
        return individual

    def _peak_based_mutation(self, individual):
        idx = random.randint(0, len(individual) - 1)
        current_node = individual[idx]

        # Find nearest unexplored peak
        nearest_peak = min(
            (p for p in self.peak_nodes if p not in individual),
            key=lambda p: self._distance_between_nodes(current_node, p),
            default=None
        )

        if nearest_peak:
            try:
                path_to_peak = nx.astar_path(self.G, current_node, nearest_peak, weight='length')
                path_from_peak = nx.astar_path(self.G, nearest_peak, individual[-1], weight='length')
                new_route = individual[:idx] + path_to_peak + path_from_peak[1:]
                return self._trim_route(new_route)
            except nx.NetworkXNoPath:
                pass
        return individual

    def _random_mutation(self, individual):
        idx = random.randint(0, len(individual) - 2)
        try:
            neighbors = list(self.G.neighbors(individual[idx]))
            new_node = random.choice(neighbors)
            path = nx.astar_path(self.G, new_node, individual[idx + 1], weight='length')
            return individual[:idx] + [new_node] + path[1:]
        except:
            return individual

    def _trim_route(self, route):
        """Ensure route doesn't exceed distance limits"""
        current_length = self.route_distance(route)
        if current_length <= self.target_distance * 1.2:
            return route

        # Remove nodes from start until under limit
        for i in range(1, len(route)):
            trimmed = route[i:]
            if self.route_distance(trimmed) <= self.target_distance * 1.2:
                return trimmed
        return route

    def local_search(self, individual):
        best_route = individual.copy()
        best_score = self.fitness(best_route)

        # Look for peak insertion opportunities
        for i in range(len(individual)):
            current_node = individual[i]
            nearby_peaks = self._find_nearby_peaks(current_node, individual)

            for peak in nearby_peaks:
                modified = self._insert_peak(individual, i, peak)
                if modified and self.fitness(modified) > best_score:
                    best_route = modified
                    best_score = self.fitness(modified)

        # Try shortcutting low-fitness segments
        for i in range(0, len(best_route) - 2):
            for j in range(i + 2, len(best_route)):
                try:
                    shortcut = nx.astar_path(self.G, best_route[i], best_route[j], weight='length')
                    if len(shortcut) < (j - i):
                        new_route = best_route[:i] + shortcut + best_route[j + 1:]
                        if self.fitness(new_route) > best_score:
                            best_route = new_route
                            best_score = self.fitness(new_route)
                except nx.NetworkXNoPath:
                    continue

        return best_route

    def _find_nearby_peaks(self, origin_node, existing_route):
        peaks = []
        for peak in self.peak_nodes:
            if peak in existing_route:
                continue
            try:
                dist = nx.shortest_path_length(self.G, origin_node, peak, weight='length')
                if dist <= self.peak_search_radius:
                    peaks.append(peak)
            except nx.NetworkXNoPath:
                continue
        return peaks

    def _insert_peak(self, route, index, peak_node):
        try:
            path_to = nx.astar_path(self.G, route[index], peak_node, weight='length')
            path_from = nx.astar_path(self.G, peak_node, route[index + 1], weight='length')
            new_route = route[:index] + path_to + path_from[1:] + route[index + 1:]
            return self._trim_route(new_route)
        except nx.NetworkXNoPath:
            return None

    def evolve(self):
        population = self.initialize_population()

        for generation in range(self.generations):
            # Evaluate and sort population
            population = sorted(population, key=lambda x: self.fitness(x), reverse=True)

            # Keep top 20% as elites
            elites = population[:int(self.population_size * 0.2)]

            # Generate new population
            new_population = elites.copy()
            while len(new_population) < self.population_size:
                parents = self.selection(population)
                child = self.crossover(parents[0], parents[1])
                child = self.mutate(child)
                # child = self.local_search(child)
                new_population.append(child)

            population = new_population[:self.population_size]

            # Early exit if we find a perfect route
            best = max(population, key=self.fitness)
            if abs(self.route_distance(best) - self.target_distance) < 100:
                break

        return max(population, key=self.fitness)

    def get_route_coordinates(self, route):
        return [(self.G.nodes[n]['y'], self.G.nodes[n]['x']) for n in route]


def generate_optimized_route():
    router = StandaloneMemeticRouter(
        north_east_coord=(53.36486137451511, -1.8160056925378616),
        south_west_coord=(53.34344386440596, -1.778107050662822),
        target_distance=10000,
        population_size=100,
        generations=200,
        peak_search_radius=800
    )

    best_route = router.evolve()
    return router.get_route_coordinates(best_route)


if __name__ == "__main__":
    route_coords = generate_optimized_route()
    print("Optimized route coordinates:", route_coords)

    # To visualize with OSMnx:
    G = Graph_Builder((53.36486137451511, -1.8160056925378616),
                      (53.34344386440596, -1.778107050662822)).get_graph()
    ox.plot.plot_graph_route(G, [n[0] for n in route_coords])