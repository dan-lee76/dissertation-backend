import time

import osmnx as ox
import networkx as nx
import random
import numpy as np
import osmnx.distance
from graph_preperation import Graph_Builder


class MemeticAlgorithm:
    def __init__(self, start_node, end_node, target_distance, population_size=100, generations=10,
                 mutation_rate=0.2):
        self.graph_builder = Graph_Builder((53.36486137451511, -1.8160056925378616),
                                           (53.34344386440596, -1.778107050662822))
        self.graph = self.graph_builder.get_graph()
        self.start_node = start_node
        self.end_node = end_node
        self.target_distance = target_distance
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def euclidean_distance(self, node1, node2):
        return abs(osmnx.distance.euclidean(self.graph.nodes[node1]['y'], self.graph.nodes[node1]['x'],
                                            self.graph.nodes[node2]['y'], self.graph.nodes[node2]['x']))

    def initialize_population(self):
        population = []
        heuristic = {node: self.euclidean_distance(node, self.end_node)
                     for node in self.graph.nodes()}
        print(f"Heuristic: {heuristic}")
        for _ in range(self.population_size):
            individual = [self.start_node]
            visited = [self.start_node]
            while True:
                neighbors = list(self.graph.neighbors(individual[-1]))
                # print(f"Neighbors: {neighbors}")
                neighbors = [neighbor for neighbor in neighbors if neighbor not in visited]
                if len(neighbors) != 0:
                    weights = [1 / (heuristic[n] + 0.001) for n in neighbors]
                    total_weight = sum(weights)
                    probabilities = [w / total_weight for w in weights]
                    # print(f"Probabilities: {probabilities}")
                    next_node = np.random.choice(neighbors, p=probabilities)
                    # next_node = random.choice(neighbors)
                    if next_node not in visited:
                        individual.append(next_node)
                        visited.append(next_node)
                    else:
                        # print("Remove")
                        individual.pop()
                    if self.route_distance(individual) >= self.target_distance:
                        break
                else:
                    individual.pop()

                # if next_node not in individual:
                #     individual.append(next_node)
                #     if self.route_distance(individual) >= self.target_distance:
                #         break
                #     ox.plot_graph_route(self.graph, individual)

                # Greedy Shit
                # individual.append(next_node)
                # if self.route_distance(individual) >= self.target_distance:
                #     break

                # ox.plot_graph_route(self.graph, individual)
            # ox.plot_graph_route(self.graph, individual)
            population.append(individual)
        return population

    def route_distance(self, route):
        distance = 0
        for i in range(len(route) - 1):
            distance += self.graph.edges[route[i], route[i + 1], 0]['length']
        # print(f"Distance: {distance}")
        return distance
        # distance = []
        # for u, v in zip(route[:-1], route[1:]):
        #     distance.append(self.graph[u][v][0]['length'])
        # return sum(distance)

    def route_distance_from_end_node(self, route):
        dist = osmnx.distance.euclidean(self.graph.nodes[route[-1]]['y'], self.graph.nodes[route[-1]]['x'],
                                        self.graph.nodes[self.end_node]['y'],
                                        self.graph.nodes[self.end_node]['x']) * 100000

        return dist

    def fitness(self, route):  # Evaluation
        # ox.plot_graph_route(self.graph, route)
        dist = self.route_distance(route)
        dist_end = self.route_distance_from_end_node(route)
        fitness_score = self.graph_builder.get_route_fitness(route)
        # print(f"Fitness: {abs(self.target_distance - dist)}")
        # print(f"")
        # return -abs(self.target_distance - dist)  # Fitness is better when distance is closer to target
        # print(f"Fitness: {dist_end}")
        # print(-abs(self.target_distance - dist),-abs(dist_end))
        # return -abs(dist_end)
        # Add significant bonus for including peaks
        peak_count = sum(1 for n in route if self.graph_builder.is_peak(n))
        peak_bonus = peak_count * 5000  # Adjust weight as needed

        # Penalize distance deviation more aggressively
        distance_penalty = -abs(self.target_distance - dist) * 2
        end_penalty = -abs(dist_end)

        # print((distance_penalty + end_penalty + peak_bonus) / 1000)
        return (distance_penalty + end_penalty + peak_bonus) / 1000
        # return (-abs(self.target_distance - dist) + (-abs(dist_end))) / 1000

    def selection(self, population):
        # Get raw fitness scores
        scores = np.array([self.fitness(ind) for ind in population])

        # Handle case where all scores are identical or invalid
        if np.all(scores == scores[0]) or not np.isfinite(scores).all():
            # Fallback to uniform selection
            return random.choices(population, k=2)

        # Scale scores for numerical stability
        scaled_scores = scores - np.max(scores)  # Subtract max to avoid overflow
        scaled_scores = np.clip(scaled_scores, -700, None)  # Avoid underflow

        # Compute probabilities using softmax
        exp_scores = np.exp(scaled_scores)
        probabilities = exp_scores / np.sum(exp_scores)

        # Ensure probabilities are valid
        if not np.isfinite(probabilities).all() or np.sum(probabilities) == 0:
            # Fallback to uniform selection if probabilities are invalid
            return random.choices(population, k=2)

        # Perform selection
        try:
            selected = random.choices(population, weights=probabilities, k=2)
            return selected
        except:
            # Fallback to uniform selection if random.choices fails
            return random.choices(population, k=2)

    def crossover(self, parent1, parent2):  # Brings two parents together
        cut1 = random.randint(1, len(parent1) - 2)
        cut2 = random.randint(1, len(parent2) - 2)

        child_bridge = nx.astar_path(self.graph, parent1[cut1 - 1], parent2[cut2], weight="length")
        return parent1[:cut1 + 1] + child_bridge + parent2[cut2 - 1:]

        # ox.plot_graph_route(self.graph, child)
        # return child

    def mutate(self, individual):  # Fix this
        if random.random() < self.mutation_rate:
            idx = random.randint(0, len(individual) - 2)
            neighbors = list(self.graph.neighbors(individual[idx]))
            new_node = random.choice(neighbors)
            if new_node not in individual:
                new_child1 = nx.astar_path(self.graph, individual[idx], new_node, weight="length")
                new_child2 = nx.astar_path(self.graph, new_node, individual[idx + 1], weight="length")
                individual[idx:idx + 1] = new_child1 + new_child2[1:-1]
        # print("Mutated")
        # self.fitness(individual)
        return individual

    def local_search(self, individual):
        best_fitness = self.fitness(individual)
        best_route = individual
        for i in range(len(individual) - 1):
            for neighbor in self.graph.neighbors(individual[i]):
                if neighbor not in individual:
                    print(f"New Individual: {individual}")
                    print(f"Neighbor: {neighbor}")
                    new_child1 = nx.astar_path(self.graph, individual[i], neighbor, weight="length")
                    new_child2 = nx.astar_path(self.graph, neighbor, individual[i + 1], weight="length")
                    print(f"New Child 1: {new_child1}")
                    print(f"New Child 2: {new_child2}")
                    new_route = individual[:i + 1] + [neighbor] + individual[i + 1:]
                    print(f"New Route: {new_route}")
                    new_fitness = self.fitness(new_route)
                    if new_fitness > best_fitness:
                        best_fitness = new_fitness
                        best_route = new_route
        return best_route

    def evolve(self):
        print("Generating Population")
        population = self.initialize_population()
        for generation in range(self.generations):
            print(f"Generation {generation + 1}")
            # print(f"Population Size: {len(population)}")
            new_population = []
            for _ in range(self.population_size // 2):
                # print(f"Selection")
                t = time.time()
                parent1, parent2 = self.selection(population)
                print(f"Selection Time: {time.time() - t}")
                # ox.plot_graph_route(self.graph, parent2)
                # ox.plot_graph_route(self.graph, parent1)
                # print(f"Crossover")
                child = self.crossover(parent1, parent2)
                # ox.plot_graph_route(self.graph, child)
                # print(f"Mutation")
                child = self.mutate(child)
                # ox.plot_graph_route(self.graph, child)
                # print(child)
                # print(f"Local Search")
                # child = self.local_search(child)
                # print(f"Search complete")
                # ox.plot_graph_route(self.graph, child)
                new_population.extend([child, parent1])
            population = sorted(new_population, key=self.fitness, reverse=True)[:self.population_size]
            # ox.plot_graph_route(self.graph, population[0])
        # for individual in population:
        #     ox.plot_graph_route(self.graph, individual)
        best_individual = max(population, key=self.fitness)
        print(f"Distance: {self.route_distance(best_individual)}")
        return best_individual


def generate_route():
    G = ox.graph_from_point((53.36486137451511, -1.8160056925378616), dist=8000, network_type='walk',
                            dist_type="network")
    G = ox.distance.add_edge_lengths(G)

    start_node = ox.nearest_nodes(G, X=-1.8160056925378616, Y=53.36486137451511)
    end_node = ox.nearest_nodes(G, Y=53.34344386440596, X=-1.778107050662822)

    # Initialize and run the memetic algorithm
    memetic_algo = MemeticAlgorithm(graph=G, start_node=start_node, end_node=end_node, target_distance=10000)
    route = memetic_algo.evolve()

    # Convert route to coordinates
    coords = []
    for node in route:
        coords.append((G.nodes[node]['y'], G.nodes[node]['x']))

    return coords

# # Example usage
if __name__ == "__main__":
    # generate_improved_route()
    # Load a graph of a specific area (e.g., a city)
    G = ox.graph_from_point((53.36486137451511, -1.8160056925378616), dist=8000, network_type='walk',
                            dist_type="network")
    G = ox.distance.add_edge_lengths(G)

    start_node = ox.nearest_nodes(G, X=-1.8160056925378616, Y=53.36486137451511)
    end_node = ox.nearest_nodes(G, Y=53.34344386440596, X=-1.778107050662822)

    # Initialize and run the memetic algorithm
    print("Running memetic algorithm...")
    memetic_algo = MemeticAlgorithm(start_node=start_node, end_node=end_node, target_distance=10000)
    print("Evolving...")
    best_route = memetic_algo.evolve()

    # Output the best route
    print("Best route found:", best_route)
    print("Distance of best route:", memetic_algo.route_distance(best_route))

    coords = []
    for node in best_route:
        coords.append((G.nodes[node]['y'], G.nodes[node]['x']))
    print(coords)
    # Visualize the route
    ox.plot_graph_route(G, best_route)
