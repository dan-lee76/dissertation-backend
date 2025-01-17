import osmnx as ox
import networkx as nx
import random
import numpy as np


# Define the memetic algorithm class
class MemeticAlgorithm:
    def __init__(self, graph, start_node, target_distance, population_size=100, generations=100, mutation_rate=0.8):
        self.graph = graph
        self.start_node = start_node
        self.target_distance = target_distance
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = [self.start_node]
            visited = [self.start_node]
            while True:
                neighbors = list(self.graph.neighbors(individual[-1]))
                # print(f"Neighbors: {neighbors}")
                neighbors = [neighbor for neighbor in neighbors if neighbor not in visited]
                if len(neighbors) != 0:
                    next_node = random.choice(neighbors)
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
            ox.plot_graph_route(self.graph, individual)
            population.append(individual)
        return population

    def route_distance(self, route):
        distance = 0
        for i in range(len(route) - 1):
            distance += self.graph.edges[route[i], route[i + 1], 0]['length']
        # print(f"Distance: {distance}")
        return distance

    def fitness(self, route): # Evaluation
        # ox.plot_graph_route(self.graph, route)
        dist = self.route_distance(route)
        # print(f"Fitness: {abs(self.target_distance - dist)}")
        # print(f"")
        return -abs(self.target_distance - dist)  # Fitness is better when distance is closer to target

    def selection(self, population):
        scores = [self.fitness(ind) for ind in population]
        probabilities = np.exp(scores) / sum(np.exp(scores))
        selected = random.choices(population, probabilities, k=2)
        return selected

    def crossover(self, parent1, parent2):
        cut1 = random.randint(1, len(parent1) - 2)
        cut2 = random.randint(1, len(parent2) - 2)
        # ox.plot_graph_route(self.graph, parent1)
        # ox.plot_graph_route(self.graph, parent2)
        # child = parent1[:cut1] + [node for node in parent2[cut2:] if node not in parent1[:cut1]]
        child_bridge = nx.astar_path(self.graph, parent1[cut1-1], parent2[cut2], weight="length")
        child = parent1[:cut1+1] + child_bridge + parent2[cut2-1:]
        # ox.plot_graph_route(self.graph, child)
        return child

    def mutate(self, individual): #Fix this
        if random.random() < self.mutation_rate:
            idx = random.randint(0, len(individual) - 2)
            neighbors = list(self.graph.neighbors(individual[idx]))
            new_node = random.choice(neighbors)
            if new_node not in individual:
                new_child1 = nx.astar_path(self.graph, individual[idx], new_node, weight="length")
                new_child2 = nx.astar_path(self.graph, new_node, individual[idx + 1], weight="length")
                individual[idx:idx + 1] = new_child1 + new_child2[1:-1]
        # print("Mutated")
        self.fitness(individual)
        return individual

    def local_search(self, individual):
        best_fitness = self.fitness(individual)
        # print(f"Fitness complete")
        best_route = individual
        for i in range(len(individual) - 1):
            for neighbor in self.graph.neighbors(individual[i]):
                if neighbor not in individual:
                    new_route = individual[:i + 1] + [neighbor] + individual[i + 1:]
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
            new_population = []
            for _ in range(self.population_size // 2):
                # print(f"Selection")
                parent1, parent2 = self.selection(population)
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
                new_population.extend([child, parent1])
            population = sorted(new_population, key=self.fitness, reverse=True)[:self.population_size]
        for individual in population:
            ox.plot_graph_route(self.graph, individual)
        best_individual = max(population, key=self.fitness)
        return best_individual

# Example usage
if __name__ == "__main__":
    # Load a graph of a specific area (e.g., a city)
    G = ox.graph_from_point((53.36486137451511, -1.8160056925378616), dist=8000, network_type='walk', dist_type="network")
    G = ox.distance.add_edge_lengths(G)


    start_node = ox.nearest_nodes(G, X=-1.8160056925378616, Y=53.36486137451511)

    print(f"Start node information: {G[start_node]}")


    # Initialize and run the memetic algorithm
    print("Running memetic algorithm...")
    memetic_algo = MemeticAlgorithm(graph=G, start_node=start_node, target_distance=10000)
    print("Evolving...")
    best_route = memetic_algo.evolve()

    # Output the best route
    print("Best route found:", best_route)
    print("Distance of best route:", memetic_algo.route_distance(best_route))

    # Visualize the route
    ox.plot_graph_route(G, best_route)
