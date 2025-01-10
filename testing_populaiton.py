import heapq
import time
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
G = ox.graph_from_point((53.36486137451511, -1.8160056925378616), dist=5000, network_type='walk')
print (time.time() - t)
# fig, ax = ox.plot_graph(G)

start_node = ox.distance.nearest_nodes(G, X=-1.8160056925378616, Y=53.36486137451511)
end_node = ox.distance.nearest_nodes(G, Y=53.35510304745989, X=-1.8055002497162305)

from deap import base, creator, tools, algorithms
import random

# Define the problem
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Toolbox setup
toolbox = base.Toolbox()
toolbox.register("attr_node", random.choice, list(G.nodes))
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_node, n=10)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Define fitness function
def calculate_path_cost(path):
    path_cost = 0
    for u, v in zip(path[:-1], path[1:]):
        if g.has_edge(u, v):
            # Handle multi-edges
            edge_data = g.get_edge_data(u, v)
            # Assume first edge if multiple exist
            path_cost += edge_data[0].get('length', float('inf'))
        else:
            # Penalize disconnected nodes
            path_cost += float('inf')
    return path_cost


toolbox.register("evaluate", calculate_path_cost)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Run the algorithm
population = toolbox.population(n=100)
algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, verbose=True)
