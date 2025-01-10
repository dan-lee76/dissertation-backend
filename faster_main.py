
import time
    t= time.time()
import networkx as nx
from osmnx import plot_graph
from pyrosm import OSM
print (time.time() - t)

lon = 53.36486137451511
lat = -1.8160056925378616


bbox = [lat - 0.05, lon - 0.05, lat + 0.05, lon + 0.05]
# ox.settings.use_cache = True
# ox.settings.cache_folder = "/_cache"
# print("Cache enabled:", ox.settings.use_cache)
# print("Cache folder:", ox.settings.cache_folder)
# ox.settings.cache_only_mode = True
# G = ox.graph_from_place('Edale, Derbyshire, England', network_type='walk')
t= time.time()
# Downloads all of the walkable streets
# G = ox.graph_from_point((53.36486137451511, -1.8160056925378616), dist=5000, network_type='walk')
osm = OSM("derbyshire-latest.osm.pbf", bounding_box=bbox)
print (time.time() - t)
t= time.time()
walkable_tags = {"highway": ["footway", "path", "pedestrian", "steps", "living_street", "track"]}
nodes, edges = osm.get_network(nodes=True, network_type="walking")
# data = osm.get_data_by_custom_criteria(custom_filter=walkable_tags)
# data.plot()
print (time.time() - t)






# # Plot nodes and edges on a map
ax = edges.plot(figsize=(6,6), color="gray")
ax = nodes.plot(ax=ax, color="red", markersize=2.5)

t= time.time()
G = osm.to_graph(nodes, edges, graph_type="networkx")
print (time.time() - t)
fig, ax = plot_graph(G)

# start_node = ox.distance.nearest_nodes(G, X=-1.8160056925378616, Y=53.36486137451511)
# end_node = ox.distance.nearest_nodes(G, Y=53.35510304745989, X=-1.8055002497162305)
#
# ox.plot_graph(G, node_size=1, bgcolor='k')

# astar_path = nx.astar_path(G, start_node, end_node, weight='length')

# t = time.time()
# route = ox.shortest_path(G, start_node, end_node, weight='length')
# print (time.time() - t)
#
# fig, ax = ox.plot_graph_route(G, route, route_linewidth=6, node_size=0, bgcolor='k')

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
#     for path in all_paths:
#         distance = calculate_path_distance(G,path) / 1000
#         if distance_km-tollerance <= distance <= distance_km+tollerance:
#             print(f"Route Found\nDistance: {distance}")
#             fig, ax = ox.plot_graph_route(G, path, route_linewidth=6, node_size=0, bgcolor='k')
#         # print(osmnx.stats.edge_length_total(path))

def calculate_path_distance(G, path):
    x=[]
    for u, v in zip(path[:-1], path[1:]):
       x.append(G[u][v][0]['length'])
    return sum(x)


# get_path_from_distance(G, start_node, end_node, 10, 500, 1)