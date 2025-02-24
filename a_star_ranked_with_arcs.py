import random
import time

import folium

from graph_preperation import Graph_Builder

t= time.time()
import networkx as nx
import osmnx as ox
from shapely.geometry import LineString, Point, Polygon
import geopandas as gpd
from heapq import heappop, heappush
from itertools import count
from networkx.algorithms.shortest_paths.weighted import _weight_function
print(time.time() - t)
t= time.time()

graph_builder = Graph_Builder((53.36486137451511, -1.8160056925378616), (53.34344386440596, -1.778107050662822))
G = graph_builder.get_graph()
print (time.time() - t)


def merge_route(route1, route2):
    return route1[:-1] + route2


def calculate_path_distance(G, path):
    x=[]
    for u, v in zip(path[:-1], path[1:]):
       x.append(G[u][v][0]['length'])
    return sum(x)

def astar_path(G, source, target, route1, heuristic=None, weight="weight", *, cutoff=None):
    if source not in G or target not in G:
        msg = f"Either source {source} or target {target} is not in G"
        raise nx.NodeNotFound(msg)

    if heuristic is None:
        # The default heuristic is h=0 - same as Dijkstra's algorithm
        def heuristic(u, v):
            return 0

    push = heappush
    pop = heappop
    weight = _weight_function(G, weight)

    G_succ = G._adj  # For speed-up (and works for both directed and undirected graphs)


    c = count()
    queue = [(0, next(c), source, 0, None)]


    enqueued = {}
    # Maps explored nodes to parent closest to the source.
    explored = {}

    while queue:
        # Pop the smallest item from queue.
        _, __, curnode, dist, parent = pop(queue)

        if curnode == target:
            path = [curnode]
            node = parent
            while node is not None:
                path.append(node)
                node = explored[node]
            path.reverse()
            return path

        if curnode in explored:
            if explored[curnode] is None:
                continue

            qcost, h = enqueued[curnode]
            if qcost < dist:
                continue

        explored[curnode] = parent

        for neighbor, w in G_succ[curnode].items():
            cost = weight(curnode, neighbor, w)
            if cost is None:
                continue
            if neighbor in route1:
                # print("Backtracking Detected")
                continue
            ncost = dist + cost
            if neighbor in enqueued:
                qcost, h = enqueued[neighbor]

                if qcost <= ncost:
                    continue
            else:
                h = heuristic(neighbor, target)

            if cutoff and ncost + h > cutoff:
                continue

            enqueued[neighbor] = ncost, h
            push(queue, (ncost + h, next(c), neighbor, ncost, curnode))

    raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}")

def route_to_coords(route):
    return [(G.nodes[node]['y'], G.nodes[node]['x']) for node in route]


class AStarArcs:
    def __init__(self, distance, tolerance):
        self.target_distance = distance
        self.tolerance = tolerance
        # 54.45163585530434, -3.281801662280587
        # self.start_node = ox.nearest_nodes(G, X=-3.281801662280587, Y=54.45163585530434)
        # # 54.47022480476181, -3.2486108035416748
        # self.end_node = ox.nearest_nodes(G, Y=54.45326631103197, X=-3.2768023866755756)

        self.start_node = ox.nearest_nodes(G, X=-1.8160056925378616, Y=53.36486137451511)
        self.end_node = ox.nearest_nodes(G, Y=53.34344386440596, X=-1.778107050662822)

        self.coords = []
        self.peaks_in_boundary = []


    def main(self):
        route = nx.astar_path(G, self.start_node, self.end_node, weight='length')
        route_coords = [(G.nodes[node]['x'], G.nodes[node]['y']) for node in route]
        route_line = LineString(route_coords)
        buffer_distance = self.target_distance * 0.005
        route_buffer = route_line.buffer(buffer_distance, quad_segs=1)
        gdf = gpd.GeoDataFrame({'geometry': [route_buffer]}, crs="EPSG:3857")
        gdf.to_file("route_buffer.geojson", driver="GeoJSON")
        random_polygon = gdf.sample(n=1).iloc[0].geometry
        print(random_polygon)
        boundary = random_polygon.boundary
        self.coords = list(boundary.coords)

        nodes_in_boundary = []


        for n, data in G.nodes(data=True):
            if random_polygon.intersects(Point(data['x'], data['y'])):
                nodes_in_boundary.append(n)
                if graph_builder.is_peak(n):
                    self.peaks_in_boundary.append(n)

        print(len(nodes_in_boundary))

        self.coords = sorted(nodes_in_boundary, key=lambda x: G.nodes[x]['fitness'])
        self.coords = [x for x in self.coords if G.nodes[x]['fitness'] < 10000]
        # print(G.nodes[sorted_nodes_fitness[0]],G.nodes[sorted_nodes_fitness[-1]])
        # print(G.nodes[sorted_nodes_fitness[0]]['fitness'], G.nodes[sorted_nodes_fitness[-1]]['fitness'])

    def generate_new(self):
        print(len(self.coords))
        routes = []
        for node in self.coords:
            point = node
            # print(point)
            converted_node = point
            route1 = nx.astar_path(G, self.start_node, converted_node, weight='fitness')
            try:
                route3 = astar_path(G, converted_node, self.end_node, route1, weight='fitness')
            except nx.NetworkXNoPath:
                continue
            merged_route = merge_route(route1, route3)
            distance = calculate_path_distance(G, merged_route) / 1000
            print(f"Distance: {distance}")
            if self.target_distance - self.tolerance < distance < self.target_distance + self.tolerance:
                print(f"Distance Achieved: {distance}")
                print(f"Time: {time.time() - t}")
                route = merged_route
                # fig, ax = ox.plot_graph_route(G, route1, route_linewidth=6, node_size=0, bgcolor='k')
                # fig, ax = ox.plot_graph_route(G, route3, route_linewidth=6, node_size=0, bgcolor='k')
                # fig, ax = ox.plot_graph_route(G, merged_route, route_linewidth=6, node_size=0, bgcolor='k')
                self.coords.remove(node)
                routes.append([route, graph_builder.get_route_fitness(route)])
                # break
        print(routes)
        route = min(routes, key=lambda x: x[1])[0]
        ox.plot_graph_route(G, route, route_linewidth=6, node_size=0, bgcolor='k')
        print(f"Assent: {graph_builder.get_route_assent(route)}")
        print(f"Descent: {graph_builder.get_route_descent(route)}")
        print(f"Distance: {calculate_path_distance(G, route) / 1000}")
        print(f"Distance: {graph_builder.get_route_distance(route)}")
        # ox.plot_route_folium(G, route, route_color='#009688').save("lakes_circle.html")
        return route_to_coords(route)

    def ridge_walker(self):
        routes = []
        print(f"Peaks: {len(self.peaks_in_boundary)}")
        for node in self.peaks_in_boundary:
            route1 = nx.astar_path(G, self.start_node, node, weight='fitness')
            try:
                route3 = astar_path(G, node, self.end_node, route1, weight='fitness')
            except nx.NetworkXNoPath:
                continue
            merged_route = merge_route(route1, route3)
            distance = calculate_path_distance(G, merged_route) / 1000
            print(f"Distance: {distance}")
            # fig, ax = ox.plot_graph_route(G, merged_route, route_linewidth=6, node_size=0, bgcolor='k')
            # self.peaks_in_boundary.remove(node)
            if self.target_distance - self.tolerance < distance < self.target_distance + self.tolerance:
                print(f"Distance Achieved: {distance}")
                print(f"Fitness: {graph_builder.get_route_fitness(merged_route)}")
                print(f"Time: {time.time() - t}")
                route = merged_route
                 # fig, ax = ox.plot_graph_route(G, route1, route_linewidth=6, node_size=0, bgcolor='k')
                 # fig, ax = ox.plot_graph_route(G, route3, route_linewidth=6, node_size=0, bgcolor='k')
                # fig, ax = ox.plot_graph_route(G, merged_route, route_linewidth=6, node_size=0, bgcolor='k')
                routes.append([route, graph_builder.get_route_fitness(route), len(graph_builder.get_peak_nodes(route))])
                 # break
            else:
                for node2 in self.peaks_in_boundary:
                    try:
                        route2 = astar_path(G, node, node2, route1, weight='fitness')
                        merged_route = merge_route(route1, route2)
                    except nx.NetworkXNoPath:
                        continue
                    try:
                        route3 = astar_path(G, node2, self.end_node, merged_route, weight='fitness')
                        merged_route = merge_route(merged_route, route3)
                    except nx.NetworkXNoPath:
                        continue
                    distance = calculate_path_distance(G, merged_route) / 1000
                    print(f"Distance: {distance}")
                    # fig, ax = ox.plot_graph_route(G, merged_route, route_linewidth=6, node_size=0, bgcolor='k')
                    if self.target_distance - self.tolerance < distance < self.target_distance + self.tolerance:
                        print(f"Distance Achieved: {distance}")
                        print(f"Fitness: {graph_builder.get_route_fitness(merged_route)}")
                        print(f"Time: {time.time() - t}")
                        route = merged_route
                        # fig, ax = ox.plot_graph_route(G, route1, route_linewidth=6, node_size=0, bgcolor='k')
                        # fig, ax = ox.plot_graph_route(G, route3, route_linewidth=6, node_size=0, bgcolor='k')
                        # fig, ax = ox.plot_graph_route(G, merged_route, route_linewidth=6, node_size=0, bgcolor='k')
                        routes.append([route, graph_builder.get_route_fitness(route), len(graph_builder.get_peak_nodes(route))])
                        # break

        # print(routes)
        # for x in routes:
        #     ox.plot_graph_route(G, x[0], route_linewidth=6, node_size=0, bgcolor='k')
        route1 = min(routes, key=lambda x: x[1])[0]
        # ox.plot_graph_route(G, route1, route_linewidth=6, node_size=0, bgcolor='k')
        route2 = max(routes, key=lambda x: x[1])[0]
        # ox.plot_graph_route(G, route, route_linewidth=6, node_size=0, bgcolor='k')
        route3 = min(routes, key=lambda x: x[2])[0]
        # ox.plot_graph_route(G, route, route_linewidth=6, node_size=0, bgcolor='k')
        route4 = max(routes, key=lambda x: x[2])[0]
        # ox.plot_graph_route(G, route, route_linewidth=6, node_size=0, bgcolor='k')

        m = folium.Map(location=[53.36486137451511, -1.8160056925378616], zoom_start=10)  # Center the map on a location

        # Function to add a route to the map
        def add_route_to_map(route, color):
            folium.PolyLine(route, color=color, weight=2.5, opacity=1).add_to(m)

        # Extract the coordinates for each route
        route1_coords = route_to_coords(route1)
        route2_coords = route_to_coords(route2)
        route3_coords = route_to_coords(route3)
        route4_coords = route_to_coords(route4)

        # Add each route to the map with a different color
        add_route_to_map(route1_coords, 'blue') #purple :) #Best heuristic
        add_route_to_map(route2_coords, 'red') #Worst Heuristic
        add_route_to_map(route3_coords, 'green') #Worst Peaks
        add_route_to_map(route4_coords, 'purple') #Best Peaks

        # Save the map to an HTML file
        m.save('Edale-Ridge-Walker.html')

        return route_to_coords(route)


if __name__ == "__main__":
    astar = AStarArcs(10, 1)
    astar.main()
    astar.generate_new()