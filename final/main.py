import time
import folium
from a_star import astar_path
from graph_builder import Graph_Builder
import networkx as nx
import osmnx as ox
from shapely.geometry import LineString, Point, Polygon
import geopandas as gpd
import random

class AStarArcs:
    def __init__(self, graph_builder, distance, tolerance):
        self.graph_builder = graph_builder
        self.G = graph_builder.get_graph()
        self.target_distance = distance
        self.tolerance = tolerance

        self.start_node = self.graph_builder.get_start_node()
        self.end_node = self.graph_builder.get_end_node()

        self.coords = []
        self.peaks_in_boundary = []


    def main(self):
        route = nx.astar_path(self.G, self.start_node, self.end_node, weight='length')
        route_coords = [(self.G.nodes[node]['x'], self.G.nodes[node]['y']) for node in route]
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

        for n, data in self.G.nodes(data=True):
            if random_polygon.intersects(Point(data['x'], data['y'])):
                nodes_in_boundary.append(n)
                if graph_builder.is_peak(n):
                    self.peaks_in_boundary.append(n)

        print(len(nodes_in_boundary))

        self.coords = sorted(nodes_in_boundary, key=lambda x: self.G.nodes[x]['fitness'])
        self.coords = [x for x in self.coords if self.G.nodes[x]['fitness'] < 10000]
        # print(G.nodes[sorted_nodes_fitness[0]],G.nodes[sorted_nodes_fitness[-1]])
        # print(G.nodes[sorted_nodes_fitness[0]]['fitness'], G.nodes[sorted_nodes_fitness[-1]]['fitness'])

    def merge_route(self, route1, route2):
        return route1[:-1] + route2


    def generate_new(self):
        print(len(self.coords))
        routes = []
        for node in self.coords:
            point = node
            # print(point)
            converted_node = point
            route1 = astar_path(self.G, self.start_node, converted_node, weight='fitness')
            try:
                route3 = astar_path(self.G, converted_node, self.end_node, route1, weight='fitness')
            except nx.NetworkXNoPath:
                continue
            merged_route = self.merge_route(route1, route3)
            distance = self.graph_builder.get_route_distance(merged_route) / 1000
            print(f"Distance: {distance}")
            if self.target_distance - self.tolerance < distance < self.target_distance + self.tolerance:
                print(f"Distance Achieved: {distance}")
                route = merged_route
                # fig, ax = ox.plot_graph_route(G, route1, route_linewidth=6, node_size=0, bgcolor='k')
                # fig, ax = ox.plot_graph_route(G, route3, route_linewidth=6, node_size=0, bgcolor='k')
                # fig, ax = ox.plot_graph_route(G, merged_route, route_linewidth=6, node_size=0, bgcolor='k')
                self.coords.remove(node)
                routes.append([route, graph_builder.get_route_fitness(route)])
                # break
        print(routes)
        route = min(routes, key=lambda x: x[1])[0]
        ox.plot_graph_route(self.G, route, route_linewidth=6, node_size=0, bgcolor='k')
        print(f"Assent: {graph_builder.get_route_assent(route)}")
        print(f"Descent: {graph_builder.get_route_descent(route)}")
        print(f"Distance: {self.graph_builder.get_route_distance(route) / 1000}")
        print(f"Distance: {graph_builder.get_route_distance(route)}")
        # ox.plot_route_folium(G, route, route_color='#009688').save("lakes_circle.html")
        return self.graph_builder.route_to_coords(route)

    def ridge_walker(self):
        routes = []
        print(f"Peaks: {len(self.peaks_in_boundary)}")
        self.graph_builder.view_peaks()

        # Define the custom weight function to favor descent
        def descent_favoring_weight(u, v, d):
            # print(u,v,d)
            # print(d[0]['fitness'])
            # rise = graph_builder.get_rise(u, v)
            fitness = d[0]['fitness']
            rise = d[0]['rise']
            if rise > 0:
                rise *= 100
            # print(abs(rise))
            return abs(rise * fitness)  # Ensure non-negative, small positive minimum

        # Local search function to find nearby peaks within a radius
        def get_nearby_peaks(current_node, radius=1000):  # radius in meters
            nearby = []
            # Access current node's position as (x, y)
            current_pos = (self.G.nodes[current_node]['x'], self.G.nodes[current_node]['y'])

            for peak in self.peaks_in_boundary:
                if peak != current_node:
                    # Access peak's position as (x, y)
                    peak_pos = (self.G.nodes[peak]['x'], self.G.nodes[peak]['y'])
                    dist = self.graph_builder.calculate_euclidean_distance(current_pos, peak_pos)
                    if dist <= radius:
                        nearby.append(peak)
            return nearby



        # Explore routes for each primary peak
        for node in self.peaks_in_boundary:
            # Step 1: Find direct route from start to primary peak
            route1 = astar_path(self.G, self.start_node, node, weight='fitness')  # Using custom astar_path
            local_routes = []

            # Step 2: Try direct route from primary peak to end
            try:
                route3 = astar_path(self.G, node, self.end_node, route1, weight=descent_favoring_weight)  # Using custom astar_path
                merged_route = self.merge_route(route1, route3)
                distance = self.graph_builder.get_route_distance(merged_route) / 1000
                # print(f"Direct Distance: {distance}")
                if self.target_distance - self.tolerance < distance < self.target_distance + self.tolerance:
                    print(f"Distance Achieved: {distance}")
                    print(f"Fitness: {self.graph_builder.get_route_fitness(merged_route)}")
                    local_routes.append([merged_route, self.graph_builder.get_route_fitness(merged_route),
                                         len(self.graph_builder.get_peak_nodes(merged_route))])
            except nx.NetworkXNoPath:
                pass


            def local_search(node, route1):
                # Step 3: Local search through nearby peaks
                nearby_peaks = get_nearby_peaks(node)
                for nearby_node in nearby_peaks:
                    if 245804896 in self.G[nearby_node]:
                        print("Found")
                    try:
                        # Route from primary peak to nearby peak
                        route2 = astar_path(self.G, node, nearby_node, route1, weight='fitness')  # Using custom astar_path
                        partial_route = self.merge_route(route1, route2)

                        # Route from nearby peak to end
                        try:
                            route3 = astar_path(self.G, nearby_node, self.end_node, weight=descent_favoring_weight, visited_nodes=partial_route)  # Using custom astar_path
                            merged_route = self.merge_route(partial_route, route3)
                            distance = self.graph_builder.get_route_distance(merged_route) / 1000
                            # print(f"Local Search Distance (via {nearby_node}): {distance}")
                            # ox.plot_graph_route(self.G, partial_route, route_linewidth=6, node_size=0, bgcolor='k')
                            if self.target_distance - self.tolerance < distance < self.target_distance + self.tolerance:
                                print(f"Distance Achieved: {distance}")
                                print(f"Fitness: {self.graph_builder.get_route_fitness(merged_route)}")
                                print(f"Peaks: {len(self.graph_builder.get_peak_nodes(merged_route))}")
                                local_routes.append([merged_route, self.graph_builder.get_route_fitness(merged_route),
                                                     len(self.graph_builder.get_peak_nodes(merged_route))])
                                # ox.plot_graph_route(self.G, partial_route, route_linewidth=6, node_size=0, bgcolor='k')
                                local_search(nearby_node, partial_route)
                        except nx.NetworkXNoPath:
                            continue
                    except nx.NetworkXNoPath:
                        continue
                # Step 4: Add the best local route to the main routes list
                if local_routes:
                    # best_local_route = min(local_routes, key=lambda x: x[1])  # Best based on fitness
                    # best_peak_route = max(local_routes, key=lambda x: x[2])  # Best based on peaks
                    # routes.append(best_local_route)
                    # routes.append(best_peak_route)
                    for x in local_routes:
                        routes.append(x)

            local_search(node, route1)


        # Handle case where no routes are found
        if not routes:
            print("No suitable routes found within the distance tolerance.")
            return None
        else:
            print(f"Number of suitable routes found: {len(routes)}")

        # Step 5: Select routes for visualization
        route1 = min(routes, key=lambda x: x[1])[0]  # Best fitness
        route2 = max(routes, key=lambda x: x[1])[0]  # Worst fitness
        route3 = min(routes, key=lambda x: x[2])[0]  # Fewest peaks
        route4 = max(routes, key=lambda x: x[2])[0]  # Most peaks
        print(f"Route 4 distance: {self.graph_builder.get_route_distance(route4) / 1000}, peaks: {len(self.graph_builder.get_peak_nodes(route4))}")
        ox.plot_graph_route(self.G, route4, route_linewidth=6, node_size=0, bgcolor='k')
        # Step 6: Create and populate Folium map
        m = folium.Map(location=[53.36486137451511, -1.8160056925378616], zoom_start=10)

        def add_route_to_map(route, color):
            route_coords = self.graph_builder.route_to_coords(route)
            folium.PolyLine(route_coords, color=color, weight=2.5, opacity=0.5).add_to(m)

        # Add routes to the map
        add_route_to_map(route1, 'blue')  # Best fitness
        add_route_to_map(route2, 'red')  # Worst fitness
        add_route_to_map(route3, 'green')  # Fewest peaks
        add_route_to_map(route4, 'purple')  # Most peaks
        # def random_color():
        #     return "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        #
        # for route in routes:
        #     if route[2] >= 3:
        #         ox.plot_graph_route(self.G, route[0], route_linewidth=6, node_size=0, bgcolor='k')
        #         add_route_to_map(route[0], random_color())

        # Save the map
        m.save('Edale-Ridge-Walker-All2.html')

        # Return coordinates of the best route
        return self.graph_builder.route_to_coords(route1)

def descent_favoring_weight(u, v, d):
    # print(u,v,d)
    # print(d[0]['fitness'])
    # rise = graph_builder.get_rise(u, v)
    fitness = d[0]['fitness']
    rise = d[0]['rise']
    if rise > 0:
        rise *= 100
    # print(abs(rise))
    return abs(rise*fitness)  # Ensure non-negative, small positive minimum

def merge_route(route1, route2):
    return route1[:-1] + route2

if __name__ == "__main__":
    graph_builder = Graph_Builder((53.36486137451511, -1.8160056925378616), (53.34344386440596, -1.778107050662822), simplify=True)

    # lose = ox.nearest_nodes(graph_builder.get_graph(), Y=53.3648903, X=-1.7714182)
    # back = ox.nearest_nodes(graph_builder.get_graph(), Y=53.3617310, X=-1.7826410)
    # mam = ox.nearest_nodes(graph_builder.get_graph(), Y=53.3492577, X=-1.8096423)
    # r0 = astar_path(graph_builder.get_graph(), graph_builder.get_start_node(), mam, weight='fitness')
    # r1 = astar_path(graph_builder.get_graph(), mam, back, weight='fitness', visited_nodes=r0)
    # merge = merge_route(r0,r1)
    # r2 = astar_path(graph_builder.get_graph(), back, lose, weight='fitness', visited_nodes=merge)
    # pre_mergemergemerge = merge_route(merge,r2)
    # ox.plot_graph_route(graph_builder.get_graph(), merge, route_linewidth=6, node_size=0, bgcolor='k')
    # r = astar_path(graph_builder.get_graph(), lose, graph_builder.get_end_node(), weight=descent_favoring_weight, visited_nodes=merge)
    # merge = merge_route(merge,r)
    # print(f"Distance: {graph_builder.get_route_distance(merge) / 1000}")
    # ox.plot_graph_route(graph_builder.get_graph(), merge, route_linewidth=6, node_size=0, bgcolor='k')

    astar = AStarArcs(graph_builder,10, 2)
    astar.main()
    astar.ridge_walker()
