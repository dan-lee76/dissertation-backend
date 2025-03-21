import time
import folium
import gpxpy

from a_star import astar_path, astar_path_with_backtracking
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
        if self.start_node != self.end_node:
            route = nx.astar_path(self.G, self.start_node, self.end_node, weight='length')
            route_coords = [(self.G.nodes[node]['x'], self.G.nodes[node]['y']) for node in route]
            route_line = LineString(route_coords)
            buffer_distance = self.target_distance * 0.005
            route_buffer = route_line.buffer(buffer_distance, quad_segs=1)
        else:
            route_point = Point(self.G.nodes[self.start_node]['x'], self.G.nodes[self.start_node]['y'])
            buffer_distance = self.target_distance * 0.005
            route_buffer = route_point.buffer(buffer_distance, cap_style='round')


        gdf = gpd.GeoDataFrame({'geometry': [route_buffer]}, crs="EPSG:3857")
        gdf.to_file("route_buffer.geojson", driver="GeoJSON")
        random_polygon = gdf.sample(n=1).iloc[0].geometry
        # print(random_polygon)
        boundary = random_polygon.boundary
        self.coords = list(boundary.coords)

        nodes_in_boundary = []

        for n, data in self.G.nodes(data=True):
            if random_polygon.intersects(Point(data['x'], data['y'])):
                nodes_in_boundary.append(n)
                if self.graph_builder.is_peak(n):
                    self.peaks_in_boundary.append(n)

        # print(len(nodes_in_boundary))

        self.coords = sorted(nodes_in_boundary, key=lambda x: self.G.nodes[x]['fitness'])
        self.coords = [x for x in self.coords if self.G.nodes[x]['fitness'] < 10000]
        # print(G.nodes[sorted_nodes_fitness[0]],G.nodes[sorted_nodes_fitness[-1]])
        # print(G.nodes[sorted_nodes_fitness[0]]['fitness'], G.nodes[sorted_nodes_fitness[-1]]['fitness'])

    def merge_route(self, route1, route2):
        return route1[:-1] + route2


    def generate_new(self):
        print("Starting")
        routes = []
        for node in self.coords:
            point = node
            # print(point)
            converted_node = point
            route1 = astar_path(self.G, self.start_node, converted_node, weight='fitness')[0]
            try:
                route3 = astar_path_with_backtracking(self.G, converted_node, self.end_node, route1, weight='fitness', max_backtracks=10)[0]
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
                routes.append([route, graph_builder.get_route_fitness_edges(route), len(graph_builder.get_peak_nodes(route))])
                # break
        print(routes)
        route = min(routes, key=lambda x: x[1])[0]
        ox.plot_graph_route(self.G, route, route_linewidth=6, node_size=0, bgcolor='k')
        print(f"Assent: {graph_builder.get_route_assent(route)}")
        print(f"Descent: {graph_builder.get_route_descent(route)}")
        print(f"Distance: {self.graph_builder.get_route_distance(route) / 1000}")
        print(f"Distance: {graph_builder.get_route_distance(route)}")
        m = folium.Map(location=[self.G.nodes[self.graph_builder.get_start_node()]['y'],self.G.nodes[self.graph_builder.get_start_node()]['x']], zoom_start=10, tiles="OpenTopoMap")
        m.add_child(folium.PolyLine(self.graph_builder.route_to_coords(route), color='blue', weight=2.5, opacity=0.75))
        m.save('acc.html')
        route = min(routes, key=lambda x: x[2])[0]
        ox.plot_graph_route(self.G, route, route_linewidth=6, node_size=0, bgcolor='k')
        print(f"Assent: {graph_builder.get_route_assent(route)}")
        print(f"Descent: {graph_builder.get_route_descent(route)}")
        print(f"Distance: {self.graph_builder.get_route_distance(route) / 1000}")
        print(f"Distance: {graph_builder.get_route_distance(route)}")
        m = folium.Map(location=[self.G.nodes[self.graph_builder.get_start_node()]['y'],
                                 self.G.nodes[self.graph_builder.get_start_node()]['x']], zoom_start=10,
                       tiles="OpenTopoMap")
        m.add_child(folium.PolyLine(self.graph_builder.route_to_coords(route), color='blue', weight=2.5, opacity=0.75))
        m.save("debug123.html")
        # ox.plot_route_folium(G, route, route_color='#009688').save("lakes_circle.html")
        return self.graph_builder.route_to_coords(route)

    def ridge_walker(self):
        t = time.time()
        routes = []
        print(f"Peaks: {len(self.peaks_in_boundary)}")
        # self.graph_builder.view_peaks()

        # Define the custom weight function to favor descent
        def descent_favoring_weight(u, v, d):
            fitness = d[0]['fitness']
            # rise = 400
            rise = d[0]['rise']
            # if rise >= 50:
            #     rise = 50
            if rise > 0:
                rise *= 100
            return abs(rise * fitness)  # Ensure non-negative, small positive minimum

        # Local search function to find nearby peaks within a radius
        def get_nearby_peaks(current_node, current_route, radius=0.02):  # radius in meters
            nearby = []
            # Access current node's position as (x, y)
            current_pos = (self.G.nodes[current_node]['x'], self.G.nodes[current_node]['y'])

            for peak in self.peaks_in_boundary:
                if peak != current_node and peak not in current_route:
                    # Access peak's position as (x, y)
                    peak_pos = (self.G.nodes[peak]['x'], self.G.nodes[peak]['y'])
                    dist = self.graph_builder.calculate_euclidean_distance(current_pos, peak_pos)
                    # print(dist)
                    if dist <= radius:
                        nearby.append(peak)
            return nearby


        p=0
        # Explore routes for each primary peak
        for node in self.peaks_in_boundary:
            p+=1
            try:
                print(f"{((p/len(self.peaks_in_boundary))*100):03.2f}% Complete. \tCurrent Peak: {self.G.nodes[node]["nature"]["name"]}\tID: {node}")
            except:
                print(f"{((p / len(self.peaks_in_boundary)) * 100):03.2f}% Complete. \tCurrent Peak: Unknown\tID: {node}")
            # Step 1: Find direct route from start to primary peak
            # ox.plot_graph(self.G, node_size=0)
            route1 = astar_path(self.G, self.start_node, node, weight='fitness', )[0]  # Using custom astar_path
            # ox.plot_graph_route(self.G, route1, route_linewidth=6, node_size=0, bgcolor='k')
            local_routes = []

            # Step 2: Try direct route from primary peak to end
            try:
                route3 = astar_path(self.G, node, self.end_node, route1, weight=descent_favoring_weight)[0]  # Using custom astar_path
                merged_route = self.merge_route(route1, route3)
                distance = self.graph_builder.get_route_distance(merged_route) / 1000
                print(f"Direct Distance: {distance}")
                if self.target_distance - self.tolerance < distance < self.target_distance + self.tolerance:
                    print(f"Distance Achieved: {distance}")
                    # print(f"Fitness: {self.graph_builder.get_route_fitness_edges(merged_route)}")
                    local_routes.append([merged_route, self.graph_builder.get_route_fitness_edges(merged_route),
                                         len(self.graph_builder.get_peak_nodes(merged_route))])
            except nx.NetworkXNoPath:
                pass


            def local_search(local_node, local_route):
                # Step 3: Local search through nearby peaks
                nearby_peaks = get_nearby_peaks(local_node, local_route)
                # node_colors = []
                # for node1 in self.G.nodes:
                #     if node1 == local_node:
                #         node_colors.append("red")
                #     elif node1 in nearby_peaks:
                #         node_colors.append("green")
                #     else:
                #         node_colors.append("none")
                # fig, ax = ox.plot_graph(self.G, node_color=node_colors, node_size=10)
                for nearby_node in nearby_peaks:
                    print(nearby_node)
                    if nearby_node == 31624075:
                        print("Fairfield Found")
                        route1 = astar_path_with_backtracking(self.G, self.start_node, 30050955, weight='fitness', max_backtracks=15)[0]  # Using custom astar_path
                        route2 = astar_path_with_backtracking(self.G, 30050955, 31624075, route1, weight='fitness', max_backtracks=15)[0]  # Using custom astar_path
                        mroute = self.merge_route(route1, route2)
                        route3 = astar_path_with_backtracking(self.G, 31624075, 30050713, mroute, weight='fitness', max_backtracks=15)[0]  # Using custom astar_path
                        mroute = self.merge_route(mroute, route3)
                        ox.plot_graph_route(self.G, mroute, route_linewidth=6, node_size=0, bgcolor='k')
                        # ox.plot_graph_route(self.G, local_route, route_linewidth=6, node_size=0, bgcolor='k')
                        # route2 = astar_path_with_backtracking(self.G, local_node, nearby_node, local_route,
                        #                     weight='fitness', backtrack_mode="iterative", max_backtracks=15)[0]  # Using custom astar_path
                        # partial_route = self.merge_route(local_route, route2)
                        # ox.plot_graph_route(self.G, partial_route, route_linewidth=6, node_size=0, bgcolor='k')
                        # print(f"Trying merge. Current peak amount: {len(self.graph_builder.get_peak_nodes(partial_route))}")
                        # route3 = astar_path_with_backtracking(self.G, nearby_node, self.end_node, weight='fitness', visited_nodes=partial_route, backtrack_mode="iterative", max_backtracks=15)  # Using custom astar_path
                        # print(f"Merge Complete, backtracks: {route3[1]}")
                        # merged_route = self.merge_route(partial_route, route3[0])
                        # ox.plot_graph_route(self.G, merged_route, route_linewidth=6, node_size=0, bgcolor='k')
                        # distance = self.graph_builder.get_route_distance(merged_route) / 1000
                        print(f"Back Tor Distance (via {nearby_node}): {distance}")
                    try:
                        # Route from primary peak to nearby peak
                        route2 = astar_path_with_backtracking(self.G, local_node, nearby_node, local_route, weight='fitness', max_backtracks=15)[0]  # Using custom astar_path
                        partial_route = self.merge_route(local_route, route2)

                        # Route from nearby peak to end
                        try:
                            # print("Trying")
                            # ox.plot_graph_route(self.G, partial_route, route_linewidth=6, node_size=0, bgcolor='k')
                            route3 = astar_path_with_backtracking(self.G, nearby_node, self.end_node, visited_nodes=partial_route, weight=descent_favoring_weight, max_backtracks=15)[0] # Using custom astar_path
                            # print("trying merge")
                            merged_route = self.merge_route(partial_route, route3)
                            # print("Merged")
                            distance = self.graph_builder.get_route_distance(merged_route) / 1000
                            # print(f"Local Search Distance (via {nearby_node}): {distance}")
                            # ox.plot_graph_route(self.G, partial_route, route_linewidth=6, node_size=0, bgcolor='k')
                            if self.target_distance - self.tolerance < distance < self.target_distance + self.tolerance:
                                # print(f"Distance Achieved: {distance}")
                                # print(f"Fitness: {self.graph_builder.get_route_fitness_edges(merged_route)}")
                                # print(f"Peaks: {len(self.graph_builder.get_peak_nodes(merged_route))}")
                                local_routes.append([merged_route, self.graph_builder.get_route_fitness_edges(merged_route),
                                                     len(self.graph_builder.get_peak_nodes(merged_route))])
                                # ox.plot_graph_route(self.G, merged_route, route_linewidth=6, node_size=0, bgcolor='k')
                                # local_search(nearby_node, partial_route)
                            if distance < self.target_distance + self.tolerance:
                                local_search(nearby_node, partial_route)
                        except nx.NetworkXNoPath:
                            continue
                    except nx.NetworkXNoPath:
                        continue
                # Step 4: Add the best local route to the main routes list
                if local_routes:
                    best_local_route = min(local_routes, key=lambda x: x[1])  # Best based on fitness
                    max_peaks = max(local_routes, key=lambda x: x[2])[2]  # Best based on peaks

                    routes.append(best_local_route)
                    [routes.append(x) for x in local_routes if x[2] == max_peaks]
                    # for x in local_routes:
                    #     routes.append(x)

            local_search(node, route1)


        # Handle case where no routes are found
        if not routes:
            print("No suitable routes found within the distance tolerance.")
            return None
        else:
            print(f"Number of suitable routes found: {len(routes)}")
            print(f"Time taken: {time.time() - t} seconds")

        # Step 5: Select routes for visualization
        route1 = min(routes, key=lambda x: x[1])[0]  # Best fitness
        route2 = max(routes, key=lambda x: x[1])[0]  # Worst fitness
        route3 = min(routes, key=lambda x: x[2])[0]  # Fewest peaks
        route4 = max(routes, key=lambda x: x[2])[0]  # Most peaks
        # ox.plot_graph_route(self.G,route1,route_linewidth=6, node_size=0, bgcolor='k')
        # ox.plot_graph_route(self.G, route2, route_linewidth=6, node_size=0, bgcolor='k')
        # ox.plot_graph_route(self.G, route3, route_linewidth=6, node_size=0, bgcolor='k')
        # ox.plot_graph_route(self.G, route4, route_linewidth=6, node_size=0, bgcolor='k')

        # print(f"Route 4 distance: {self.graph_builder.get_route_distance(route4) / 1000}, peaks: {len(self.graph_builder.get_peak_nodes(route4))}")
        # ox.plot_graph_route(self.G, route4, route_linewidth=6, node_size=0, bgcolor='k')
        # # Step 6: Create and populate Folium map
        m = folium.Map(location=[self.G.nodes[self.graph_builder.get_start_node()]['y'],self.G.nodes[self.graph_builder.get_start_node()]['x']], zoom_start=10, tiles="OpenTopoMap")
        #
        folium.TileLayer(
            name="Mapbox Outdoors",
            tiles="https://api.mapbox.com/styles/v1/mapbox/outdoors-v12/tiles/{z}/{x}/{y}?access_token=pk.eyJ1IjoiZGFuLWxlZTc2IiwiYSI6ImNsaXJ4d3N3azE3M3Mza28xdnVmdGxwczcifQ.j0THIt59Yt7D6qG1WKXTPg",
            attr="© Mapbox",
            overlay=False,
            control=True,
            show=False,
        ).add_to(m)
        folium.LayerControl().add_to(m)
        def add_route_to_map(route, color):
            route_coords = self.graph_builder.route_to_coords(route)
            folium.PolyLine(route_coords, color=color, weight=5, opacity=1).add_to(m)
        #
        # # # Add routes to the map
        # add_route_to_map(route1, 'blue')  # Best fitness
        # add_route_to_map(route2, 'red')  # Worst fitness
        # add_route_to_map(route3, 'green')  # Fewest peaks
        # add_route_to_map(route4, 'purple')  # Most peaks
        def random_color():
            return "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        #
        # for route in routes:
        #     ox.plot_graph_route(self.G, route[0], route_linewidth=6, node_size=0, bgcolor='k')
        #     add_route_to_map(route[0], random_color())
        print(f"Number of routes: {len(routes)}")
        route_no_duplicates = []
        for route in routes:
            if route[0] not in route_no_duplicates:
                route_no_duplicates.append(route[0])
                # ox.plot_graph_route(self.G, route[0], route_linewidth=6, node_size=0, bgcolor='k')
        print(f"Number of routes (no duplicates): {len(route_no_duplicates)}")
        peak_4 = [x for x in routes if x[2] >= 4]
        print(f"Number of 4 peak routes: {len(peak_4)}")
        peak_4_no_duplicates = []
        for route in peak_4:
            if route[0] not in peak_4_no_duplicates:
                ox.plot_graph_route(self.G, route[0], route_linewidth=6, node_size=0, bgcolor='k')
                peak_4_no_duplicates.append(route[0])
                # add_route_to_map(route[0], random_color())
        print(f"Number of 4 peak routes (no duplicates): {len(peak_4_no_duplicates)}")
        best_peak_fitness = min(peak_4, key=lambda x: x[1])[0]
        # ox.plot_graph_route(self.G, best_peak_fitness, route_linewidth=6, node_size=0, bgcolor='k')
        print(f"Best Fitness details \nFitness: {self.graph_builder.get_route_fitness_edges(best_peak_fitness)}\nPeaks: {len(self.graph_builder.get_peak_nodes(best_peak_fitness))}\nDistance: {self.graph_builder.get_route_distance(best_peak_fitness) / 1000}\nElevation Gain: {self.graph_builder.get_route_assent(best_peak_fitness)}\nElevation Loss: {self.graph_builder.get_route_descent(best_peak_fitness)}")
        # ox.plot_graph_route(self.G, best_peak_fitness, route_linewidth=6, node_size=0, bgcolor='k')
        add_route_to_map(best_peak_fitness, 'red')
        # # Save the map
        m.save('Debug.html')

        # Return coordinates of the best route
        return self.graph_builder.route_to_coords(best_peak_fitness)

# def descent_favoring_weight(u, v, d):
#     # print(u,v,d)
#     # print(d[0]['fitness'])
#     # rise = graph_builder.get_rise(u, v)
#     fitness = d[0]['fitness']
#     rise = d[0]['rise']
#     if rise > 0:
#         rise *= 100
#     # print(abs(rise))
#     return abs(rise*fitness)  # Ensure non-negative, small positive minimum

# def merge_route(route1, route2):
#     return route1[:-1] + route2

def highway_score(highway):
    if isinstance(highway, list):
        total = 0
        for h in highway:
            total += highway_score(h)
        return total / len(highway)
    else:
        if highway in ["footway", "path", "steps", "pedestrian", "track", "bridleway", "cycleway", "service"]:
            return 0.25
        elif highway in ["residential", "living_street", "unclassified", "road"]:
            return 1
        elif highway in ["tertiary", "tertiary_link", "secondary", "secondary_link", "primary", "primary_link"]:
            return 1.5
        elif highway in ["trunk", "trunk_link", "motorway", "motorway_link"]:
            return 2
        else:
            return 3

def accessible_highway_score(highway):
    if isinstance(highway, list):
        total = 0
        for h in highway:
            total += accessible_highway_score(h)
        return total / len(highway)
    if highway in ["pedestrian", "track", "bridleway", "cycleway", "service"]:
         return 0.25
    elif highway in ["residential", "living_street", "unclassified", "road"]:
        return 1
    elif highway in ["tertiary", "tertiary_link", "secondary", "secondary_link", "primary", "primary_link"]:
        return 1
    elif highway in ["trunk", "trunk_link", "motorway", "motorway_link", "steps", "footway", "path"]:
        return 3
    else:
        return 5

def accessible_surface_score(surface):
    if isinstance(surface, list):
        total = 0
        for h in surface:
            total += accessible_surface_score(h)
        return total / len(surface)
    if surface in ["paved", "asphalt", "chipseal", "concrete", "concrete:plates", "paving_stones", "bricks", "wood", "rubber", "tiles", "fibre_reinforced_polymer_grate"]:
        return 0.25
    elif surface in ["concrete:lanes", "paving_stones:lanes", "grass_paver", "sett", "metal", "metal_grid"]:
        return 1
    elif surface in ["unhewn_cobblestone", "cobblestone", "compacted"]:
        return 2
    elif surface in ["stepping_stones", "fine_gravel", "gravel", "shells", "pebblestone", "ground", "dirt", "earth", "grass", "mud", "sand", "woodchips", "grit", "salt", "wood"]:
        return 3
    else:
        return 4



if __name__ == "__main__":
    # Edale to Castleton
    # graph_builder = Graph_Builder((53.36486137451511, -1.8160056925378616), (53.34344386440596, -1.778107050662822), simplify=False)
    # graph_builder = Graph_Builder((53.34344386440596, -1.778107050662822), (53.34344386440596, -1.778107050662822),simplify=False)
    # graph_builder = Graph_Builder((53.36486137451511, -1.8160056925378616), (53.34344386440596, -1.778107050662822),simplify=True, highway_score_function=accessible_highway_score, surface_score_function=accessible_surface_score)
    # Dragons Back Circle
    # graph_builder = Graph_Builder((53.181253754043624, -1.868378920162422), (53.181253754043624, -1.868378920162422),simplify=False)
    # Rydal
    # graph_builder = Graph_Builder((54.44851570433196, -2.9805133121389544), (54.44851570433196, -2.9805133121389544),simplify=True, highway_score_function=accessible_highway_score, surface_score_function=accessible_surface_score)
    graph_builder = Graph_Builder((54.44851570433196, -2.9805133121389544), (54.44851570433196, -2.9805133121389544),simplify=False)




    astar = AStarArcs(graph_builder,16, 3)
    astar.main()
    route_coords = astar.ridge_walker()

    gpx = gpxpy.gpx.GPX()
    gpx_track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(gpx_track)
    gpx_segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment)

    for coord in route_coords:
        gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(coord[0], coord[1]))

    with open("route.gpx", "w") as f:
        f.write(gpx.to_xml())
    # astar.generate_new()
