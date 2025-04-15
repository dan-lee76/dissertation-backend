import time
import folium
import gpxpy
import argparse
from a_star import astar_path, astar_path_with_backtracking
from graph_builder import Graph_Builder
import networkx as nx
import osmnx as ox
from shapely.geometry import LineString, Point, Polygon
import geopandas as gpd
import random

class AStarArcs:
    def __init__(self, graph_builder, distance, tolerance, favor_descent=True, normal_nodes='all', max_backtracks=5):
        self.graph_builder = graph_builder
        self.G = graph_builder.get_graph()
        self.target_distance = distance
        self.tolerance = tolerance
        self.favor_descent = favor_descent  # New parameter for descent favoring
        self.normal_nodes = normal_nodes    # New parameter for node selection

        self.start_node = self.graph_builder.get_start_node()
        self.end_node = self.graph_builder.get_end_node()
        self.max_backtracks = max_backtracks

        self.output_coords = []
        self.coords = []
        self.peaks_in_boundary = []

        self.min_factor = min(
            self.G.nodes[v]["fitness"] * data["highway_score"]
            for u, v, data in self.G.edges(data=True)
            if "fitness" in self.G.nodes[v] and "highway_score" in data
        )

        self.sp_distance = nx.single_source_dijkstra_path_length(
            self.G, self.end_node, weight='length'
        )

        def heuristic(u, v):
            return self.min_factor * self.sp_distance.get(u, float('inf'))
            # return 0 # Turns search into UCS which is quicker

        self.heuristic = heuristic
        self.edge_lengths = self.graph_builder.get_edge_lengths()

    def boundary_generation(self):
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
        random_polygon = gdf.sample(n=1).iloc[0].geometry
        boundary = random_polygon.boundary
        self.coords = list(boundary.coords)

        nodes_in_boundary = []
        for n, data in self.G.nodes(data=True):
            if random_polygon.intersects(Point(data['x'], data['y'])):
                nodes_in_boundary.append(n)
                if self.graph_builder.is_peak(n):
                    self.peaks_in_boundary.append(n)

        self.coords = sorted(nodes_in_boundary, key=lambda x: self.G.nodes[x]['fitness'])

    def merge_route(self, route1, route2):
        return route1[:-1] + route2

    def normal_generation(self):
        routes = []
        if self.normal_nodes == 'all':
            nodes_to_use = self.coords
        else:  # 'top10'
            num_nodes = max(1, int(len(self.coords) * 0.1))
            nodes_to_use = self.coords[:num_nodes]  
        print("Running normal generation...")
        for node in nodes_to_use:
            converted_node = node
            try:
                route1 = astar_path(self.G, self.start_node, converted_node, weight='fitness')[0]
                route2 = astar_path_with_backtracking(self.G, converted_node, self.end_node, route1, weight='fitness',
                                                      max_backtracks=self.max_backtracks)[0]
                merged_route = self.merge_route(route1, route2)
                distance = self.graph_builder.get_route_distance(merged_route) / 1000
                if self.target_distance - self.tolerance < distance < self.target_distance + self.tolerance:
                    routes.append({'route': merged_route,
                                   'fitness': self.graph_builder.get_route_fitness_edges(merged_route),
                                   'peaks': len(self.graph_builder.get_peak_nodes(merged_route)),
                                   'distance': distance})
            except nx.NetworkXNoPath:
                continue
        print("Generation complete.")
        return routes

    def peak_bagger_generation(self):
        t = time.time()
        routes = []
        print(f"Peaks: {len(self.peaks_in_boundary)}")

        def descent_favoring_weight(u, v, d):
            fitness = d[0]['fitness']
            rise = d[0]['rise']
            if rise > 0:
                rise *= 100
            return abs(rise * fitness)

        # Select weight function based on favor_descent
        return_weight = descent_favoring_weight if self.favor_descent else 'fitness'

        def get_nearby_peaks(current_node, current_route, radius=0.02):
            nearby = []
            current_pos = (self.G.nodes[current_node]['x'], self.G.nodes[current_node]['y'])
            for peak in self.peaks_in_boundary:
                if peak != current_node and peak not in current_route:
                    peak_pos = (self.G.nodes[peak]['x'], self.G.nodes[peak]['y'])
                    dist = self.graph_builder.calculate_euclidean_distance(current_pos, peak_pos)
                    if dist <= radius:
                        nearby.append(peak)
            return nearby

        p = 0
        for node in self.peaks_in_boundary:
            p += 1
            try:
                print(f"{((p / len(self.peaks_in_boundary)) * 100):03.2f}% Complete. \tCurrent Peak: {self.G.nodes[node]['nature']['name']}\tID: {node}")
            except:
                print(f"{((p / len(self.peaks_in_boundary)) * 100):03.2f}% Complete. \tCurrent Peak: Unknown\tID: {node}")
            route1 = astar_path(self.G, self.start_node, node, weight='fitness', heuristic=self.heuristic)[0]
            local_routes = []
            try:
                route3 = astar_path(self.G, node, self.end_node, route1, weight=return_weight, heuristic=self.heuristic)[0]
                merged_route = self.merge_route(route1, route3)
                distance = self.graph_builder.get_route_distance(merged_route) / 1000
                if self.target_distance - self.tolerance < distance < self.target_distance + self.tolerance:
                    local_routes.append({'route': merged_route,
                                         'fitness': self.graph_builder.get_route_fitness_edges(merged_route),
                                         'peaks': len(self.graph_builder.get_peak_nodes(merged_route)),
                                         'distance': distance})
            except nx.NetworkXNoPath:
                pass

            def local_search(local_node, local_route, running_dist=0):
                nearby_peaks = get_nearby_peaks(local_node, local_route)
                for nearby_node in nearby_peaks:
                    try:
                        route2 = astar_path_with_backtracking(self.G, local_node, nearby_node, local_route, weight='fitness',
                                                              max_backtracks=self.max_backtracks,
                                                              target_distance=(self.target_distance - running_dist),
                                                              heuristic=self.heuristic, lengths=self.edge_lengths)[0]
                        partial_route = self.merge_route(local_route, route2)
                        try:
                            route3 = astar_path_with_backtracking(self.G, nearby_node, self.end_node, visited_nodes=partial_route,
                                                                  weight=return_weight, max_backtracks=self.max_backtracks,
                                                                  heuristic=self.heuristic, lengths=self.edge_lengths)[0]
                            merged_route = self.merge_route(partial_route, route3)
                            distance = self.graph_builder.get_route_distance(merged_route) / 1000
                            if self.target_distance - self.tolerance < distance < self.target_distance + self.tolerance:
                                local_routes.append({'route': merged_route,
                                                     'fitness': self.graph_builder.get_route_fitness_edges(merged_route),
                                                     'peaks': len(self.graph_builder.get_peak_nodes(merged_route)),
                                                     'distance': distance})
                            running_dist = self.graph_builder.get_route_distance(partial_route) / 1000
                            if running_dist < self.target_distance + self.tolerance: # using distance instead of running_distance speeds performance, however, causes missed routes
                                local_search(nearby_node, partial_route, running_dist)
                        except nx.NetworkXNoPath:
                            continue
                    except nx.NetworkXNoPath:
                        continue
                if local_routes:
                    [routes.append(x) for x in local_routes if x not in routes]

            local_search(node, route1)

        if not routes:
            print("No suitable routes found within the distance tolerance.")
            return None
        else:
            print(f"Number of suitable routes found: {len(routes)}")
            # print(f"Time to generate: {time.time() - t:.2f} seconds")
        return routes

    def export_to_gpx(self):
        gpx = gpxpy.gpx.GPX()
        gpx_track = gpxpy.gpx.GPXTrack()
        gpx.tracks.append(gpx_track)
        gpx_segment = gpxpy.gpx.GPXTrackSegment()
        gpx_track.segments.append(gpx_segment)

        for coord in self.output_coords:
            gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(coord[0], coord[1]))

        with open("route.gpx", "w") as f:
            f.write(gpx.to_xml())

    def export_to_html(self):
        m = folium.Map(location=[self.output_coords[0][0], self.output_coords[0][1]], zoom_start=13)
        folium.PolyLine(self.output_coords, color='blue', weight=5, opacity=0.7).add_to(m)
        m.save("route.html")

    def get_route(self, route_list, best_fitness=True, most_peaks=True, least_peaks=False, closest_to_target=False):
        if most_peaks:
            max_peaks = max(route_list, key=lambda x: x["peaks"])["peaks"]
            route_list = [x for x in route_list if x["peaks"] == max_peaks]
        if least_peaks:
            min_peaks = min(route_list, key=lambda x: x["peaks"])["peaks"]
            route_list = [x for x in route_list if x["peaks"] == min_peaks]
        if closest_to_target:
            min_distance = min(route_list, key=lambda x: abs(x["distance"] - self.target_distance))["distance"]
            route_list = [x for x in route_list if abs(x["distance"] - self.target_distance) == abs(min_distance - self.target_distance)]
        route_list = sorted(route_list, key=lambda x: x["fitness"])
        if best_fitness:
            print(f"Route Details\nFitness: {route_list[0]['fitness']}\nDistance: {route_list[0]['distance']}\nPeaks: {route_list[0]['peaks']}")
            return route_list[0]["route"]
        print(f"Route Details\nFitness: {route_list[-1]['fitness']}\nDistance: {route_list[-1]['distance']}\nPeaks: {route_list[-1]['peaks']}")
        return route_list[-1]["route"]

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
    if surface in ["paved", "asphalt", "chipseal", "concrete", "concrete:plates", "paving_stones", "bricks", "wood",
                   "rubber", "tiles", "fibre_reinforced_polymer_grate"]:
        return 0.25
    elif surface in ["concrete:lanes", "paving_stones:lanes", "grass_paver", "sett", "metal", "metal_grid"]:
        return 1
    elif surface in ["unhewn_cobblestone", "cobblestone", "compacted"]:
        return 2
    elif surface in ["stepping_stones", "fine_gravel", "gravel", "shells", "pebblestone", "ground", "dirt", "earth",
                     "grass", "mud", "sand", "woodchips", "grit", "salt", "wood"]:
        return 3
    else:
        return 4

def fitness(graph, n1, n2, data):
    return data["length"] * graph.nodes[n2]["fitness"] * data["highway_score"]

def accessible_fitness(graph, n1, n2, data):
    return data["length"] * graph.nodes[n2]["fitness"] * data["highway_score"] * data["impedance"] * data["surface_score"] * data["access_score"]

def main():
    def parse_coordinates(coord_str):
        try:
            lat, lon = map(float, coord_str.split(','))
            return (lat, lon)
        except ValueError:
            raise argparse.ArgumentTypeError("Coordinates must be in format 'lat,lon'")

    parser = argparse.ArgumentParser(description="Generative Pathfinding for UK Hiking Routes - Dan Lee 2025")
    parser.add_argument('--start', type=parse_coordinates, required=True, help="Start coordinates as 'lat,lon'")
    parser.add_argument('--end', type=parse_coordinates, required=False, help="End coordinates as 'lat,lon'")
    parser.add_argument('--distance', type=float, required=True, help="Target distance in kilometers")
    parser.add_argument('--simplify', action='store_true', default=False, help="Simplify the graph")
    parser.add_argument('--tolerance', type=float, default=2.0, help="Distance tolerance in kilometers (default: 2.0)")
    parser.add_argument('--mode', choices=['normal', 'peak-bagger'], default='peak-bagger',
                        help="Generation mode: normal or peak-bagger (default: peak-bagger)")
    parser.add_argument('--all-peaks', action='store_true', default=False,
                        help="Include all peaks in the route")
    parser.add_argument('--worst-fitness', action='store_true',
                        help="Select route with worst fitness")
    parser.add_argument('--peak-fitness', choices=['most', 'least', 'ignore'], default='most',
                        help="Change how the program will select the route based on peak fitness. Options: 'most', 'least', 'ignore' (default: 'most')")
    parser.add_argument('--closest-distance', action='store_true', default=False,
                        help="Select route closest to target distance")
    parser.add_argument('--favor-descent', action='store_true', default=True,
                        help="Favor descent in peak-bagger mode")
    parser.add_argument('--normal-nodes', choices=['all', 'top10'], default='all',
                        help="Nodes to use in normal generation: 'all' or 'top10' (default: 'all')")
    parser.add_argument('--fitness-mode', choices=['normal','accessible', 'custom'], default='normal',
                        help="Customise the fitness used in path generation. Use config.py to modify custom fitness. (default: normal)")
    parser.add_argument('--max-backtracks', type=int, default=5,
                        help="Maximum number of backtracks for pathfinding (default: 5)")
    parser.add_argument('--output-gpx', action='store_true', default=False,
                        help="Export the selected route to a GPX file")
    parser.add_argument('--output-html', action='store_true', default=False,
                        help="Export the selected route to a HTML file")

    args = parser.parse_args()
    if args.end is None:
        args.end = args.start

    if args.fitness_mode == 'normal':
        graph_builder = Graph_Builder(args.start, args.end, simplify=args.simplify, all_peaks=args.all_peaks)
    elif args.fitness_mode == 'accessible':
        graph_builder = Graph_Builder(args.start, args.end, simplify=args.simplify, all_peaks=args.all_peaks, highway_score_function=accessible_highway_score, surface_score_function=accessible_surface_score, fitness_score_function=accessible_fitness)
    elif args.fitness_mode == 'custom':
        import config
        graph_builder = Graph_Builder(args.start, args.end, simplify=args.simplify, all_peaks=args.all_peaks, highway_score_function=config.custom_highway_score, surface_score_function=config.custom_surface_score, fitness_score_function=config.custom_fitness)

    astar = AStarArcs(graph_builder, args.distance, args.tolerance,
                     favor_descent=args.favor_descent, normal_nodes=args.normal_nodes, max_backtracks=args.max_backtracks)
    astar.boundary_generation()

    if args.mode == 'normal':
        routes = astar.normal_generation()
    else:
        routes = astar.peak_bagger_generation()

    if not routes:
        print("No routes found. Exiting.")
        return

    if args.peak_fitness == 'most':
        most_peaks = True
        least_peaks = False
    elif args.peak_fitness == 'least':
        most_peaks = False
        least_peaks = True
    else:
        most_peaks = False
        least_peaks = False
    route = astar.get_route(routes, best_fitness=(not args.worst_fitness), most_peaks=most_peaks,
                           least_peaks=least_peaks, closest_to_target=args.closest_distance)

    if args.output_gpx:
        astar.output_coords = [(astar.G.nodes[node]['y'], astar.G.nodes[node]['x']) for node in route]
        astar.export_to_gpx()
        print("Route exported to route.gpx")

    if args.output_html:
        astar.output_coords = [(astar.G.nodes[node]['y'], astar.G.nodes[node]['x']) for node in route]
        astar.export_to_html()
        print("Route map saved as route.html")

if __name__ == "__main__":
    main()