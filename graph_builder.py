import math
import time

import osmnx as ox
import pandas as pd
import numpy as np
import geopandas as gpd
import requests
from shapely.geometry import Point
from geopy.distance import distance
from shapely.geometry.linestring import LineString
from pyrosm import OSM
from pyrosm import get_data


def default_highway_score(highway):
    if isinstance(highway, list):
        total = 0
        for h in highway:
            total += default_highway_score(h)
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


def default_surface_score(surface):
    return 1


def default_fitness_score(graph, n1, n2, data):
    return data["length"] * graph.nodes[n2]["fitness"] * data["highway_score"]


class Graph_Builder:
    def __init__(self, start_node, end_node, target_distance=10000, simplify=False,
                 highway_score_function=default_highway_score,
                 surface_score_function=default_surface_score,
                 fitness_score_function=default_fitness_score,
                 all_peaks=False,
                 server=False):
        self.target_distance = target_distance
        self.distance = target_distance * 0.6
        self.start_point = start_node
        print(f"Downloading Graph...")
        if server:
            ## Uses local file to get area
            # osm = OSM("derbyshire-latest.osm.pbf")
            # nodes, edges = osm.get_network(nodes=True, network_type="walking")
            # self.G = osm.to_graph(nodes, edges, graph_type="networkx")

            ## Uses Overpass API to get area
            ox.graph_from_place("Derbyshire, UK", network_type='walk', simplify=False)
        else:
            self.G = ox.graph_from_point(start_node, dist=self.distance, network_type='walk', simplify=False)

        self.add_highway_score_function = highway_score_function
        self.add_surface_score_function = surface_score_function
        self.add_fitness_score_function = fitness_score_function
        if simplify:
            print("Simplifying Graph...")
            self.G = ox.simplify_graph(self.G, remove_rings=True)
            self.remove_simplified_loops()
        print(f"Node Amount: {len(self.G.nodes)}")
        print(f"Edge Amount: {len(self.G.edges)}")
        print(f"Building Graph...")
        self.start_node = ox.nearest_nodes(self.G, Y=start_node[0], X=start_node[1])
        self.end_node = ox.nearest_nodes(self.G, Y=end_node[0], X=end_node[1])
        self.bbox = ox.utils_geo.bbox_from_point(self.start_point, dist=self.distance)
        t = time.time()
        print(f"\t- Adding Topo...")
        ox.settings.elevation_url_template = ("https://api.opentopodata.org/v1/eudem25m?locations={locations}")
        if server:
            ## API Method
            self.G = ox.add_node_elevations_google(self.G, batch_size=90, pause=1)

            ## Local File Method
            # self.G = ox.add_node_elevations_raster(self.G, filepath="output_AW3D30.tif", cpus=1)
        else:
            self.G = ox.add_node_elevations_google(self.G, batch_size=90, pause=1)
        self.G = ox.distance.add_edge_lengths(self.G)
        self.G = ox.add_edge_grades(self.G)

        grades = pd.Series([d["grade_abs"] for _, _, d in ox.convert.to_undirected(self.G).edges(data=True)])
        grades = grades.replace([np.inf, -np.inf], np.nan).dropna()

        print(f"\t- Adding Green...")
        self.add_green_score()
        print(f"\t- Adding Peaks...")
        if all_peaks:
            self.add_peaks()
        else:
            self.add_peaks_with_barrier_check()
        print(f"\t- Adding Distance...")
        self.add_distance_score()
        print(f"\t- Adding Surface...")
        self.add_surface()

        print(f"\t- Adding Combined Fitness...")
        self.add_combined_fitness()
        print(f"\t- Adding Edge Gradient...")
        self.add_edge_gradient()

        print(f"\t- Adding Edge Lengths...")
        self.edge_lengths = {}
        for u, v, data in self.G.edges(data=True):
            self.edge_lengths.setdefault(u, {})[v] = data['length']


        print(f"Graph Built in {time.time() - t}")

    def remove_simplified_loops(self):
        self.G.remove_edges_from([(u, v, k) for u, v, k in self.G.edges(keys=True) if u == v])


    def add_combined_fitness(self):
        for n, data in self.G.nodes(data=True):
            if "nature" in self.G.nodes[n] and self.G.nodes[n]["nature"].get("natural") == "peak":
                data["fitness"] = 1 / ((data["green_score"] * data["distance_score"] * 2) + 1)
            else:
                data["fitness"] = 1 / ((data["green_score"] * data["distance_score"] * 1) + 1)

    def add_peaks(self):
        tags = {"natural": "peak"}
        features = ox.features_from_bbox(bbox=self.bbox, tags=tags)
        feature_points = features.representative_point()
        nn = ox.distance.nearest_nodes(self.G, feature_points.x, feature_points.y)
        useful_tags = ["name", "ele", "natural"]
        for node, feature in zip(nn, features[useful_tags].to_dict(orient="records")):
            if isinstance(feature["name"], str):
                feature = {k: v for k, v in feature.items() if pd.notna(v)}
                self.G.nodes[node].update({"nature": feature})

    def add_peaks_within_distance(self):
        tags = {"natural": "peak"}
        features = ox.features_from_bbox(bbox=self.bbox, tags=tags)
        feature_points = features.representative_point()
        nn = ox.distance.nearest_nodes(self.G, feature_points.x, feature_points.y)
        useful_tags = ["name", "ele", "natural"]
        for node, point, feature in zip(nn, feature_points, features[useful_tags].to_dict(orient="records")):
            if isinstance(feature["name"], str):
                feature = {k: v for k, v in feature.items() if pd.notna(v)}
                distance1 = self.calculate_euclidean_distance((self.G.nodes[node]['y'], self.G.nodes[node]['x']),
                                                              (point.y, point.x))
                if distance((self.G.nodes[node]['y'], self.G.nodes[node]['x']), (point.y, point.x)).m <= 500:
                    self.G.nodes[node].update({"nature": feature})

    def add_peaks_with_barrier_check(self):
        tags = {"natural": "peak"}
        features = ox.features_from_bbox(bbox=self.bbox, tags=tags)
        feature_points = features.representative_point()
        nn = ox.distance.nearest_nodes(self.G, feature_points.x, feature_points.y)
        useful_tags = ["name", "ele", "natural"]
        barrier_tags = {"barrier": True}
        barriers = ox.features_from_bbox(bbox=self.bbox, tags=barrier_tags)
        barrier_geoms = barriers[barriers.geometry.type.isin(['LineString', 'Polygon'])].geometry
        barrier_sindex = barrier_geoms.sindex if not barrier_geoms.empty else None
        for node, point, feature in zip(nn, feature_points, features[useful_tags].to_dict(orient="records")):
            if isinstance(feature["name"], str):
                feature = {k: v for k, v in feature.items() if pd.notna(v)}
                node_point = Point(self.G.nodes[node]['x'], self.G.nodes[node]['y'])
                line = LineString([point, node_point])

                intersects_any = False
                if barrier_geoms.empty:
                    intersects_any = False  # No barriers, proceed
                else:
                    # Use spatial index to find candidate barriers
                    candidates = list(barrier_sindex.intersection(line.bounds))
                    intersects_any = any(line.intersects(barrier_geoms.iloc[candidate])
                                         for candidate in candidates)
                # Attach peak if no barrier intersects
                if not intersects_any:
                    self.G.nodes[node].update({"nature": feature})

    def view_peaks(self):
        node_colors = []
        for node in self.G.nodes:
            if "nature" in self.G.nodes[node] and self.G.nodes[node]["nature"].get("natural") == "peak":
                node_colors.append("red")  # Color for peak nodes
            else:
                node_colors.append("none")  # Color for other nodes
        fig, ax = ox.plot_graph(self.G, node_color=node_colors, node_size=10)

    def add_surface(self):
        features = ox.features_from_bbox(bbox=self.bbox, tags={"surface": True})
        feature_points = features.representative_point()
        nn = ox.distance.nearest_nodes(self.G, feature_points.x, feature_points.y)
        useful_tags = ["access", "surface"]
        for node, feature in zip(nn, features[useful_tags].to_dict(orient="records")):
            feature = {k: v for k, v in feature.items() if pd.notna(v)}
            self.G.nodes[node].update({"surface": feature})

    def view_green(self):
        node_colors = []
        for node in self.G.nodes:
            if "green_score" in self.G.nodes[node] and self.G.nodes[node]["green_score"]:
                node_colors.append("green")
            else:
                node_colors.append("none")
        fig, ax = ox.plot_graph(self.G, node_color=node_colors, node_size=10)

    def add_distance_score(self):
        end_node_coords = (self.G.nodes[self.end_node]['y'], self.G.nodes[self.end_node]['x'])
        distances = []
        for node, data in self.G.nodes(data=True):
            node_coords = (data['y'], data['x'])
            distance = math.sqrt((node_coords[0] - end_node_coords[0]) ** 2 +
                                 (node_coords[1] - end_node_coords[1]) ** 2)
            distances.append((node, distance))
        max_distance = max(distances, key=lambda x: x[1])[1]
        min_distance = min(distances, key=lambda x: x[1])[1]
        for node, distance in distances:
            if max_distance == min_distance:
                self.G.nodes[node]['distance_score'] = 1.0
            else:
                self.G.nodes[node]['distance_score'] = 1 - ((distance - min_distance) / (max_distance - min_distance))

    def view_distance_score(self):
        distance_scores = [data['distance_score'] for _, data in self.G.nodes(data=True)]
        fig, ax = ox.plot_graph(self.G, node_color=distance_scores, node_size=10)

    def view_surface_score(self):
        ec = ox.plot.get_edge_colors_by_attr(self.G, "surface_score", cmap="plasma", num_bins=5, equal_size=False)
        fig, ax = ox.plot.plot_graph(self.G, edge_color=ec, edge_linewidth=0.5, node_size=0)

    def view_highway_score(self):
        ec = ox.plot.get_edge_colors_by_attr(self.G, "highway_score", cmap="plasma", num_bins=5, equal_size=False)
        fig, ax = ox.plot.plot_graph(self.G, edge_color=ec, edge_linewidth=1, node_size=0)

    def green_spaces(self):
        url = "https://services.arcgis.com/JJzESW51TqeY9uat/arcgis/rest/services/CRoW_Act_2000_Access_Layer/FeatureServer/0/query"

        params = {
            "where": "1=1",  # Adjust this to filter data if needed
            "outFields": "Descrip",  # Include all fields
            "outSR": "4326",  # Output spatial reference (WGS84)
            "f": "geojson",  # Output format
            "geometry": f"{self.bbox[0]},{self.bbox[1]},{self.bbox[2]},{self.bbox[3]}",  # Bounding box
            "geometryType": "esriGeometryEnvelope",  # Bounding box geometry type
            "inSR": "4326",  # Input spatial reference (WGS84)
            "spatialRel": "esriSpatialRelIntersects"  # Spatial relationship (intersects)
        }

        response = requests.get(url, params=params)
        data = response.json()
        if not data["features"]:
            print("No Green Spaces Found")
            return None
        gdf = gpd.GeoDataFrame.from_features(data["features"])
        gdf.set_crs(epsg=4326, inplace=True)

        return gdf

    def add_green_score(self):
        green_spaces_shape = self.green_spaces()
        # Adds a green score to each node based on the amount of neighboring green spaces
        for n, data in self.G.nodes(data=True):
            if green_spaces_shape is None:
                data["green_score"] = 1
            else:
                data["green_score"] = sum(
                    green_spaces_shape.intersects(Point(self.G.nodes[x]["x"], self.G.nodes[x]["y"])).any() for x in
                    self.G.neighbors(n))

    def get_route_fitness(self, route):
        return sum(self.G.nodes[n]["fitness"] for n in route)

    def get_route_fitness_edges(self, route):
        return sum(self.G.edges[u, v, 0]["fitness"] for u, v in zip(route[:-1], route[1:]))

    def add_edge_gradient(self):
        def impedance(length, grade):
            penalty = grade ** 2
            return length * penalty

        for n1, n2, _, data in self.G.edges(keys=True, data=True):
            data["impedance"] = impedance(data["length"], data["grade_abs"])
            data["rise"] = data["length"] * data["grade"]
            data["highway_score"] = self.add_highway_score_function(data["highway"])
            try:
                if self.G.nodes[n2]["surface"]["access"] == 'private':
                    data["access_score"] = 99
                elif self.G.nodes[n2]["surface"]["access"] == "yes":
                    data["access_score"] = 0.25
                else:
                    data["access_score"] = 1
            except:
                data["access_score"] = 2
            try:
                if self.G.nodes[n2]["surface"]:
                    data["surface_score"] = self.add_surface_score_function(n2["surface"])
            except:
                data["surface_score"] = 10
            data["fitness"] = self.add_fitness_score_function(self.G, n1, n2, data)


    def calculate_path_distance(self, route):
        x = []
        for u, v in zip(route[:-1], route[1:]):
            x.append(self.G[u][v][0]['length'])
        return sum(x)

    def get_edge_lengths(self):
        return self.edge_lengths

    def get_graph(self):
        return self.G

    def get_peak_nodes(self, route):
        peaks = []
        for node in route:
            if "nature" in self.G.nodes[node] and self.G.nodes[node]["nature"].get("natural") == "peak":
                peaks.append(node)
        return peaks

    def is_peak(self, node):
        if "nature" in self.G.nodes[node] and self.G.nodes[node]["nature"].get("natural") == "peak":
            return True
        return False

    def get_route_distance(self, route):
        distance = 0
        for i in range(len(route) - 1):
            distance += self.G.edges[route[i], route[i + 1], 0]['length']
        return distance

    def get_route_assent(self, route):
        assent = 0
        for i in range(len(route) - 1):
            rise = self.G.edges[route[i], route[i + 1], 0]['rise']
            if rise > 0:
                assent += rise
        return assent

    def get_route_descent(self, route):
        descent = 0
        for i in range(len(route) - 1):
            rise = self.G.edges[route[i], route[i + 1], 0]['rise']
            if rise < 0:
                descent += rise
        return descent

    def route_to_coords(self, route):
        return [(self.G.nodes[node]['y'], self.G.nodes[node]['x']) for node in route]

    def get_start_node(self):
        return self.start_node

    def get_end_node(self):
        return self.end_node

    def set_start_node(self, coords):
        self.start_node = ox.nearest_nodes(self.G, Y=float(coords[0]), X=float(coords[1]))

    def set_end_node(self, coords):
        self.end_node = ox.nearest_nodes(self.G, Y=float(coords[0]), X=float(coords[1]))

    def calculate_euclidean_distance(self, start, end):
        return math.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)

    def get_rise(self, u, v):
        rise = (self.G.nodes[u]["elevation"] - self.G.nodes[v]["elevation"])
        # if rise != self.G[u][v][0]["rise"]:
        #     print(f"Rise: {rise}, Rise2: {self.G[u][v][0]['rise']}")
        return rise


