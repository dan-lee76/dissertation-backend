import math
import time
import osmnx as ox
import pandas as pd
import numpy as np
import geopandas as gpd
from matplotlib import pyplot as plt
import requests
from shapely.geometry import Point
import folium
from folium.plugins import HeatMap


class Graph_Builder:
    def __init__(self, start_node, end_node, target_distance=10000, simplify=True):
        self.target_distance = target_distance
        self.distance = target_distance * 0.55
        self.start_point = start_node
        self.G = ox.graph_from_point(start_node, dist=self.distance, network_type='walk', simplify=simplify)
        print(f"Node Amount: {len(self.G.nodes)}")
        print(f"Edge Amount: {len(self.G.edges)}")
        print(f"Building Graph...")
        self.start_node = ox.nearest_nodes(self.G, Y=start_node[0], X=start_node[1])
        self.end_node = ox.nearest_nodes(self.G, Y=end_node[0], X=end_node[1])
        self.bbox = ox.utils_geo.bbox_from_point(self.start_point, dist=self.distance)
        t=time.time()
        ox.settings.elevation_url_template = ("https://api.opentopodata.org/v1/eudem25m?locations={locations}")
        self.G = ox.add_node_elevations_google(self.G, batch_size=100, pause=1)
        self.G = ox.distance.add_edge_lengths(self.G)
        self.G = ox.add_edge_grades(self.G)

        grades = pd.Series([d["grade_abs"] for _, _, d in ox.convert.to_undirected(self.G).edges(data=True)])
        grades = grades.replace([np.inf, -np.inf], np.nan).dropna()

        self.green_spaces_shape = self.green_spaces()
        self.add_green_score()
        self.get_peaks()
        self.add_distance_score()
        # self.view_distance_score()

        # self.view_peaks()


        # self.view_green()

        self.add_combined_fitness()
        self.add_edge_gradient()
        ec = ox.plot.get_edge_colors_by_attr(self.G, "fitness", cmap="plasma", num_bins=5, equal_size=False)
        fig, ax = ox.plot.plot_graph(self.G, edge_color=ec, edge_linewidth=0.5, node_size=0, bgcolor="k")
        ox.plot_graph(self.G, node_color="green", node_size=10, edge_linewidth=0.5, edge_color='gray')
        print(f"Graph Built in {time.time()-t}")

        # self.generate_folium_map_with_layers()

    def generate_folium_map_with_layers(self):
        # Create a Folium map centered at the start node
        map_center = self.start_point
        folium_map = folium.Map(location=map_center, zoom_start=14, tiles="OpenTopoMap")

        # Layer 1: Fitness Heatmap
        fitness_data = []
        for node, data in self.G.nodes(data=True):
            if "fitness" in data:
                fitness_value = data["fitness"]
                fitness_data.append([data['y'], data['x'], fitness_value])

        # Normalize fitness values (optional)
        fitness_values = [data[2] for data in fitness_data]
        max_fitness = max(fitness_values) if fitness_values else 1
        if max_fitness > 0:
            fitness_data = [[data[0], data[1], data[2] / max_fitness] for data in fitness_data]

        fitness_heatmap = HeatMap(fitness_data, name="Fitness Heatmap", show=False)
        fitness_heatmap.add_to(folium_map)

        # Layer 2: Peaks (nature == "peak")
        peak_data = []
        for node, data in self.G.nodes(data=True):
            if "nature" in data and data["nature"].get("natural") == "peak":
                peak_data.append([data['y'], data['x'], 1])  # Use a constant value for peaks

        peak_heatmap = HeatMap(peak_data, name="Peaks", show=False)
        peak_heatmap.add_to(folium_map)

        # Layer 3: Green Score
        green_data = []
        for node, data in self.G.nodes(data=True):
            if "green_score" in data:
                green_value = data["green_score"]
                green_data.append([data['y'], data['x'], green_value])

        # Normalize green scores (optional)
        green_values = [data[2] for data in green_data]
        max_green = max(green_values) if green_values else 1
        if max_green > 0:
            green_data = [[data[0], data[1], data[2] / max_green] for data in green_data]

        green_heatmap = HeatMap(green_data, name="Green Score", show=False)
        green_heatmap.add_to(folium_map)

        # Layer 4: Distance
        distance_data = []
        for node, data in self.G.nodes(data=True):
            if "distance_score" in data:
                distance_data.append([data['y'], data['x'], data["distance_score"]])

        distance_heatmap = HeatMap(distance_data, name="Distance Score", show=False)
        distance_heatmap.add_to(folium_map)

        # Add layer control to toggle layers
        folium.LayerControl().add_to(folium_map)

        # Save the map to an HTML file or display it
        folium_map.save("folium_map_with_layers_with_distance.html")
        return folium_map


    def add_combined_fitness(self):
        for n, data in self.G.nodes(data=True):
            # data["fitness"] = data["green_score"] +
            if "nature" in self.G.nodes[n] and self.G.nodes[n]["nature"].get("natural") == "peak":
                # print((data["green_score"]  * data["elevation"] * data["distance_score"]) * 2)
                data["fitness"] = abs(1000-((data["green_score"] * data["elevation"] * data["distance_score"]) * 2))
            else:
                data["fitness"] = abs(1000-((data["green_score"] * data["elevation"] * data["distance_score"]) * 1))
            # if data["fitness"] == 0:
            #     data["fitness"] = 999999

    def get_peaks(self):
        tags = {"natural": "peak"}
        features = ox.features_from_bbox(bbox=self.bbox, tags=tags)
        feature_points = features.representative_point()
        nn = ox.distance.nearest_nodes(self.G, feature_points.x, feature_points.y)
        useful_tags = ["name", "ele", "natural"]
        for node, feature in zip(nn, features[useful_tags].to_dict(orient="records")):
            feature = {k: v for k, v in feature.items() if pd.notna(v)}
            self.G.nodes[node].update({"nature": feature})

    def view_peaks(self):
        node_colors = []
        for node in self.G.nodes:
            if "nature" in self.G.nodes[node] and self.G.nodes[node]["nature"].get("natural") == "peak":
                node_colors.append("red")  # Color for peak nodes
            else:
                node_colors.append("none")  # Color for other nodes
        fig, ax = ox.plot_graph(self.G, node_color=node_colors, node_size=10)

    def view_green(self):
        node_colors = []
        for node in self.G.nodes:
            if "green_score" in self.G.nodes[node] and self.G.nodes[node]["green_score"]:
                node_colors.append("green")
            else:
                node_colors.append("none")
        fig, ax = ox.plot_graph(self.G, node_color=node_colors, node_size=10)

    def add_distance_score(self):
        # Get the coordinates of the end_node
        end_node_coords = (self.G.nodes[self.end_node]['y'], self.G.nodes[self.end_node]['x'])

        # Calculate distances and store them in a list
        distances = []
        for node, data in self.G.nodes(data=True):
            node_coords = (data['y'], data['x'])
            distance = math.sqrt((node_coords[0] - end_node_coords[0]) ** 2 +
                                 (node_coords[1] - end_node_coords[1]) ** 2)
            distances.append((node, distance))

        # Find the maximum and minimum distances for normalization
        max_distance = max(distances, key=lambda x: x[1])[1]
        min_distance = min(distances, key=lambda x: x[1])[1]

        # Normalize distances and assign distance_score
        for node, distance in distances:
            if max_distance == min_distance:
                # If all distances are the same, assign a default score (e.g., 1)
                self.G.nodes[node]['distance_score'] = 1.0
            else:
                # Normalize to a range of 0 to 1 (higher score for closer nodes)
                self.G.nodes[node]['distance_score'] = 1 - ((distance - min_distance) / (max_distance - min_distance))

    def view_distance_score(self):
        # Get the distance scores
        distance_scores = [data['distance_score'] for _, data in self.G.nodes(data=True)]

        # Plot the graph with node colors based on distance scores
        fig, ax = ox.plot_graph(self.G, node_color=distance_scores, node_size=10, edge_linewidth=0.5, edge_color='gray')

    def green_spaces(self):
        url = "https://services.arcgis.com/JJzESW51TqeY9uat/arcgis/rest/services/CRoW_Act_2000_Access_Layer/FeatureServer/0/query"

        params = {
            "where": "1=1",  # Adjust this to filter data if needed
            "outFields": "Descrip",  # Include all fields
            "outSR": "4326",  # Output spatial reference (WGS84)
            "f": "geojson",  # Output format
            "geometry": f"{self.bbox[3]},{self.bbox[1]},{self.bbox[2]},{self.bbox[0]}",  # Bounding box
            "geometryType": "esriGeometryEnvelope",  # Bounding box geometry type
            "inSR": "4326",  # Input spatial reference (WGS84)
            "spatialRel": "esriSpatialRelIntersects"  # Spatial relationship (intersects)
        }

        response = requests.get(url, params=params)
        data = response.json()

        # Load the GeoJSON data
        gdf = gpd.GeoDataFrame.from_features(data["features"])

        # Ensure the CRS is set to WGS84 (EPSG:4326)
        gdf.set_crs(epsg=4326, inplace=True)

        return gdf

    def add_green_score(self):
        # Adds a green score to each node based on the amount of neighboring green spaces
        for n, data in self.G.nodes(data=True):
            data["green_score"] = sum(self.green_spaces_shape.intersects(Point(self.G.nodes[x]["x"], self.G.nodes[x]["y"])).any() for x in self.G.neighbors(n))

    def get_route_fitness(self, route):
        return sum(self.G.nodes[n]["fitness"] for n in route)

    def add_surface_score(self, highway):
        if highway in ["footway", "path", "steps", "pedestrian", "track", "bridleway", "cycleway"]:
            return 0.5
        elif highway in ["residential", "living_street", "service", "unclassified", "road"]:
            return 0.5
        elif highway in ["tertiary", "tertiary_link", "secondary", "secondary_link", "primary", "primary_link"]:
            return 1
        elif highway in ["trunk", "trunk_link", "motorway", "motorway_link"]:
            return 1.5
        else:
            return 2

    def add_edge_gradient(self):
        def impedance(length, grade):
            penalty = grade ** 2
            return length * penalty

        for n1, n2, _, data in self.G.edges(keys=True, data=True):
            data["impedance"] = impedance(data["length"], data["grade_abs"])
            data["rise"] = data["length"] * data["grade"]
            data["surface_score"] = self.add_surface_score(data["highway"])
            data["fitness"] = ((self.G.nodes[n1]["fitness"] + self.G.nodes[n2]["fitness"]))
            # print(data["rise"])
            # data["homing"] = data["fitness"] * data["rise"]

    def calculate_path_distance(self, route):
        x = []
        for u, v in zip(route[:-1], route[1:]):
            x.append(self.G[u][v][0]['length'])
        return sum(x)

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

    def calculate_euclidean_distance(self, start, end):
        return math.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)

    def get_rise(self, u, v):
        rise = (self.G.nodes[u]["elevation"] - self.G.nodes[v]["elevation"])
        # if rise != self.G[u][v][0]["rise"]:
        #     print(f"Rise: {rise}, Rise2: {self.G[u][v][0]['rise']}")
        return rise

if __name__ == "__main__":
    start_node = (53.36486137451511, -1.8160056925378616)
    end_node = (53.34344386440596, -1.778107050662822)
    gb = Graph_Builder(start_node, end_node)
    # gs = GreenSpaces()
    print("Done")
