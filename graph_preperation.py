import json
import time
import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
import geopandas as gpd
from matplotlib import pyplot as plt
import requests
from shapely.geometry import Point


class Graph_Builder:
    def __init__(self, start_node, end_node, target_distance=10000):
        self.target_distance = target_distance
        self.distance = 5000
        self.start_point = start_node
        self.G = ox.graph_from_point(start_node, dist=5000, network_type='walk', simplify=True)
        self.start_node = ox.nearest_nodes(self.G, X=start_node[1], Y=start_node[0])
        self.end_node = ox.nearest_nodes(self.G, Y=end_node[0], X=end_node[1])
        t=time.time()
        ox.settings.elevation_url_template = ("https://api.opentopodata.org/v1/eudem25m?locations={locations}")
        self.G = ox.add_node_elevations_google(self.G, batch_size=100, pause=1)
        self.G = ox.add_edge_grades(self.G)

        grades = pd.Series([d["grade_abs"] for _, _, d in ox.convert.to_undirected(self.G).edges(data=True)])
        grades = grades.replace([np.inf, -np.inf], np.nan).dropna()

        # nc = ox.plot.get_node_colors_by_attr(self.G, "elevation", cmap="plasma")
        # fig, ax = ox.plot.plot_graph(self.G, node_color=nc, node_size=5, edge_color="#333333", bgcolor="k")
        #
        # ec = ox.plot.get_edge_colors_by_attr(self.G, "grade_abs", cmap="plasma", num_bins=5, equal_size=True)
        # fig, ax = ox.plot.plot_graph(self.G, edge_color=ec, edge_linewidth=0.5, node_size=0, bgcolor="k")
        self.green_spaces_shape = self.green_spaces()
        self.get_peaks()
        # self.view_peaks()


        self.add_green_score()

        # node_colors = []
        # for node in self.G.nodes:
        #     if "green_score" in self.G.nodes[node] and self.G.nodes[node]["green_score"]:
        #         node_colors.append("green")
        #     else:
        #         node_colors.append("none")
        # fig, ax = ox.plot_graph(self.G, node_color=node_colors, node_size=10)


        self.extend_nodes()
        print(time.time()-t)

        print(self.G[self.start_node])


        route0 = self.shortest_path_with_elevation_favor_and_target_distance()
        self.print_route_stats(route0)
        route1 = self.shortest_path_with_trip_distance()
        self.print_route_stats(route1)
        route2 = self.shortest_path_with_trip_impedance()
        self.print_route_stats(route2)

    def get_peaks(self):
        tags = {"natural": "peak"}
        features = ox.features_from_place("Peak District, United Kingdom", tags)
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


    def green_spaces(self):
        url = "https://services.arcgis.com/JJzESW51TqeY9uat/arcgis/rest/services/CRoW_Act_2000_Access_Layer/FeatureServer/0/query"

        bbox = ox.utils_geo.bbox_from_point(self.start_point, dist=self.distance)


        print(bbox)
        params = {
            "where": "1=1",  # Adjust this to filter data if needed
            "outFields": "Descrip",  # Include all fields
            "outSR": "4326",  # Output spatial reference (WGS84)
            "f": "geojson",  # Output format
            "geometry": f"{bbox[3]},{bbox[1]},{bbox[2]},{bbox[0]}",  # Bounding box
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
        print(gdf.head())

        return gdf

    def impedance_distance(self, length, grade, total_distance):
        # Reward higher elevations
        elevation_gain = abs(grade * length)  # Elevation gain is the absolute rise
        elevation_reward = 1 / (1 + elevation_gain)  # Higher elevation gain reduces impedance

        # Penalize deviations from the target distance
        distance_penalty = abs(total_distance - self.target_distance) / self.target_distance

        # Combine elevation reward and distance penalty
        return length * (elevation_reward + distance_penalty)

    def shortest_path_with_elevation_favor_and_target_distance(self):
        # Create a custom weight that rewards higher elevations and penalizes deviations from the target distance
        for u, v, k, data in self.G.edges(keys=True, data=True):
            # Calculate the cumulative distance up to this edge
            cumulative_distance = nx.shortest_path_length(self.G, source=self.start_node, target=u, weight="length") + \
                                  data["length"]
            data["elevation_favor_distance"] = self.impedance_distance(data["length"], data["grade_abs"], cumulative_distance)

        # Find the shortest path based on the custom weight
        route_by_elevation_favor_distance = ox.routing.shortest_path(self.G, self.start_node, self.end_node,
                                                                     weight="elevation_favor_distance")

        # Plot the route
        ec = ox.plot.get_edge_colors_by_attr(self.G, "grade_abs", cmap="plasma", num_bins=5, equal_size=True)
        fig, ax = ox.plot.plot_graph_route(self.G, route_by_elevation_favor_distance, node_size=0, edge_color=ec)

        return route_by_elevation_favor_distance

    # def add_elevation(self):
    def impedance(self, length, grade):
        penalty = grade ** 2
        return length * penalty

    def add_green_score(self):
        for n, data in self.G.nodes(data=True):
            # print(self.green_spaces_shape.intersects(Point(data["x"], data["y"])).any())
            data["green_score"] = self.green_spaces_shape.intersects(Point(data["x"], data["y"])).any()

    def extend_nodes(self):
        for n1, n2, _, data in self.G.edges(keys=True, data=True):
            data["impedance"] = self.impedance(data["length"], data["grade_abs"])
            data["rise"] = data["length"] * data["grade"]
            # data["greenary"] = self.green_spaces_shape.intersects(data).any()
            # print(self.green_spaces_shape.intersects(data).any())

    def shortest_path_with_trip_distance(self):
        route_by_length = ox.routing.shortest_path(self.G, self.start_node, self.end_node, weight="length")
        fig, ax = ox.plot.plot_graph_route(self.G, route_by_length, node_size=0)
        return route_by_length

    def shortest_path_with_trip_impedance(self):
        route_by_impedance = ox.routing.shortest_path(self.G, self.start_node, self.end_node, weight="impedance")
        ec = ox.plot.get_edge_colors_by_attr(self.G, "grade_abs", cmap="plasma", num_bins=5, equal_size=True)
        fig, ax = ox.plot.plot_graph_route(self.G, route_by_impedance, node_size=0, edge_color=ec)
        return route_by_impedance

    def print_route_stats(self, route):
        route_grades = ox.routing.route_to_gdf(self.G, route, weight="grade_abs")["grade_abs"]
        msg = "The average grade is {:.1f}% and the max is {:.1f}%"
        print(msg.format(np.mean(route_grades) * 100, np.max(route_grades) * 100))

        route_rises = ox.routing.route_to_gdf(self.G, route, weight="rise")["rise"]
        ascent = np.sum([rise for rise in route_rises if rise >= 0])
        descent = np.sum([rise for rise in route_rises if rise < 0])
        msg = "Total elevation change is {:.1f} meters: {:.0f} meter ascent and {:.0f} meter descent"
        print(msg.format(np.sum(route_rises), ascent, abs(descent)))

        route_lengths = ox.routing.route_to_gdf(self.G, route, weight="length")["length"]
        print(f"Total trip distance: {np.sum(route_lengths):,.0f} meters")


class GreenSpaces:
    def __init__(self):
        self.G = ox.graph_from_point((53.36486137451511, -1.8160056925378616), dist=5000, network_type='walk',
                                     simplify=True)
        self.file_path = "data/opgrsp_gb.gpkg"
        print("Reading Green Spaces")
        self.green_spaces = gpd.read_file(self.file_path)
        print(self.green_spaces.head())
        print(self.green_spaces.columns)
        self.geometry = self.green_spaces.geometry.to_list()
        # gdf = gpd.GeoDataFrame(self.green_spaces, geometry=self.geometry, crs="EPSG:3857")
        # gdf.to_file("data/green_spaces.geojson", driver="GeoJSON")
        fig, ax = ox.plot_graph(self.G, show=False, close=False, edge_color="gray")
        self.geometry.plot(ax=ax, color="green", alpha=0.5)
        plt.show()


if __name__ == "__main__":
    start_node = (53.36486137451511, -1.8160056925378616)
    end_node = (53.34344386440596, -1.778107050662822)
    gb = Graph_Builder(start_node, end_node)
    # gs = GreenSpaces()
    print("Done")
