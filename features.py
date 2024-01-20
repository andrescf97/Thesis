import pandas as pd
import numpy as np
import networkx as nx


class StreetNetwork():
    def __init__(self):
        self.nodes = None
        self.links = None

def create_nx_from_matsim(matsim_graph):
    G = nx.DiGraph()
    G.add_nodes_from(list(matsim_graph.nodes["node_id"]))
    G.add_weighted_edges_from(matsim_graph.links[["from_node", "to_node", "length"]].values.tolist())
    return G

def read_in_network(street_network_file):
    net = StreetNetwork()
    nodes = pd.read_csv(street_network_file + "nodes.csv")
    links = pd.read_csv(street_network_file + "links.csv")
    net.nodes = nodes[(nodes["x"] > 4456269.0) & (nodes["x"] < 4478126.2) & (nodes["y"] > 5326682.0) & (nodes["y"] < 5341867.8)].reset_index(drop=True)
    net.links = links[((links["from_node"].isin(list(net.nodes["node_id"]))) & (links["to_node"].isin(list(net.nodes["node_id"]))))]
    net_nx = create_nx_from_matsim(net)
    return net, net_nx


class FeatureGenerator():
    def __init__(self, charger_data, vehicle_path, vehicle_type_path, population_path, street_network_path):
        self.charger_data = charger_data
        self.vehicle_data = pd.read_csv(vehicle_path, sep=";")
        self.vehicle_data = self.vehicle_data.merge(pd.read_csv(vehicle_type_path, sep=";"), left_on="vehicle_vehicle_type", right_on="type_name", how="left")
        self.population_data = pd.read_csv(population_path, sep=",")
        self.population_data = self.population_data.merge(self.vehicle_data.add_suffix('_vehicle'), left_on="id", right_on="vehicle_id_vehicle", how="left")
        self.street_network, self.nx_network = read_in_network(street_network_path)
        self.street_network.links = self.street_network.links.merge(self.street_network.nodes, left_on="from_node", right_on="node_id", how="left").rename(columns={"x": "starting_node_x", "y": "starting_node_y"})
        self.street_network.links = self.street_network.links.merge(self.street_network.nodes, left_on="to_node", right_on="node_id", how="left").rename(columns={"x": "arriving_node_x", "y": "arriving_node_y"})
        self.population_data["home_charger_exists"] = self.population_data["homeChargerPower"] > 0
        self.population_data["work_charger_exists"] = self.population_data["workChargerPower"] > 0
        #print("Add distance to work again!!!!!!!!!!!!!!!!!!!") # speedup
        self.population_data["distance_to_work"] = self.get_distance_to_work()


    def set_graph(self, graph):
        self.graph = graph

    def set_target_items(self, target_items):
        self.target_items = target_items

    def get_nearest_intersection(self, x, y):
        return self.street_network.nodes.iloc[np.argmin(np.square(self.street_network.nodes["x"] - x) + np.square(self.street_network.nodes["y"] - y))]

    def get_distance(self, start_x, start_y, end_x, end_y):
        source = self.get_nearest_intersection(start_x, start_y)
        target = self.get_nearest_intersection(end_x, end_y)
        return nx.shortest_path_length(self.nx_network, source=source["node_id"], target=target["node_id"])

    def get_distance_to_work(self):
        distance_to_work = []
        for person in self.population_data.to_dict("records"):
            try:
                distance_to_work.append(self.get_distance(start_x=person["home_x"], start_y=person["home_y"], end_x=person["work_x"], end_y=person["work_y"]))
            except:
                distance_to_work.append(0)
        return distance_to_work

    def get_distance_edges(self):
        return np.sqrt((self.graph.edges["starting_node_x"]-self.graph.edges["arriving_node_x"])**2 + (self.graph.edges["starting_node_y"]-self.graph.edges["arriving_node_y"])**2)

    def get_node_features_main_filler(self, population_data):
        return pd.DataFrame(np.zeros((self.graph.nodes.shape[0] - population_data.shape[0], population_data.shape[1])), columns=population_data.columns)

    def get_node_features_population(self):
        population_data = self.population_data.drop(columns=["id", "vehicle_charger_types_vehicle", "vehicle_id_vehicle", "vehicle_vehicle_type_vehicle", "type_name_vehicle"])
        return population_data

    def get_edge_features(self):
        print("START: generate edge features...")
        edge_features = {}
        edge_features["length"] = self.get_distance_edges()
        return pd.DataFrame(edge_features)
