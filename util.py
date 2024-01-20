import pandas as pd
import numpy as np
from features import FeatureGenerator


def get_data_directories():
    return {"charger_file": f"./data/chargerScenario_seed-1.csv",
            "vehicle_file": f"./data/evehicles.csv",
            "population_file": f"./data/population.csv",
            "vehicle_type_file": f"./data/vehicletypes.csv",
            "street_network_file": f"./data/network_"}


class Graph():
    def __init__(self):
        self.nodes = None
        self.nodes_features = None
        self.edges = None
        self.edges_features = None

    def get_node_indices(self):
        return list(self.nodes["index"])

    def get_edge_indices(self):
        return np.array(self.edges[["starting_node", "arriving_node"]])

    def get_node_features_array(self):
        return self.nodes.values

    def get_edge_features_array(self):
        return self.edges.values


def create_training_data(training_data_directories):

    def get_pop_main_features(feature_generator):
        node_population_features = feature_generator.get_node_features_population()
        nodes_main_features = feature_generator.get_node_features_main_filler(node_population_features)
        feature_generator.graph.nodes_features = pd.concat([nodes_main_features, node_population_features])
        feature_generator.graph.edges_features = feature_generator.graph.edges[["weights_distance", "boolean_home", "boolean_work"] if "boolean_home" in feature_generator.graph.edges.columns else ["weights_distance"]]

    def generate_population_graph(target_representation, population_data, street_network):
        def get_edges_population(nodes_main, nodes_pop):
            nodes_starting = []
            nodes_ingoing = []
            weights_distance = []
            boolean_home = []
            boolean_work = []
            for node_pop in nodes_pop.to_dict("records"):
                main_node_idx_home = np.argmin(np.square(nodes_main["x"] - node_pop["home_x"]) + np.square(nodes_main["y"] - node_pop["home_y"]))
                main_node_idx_work = np.argmin(np.square(nodes_main["x"] - node_pop["work_x"]) + np.square(nodes_main["y"] - node_pop["work_y"]))
                nodes_starting.append(node_pop["index"])
                nodes_ingoing.append(main_node_idx_home)
                weights_distance.append(np.sqrt(np.square(nodes_main.iloc[main_node_idx_home]["x"] - node_pop["home_x"]) + np.square(nodes_main.iloc[main_node_idx_home]["y"] - node_pop["home_y"])))
                boolean_home.append(1)
                boolean_work.append(0)
                nodes_starting.append(node_pop["index"])
                nodes_ingoing.append(main_node_idx_work)
                weights_distance.append(np.sqrt(np.square(nodes_main.iloc[main_node_idx_work]["x"] - node_pop["work_x"]) + np.square(nodes_main.iloc[main_node_idx_work]["y"] - node_pop["work_y"])))
                boolean_home.append(0)
                boolean_work.append(1)
            return pd.DataFrame({"starting_node": nodes_starting, "arriving_node": nodes_ingoing, "weights_distance": weights_distance, "boolean_home": boolean_home, "boolean_work": boolean_work})

        graph = Graph()
        nodes = target_representation
        nodes_pop = population_data
        nodes_pop["x"] = nodes_pop["home_x"]
        nodes_pop["y"] = nodes_pop["home_y"]
        nodes_pop["index"] = max(nodes.index) + 1 + nodes_pop.index
        graph.edges = get_edges_population(nodes, nodes_pop)
        graph.nodes = pd.concat([pd.DataFrame(np.zeros((nodes.shape[0], nodes_pop.shape[1])), columns=nodes_pop.columns), nodes_pop], axis=0)  # np.array(list(nodes["index"]) + list(nodes_pop["index"]))
        return graph, len(nodes), None

    def get_target_charger_utilization():
        targets = training_data_directories["charger_data"].rename(columns={"charger_x": "x", "charger_y": "y"})
        targets["item_id"] = targets["charger_id"].astype("int")
        return targets

    training_data = {"node_features": [], "edge_features": [], "targets": [], "num_main_nodes": [], "num_main_edges": [], "edges": [], "target_representation": [],
                     "assignments": [], "node_features_pandas": [], "representation_itemNode": [], "representation_cellNode": [], "representation_holisticNode": [], "output_directory": []}

    # Initialize feature generator
    feature_generator = FeatureGenerator(charger_data=training_data_directories["charger_data"],
                                         vehicle_path=training_data_directories["vehicle_file"],
                                         vehicle_type_path=training_data_directories["vehicle_type_file"],
                                         population_path=training_data_directories["population_file"],
                                         street_network_path=training_data_directories["street_network_file"])

    # Get the item of interest which produces KPIs
    # Attributes: x, y, KPI-value
    target_items = get_target_charger_utilization()
    feature_generator.set_target_items(target_items)

    # The index identifies a single target representation and serves as identifier
    target_items["index"] = target_items.index

    # Detail different graph structures -> nodes / edges
    graph, num_main_nodes, num_main_edges = generate_population_graph(target_items, feature_generator.population_data, feature_generator.street_network)
    feature_generator.set_graph(graph)

    # Generate features
    get_pop_main_features(feature_generator)

    ### SET TRAINING DATA
    training_data["num_main_nodes"].append(num_main_nodes)
    training_data["num_main_edges"].append(num_main_edges)
    training_data["node_features"].append(feature_generator.graph.nodes_features.values)
    training_data["node_features_pandas"].append(feature_generator.graph.nodes_features)  # needed for visualization of features
    training_data["edge_features"].append(feature_generator.graph.edges_features.values)
    training_data["edges"].append(feature_generator.graph.get_edge_indices())
    training_data["target_representation"].append(target_items)
    return training_data