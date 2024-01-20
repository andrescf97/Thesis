import tensorflow as tf
import keras
import numpy as np


class Graph_Neural_Network_subgraph():
    def __init__(self):
        ...

    def load(self):
        self.model = keras.models.load_model("./data/GNNPopulation_epochs-100000_batchS-1_rep-itemNode_featureS-100_target-chargerUtilization_bestVal")

    def predict(self, data):
        x = self.batch_dataset(data)
        y = np.squeeze(self.model(x))
        return sum(y)


    def batch_dataset(self, graph):

        def generate_tf_data(graph=None):
            # Convert lists to ragged tensors for tf.data.Dataset later on
            return (
                        tf.ragged.constant(graph["node_features"], dtype=tf.float32),
                        tf.ragged.constant(graph["edge_features"], dtype=tf.float32),
                        tf.ragged.constant(graph["edges"], dtype=tf.int64),
                    )

        X = generate_tf_data(graph)

        def prepare_batch(x_batch):
            node_features, edge_features, edges = x_batch
            edges = edges.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
            node_features = node_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
            edge_features = edge_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
            return (node_features, edge_features, edges)  # graph_node_indicator

        return prepare_batch(X)
