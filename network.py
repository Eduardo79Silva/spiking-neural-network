from layers.spiking_layer import SpikingLayer
from synapse import Synapse


class Network:

    def __init__(self, layers: SpikingLayer = []):
        self.layers = []
        self.synapses = []

    def add_layer(self, layer: SpikingLayer):
        self.layers.append(layer)

    def create_synapse(self, pre_layer_id: int, post_layer_id: int):
        self.synapses.append(
            Synapse(self.layers[pre_layer_id], self.layers[post_layer_id])
        )
