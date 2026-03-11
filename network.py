import numpy as np

from layers.spiking_layer import SpikingLayer
from utils.poisson_input import poisson_input
from synapse import Synapse


class Network:

    def __init__(self, layers: list[SpikingLayer] = [], timesteps: int = 50):
        self.layers = layers
        self.timesteps = timesteps
        self.synapses: list[Synapse] = []
        self.spikes = []

    def add_layer(self, layer: SpikingLayer):
        self.layers.append(layer)

    def create_synapse(self, pre_layer_id: int, post_layer_id: int, learning_rule=None):
        self.synapses.append(
            Synapse(
                self.layers[pre_layer_id], self.layers[post_layer_id], learning_rule
            )
        )

    def run(self):
        inputs = poisson_input(
            num_neurons=5, firing_rate=50.0, dt=1.0, timesteps=self.timesteps
        )

        for t in range(self.timesteps):
            self.layers[0].input_current = np.dot(self.synapses[0].weights, inputs[t])

            for synapse in self.synapses:
                synapse.compute_current()

            for layer in self.layers:
                layer.step(t)

            for synapse in self.synapses:
                synapse.update_weights(t)

            self.spikes.append(self.layers[-1].last_spikes)

    def get_output_spikes(self):
        return np.array(self.spikes).T
