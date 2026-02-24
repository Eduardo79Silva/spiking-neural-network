import numpy as np
from layers.spiking_layer import SpikingLayer


class ConstantCurrentPopulationExperiment:

    def __init__(self, layer: SpikingLayer, T=50, k=5.0):
        self.layer = layer
        self.T = T
        self.k = k

    def build_inputs(self):
        return np.linspace(0.0, 1.0, self.layer.num_neurons)

    def run(self):
        inputs = self.build_inputs()
        spikes_over_time = []

        for t in range(self.T):
            currents = self.k * inputs
            spikes = self.layer.step(currents, t)
            spikes_over_time.append(spikes)

        return np.array(spikes_over_time).T
