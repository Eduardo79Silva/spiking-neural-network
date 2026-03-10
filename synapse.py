import numpy as np
from layers.spiking_layer import SpikingLayer
from rules.stdp import STDP


class Synapse:

    def __init__(
        self, pre_layer: SpikingLayer, post_layer: SpikingLayer, learning_rule=STDP()
    ):
        self.pre_layer = pre_layer
        self.post_layer = post_layer
        self.weights = np.random.normal(
            0.5, 0.1, (post_layer.num_neurons, pre_layer.num_neurons)
        )
        self.learning_rule = learning_rule

    def step(self, t):
        pre_spikes = self.pre_layer.last_spikes

        passing_current = self.weights @ pre_spikes
        self.post_layer.step(passing_current, t)

        post_spikes = self.post_layer.last_spikes
        self.learning_rule.update(pre_spikes, post_spikes, self.weights, t)
