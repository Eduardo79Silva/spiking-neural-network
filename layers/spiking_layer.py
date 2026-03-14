import numpy as np

from neuron import LIFNeuron


class SpikingLayer:
    def __init__(
        self, num_neurons: int, tau, v_rest, v_th, v_reset, refractory_period=2
    ):
        self.num_neurons = num_neurons

        self.input_current = np.ndarray(shape=(1,))

        self.last_spikes = [0] * num_neurons

        self.neurons = [
            LIFNeuron(v_rest, v_th, v_reset, tau, refractory_period)
            for _ in range(num_neurons)
        ]

    def step(self, t: float):

        spikes = np.ndarray(shape=(self.num_neurons,))

        for i in range(self.num_neurons):
            spikes[i] = self.neurons[i].update(self.input_current[i], t)

        self.last_spikes = spikes

        return spikes
