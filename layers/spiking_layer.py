import numpy as np


class SpikingLayer:
    def __init__(self, num_inputs: int, num_neurons: int, tau, v_rest, v_th, v_reset):
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs

        self.weights = np.random.normal(0.5, 0.1, (num_neurons, num_inputs))

        self.v = np.full(num_neurons, v_rest)
        self.tau = tau
        self.v_rest = v_rest
        self.v_th = v_th
        self.v_reset = v_reset

    def step(self, input_spikes: np.ndarray, dt: float):
        """
        input_spikes: binary vector (1s and 0s) from the source/previous layer
        """
        current = np.dot(self.weights, input_spikes)

        dv = (-(self.v - self.v_rest) / self.tau + current) * dt
        self.v += dv

        spikes = (self.v >= self.v_th).astype(float)

        self.v[spikes > 0] = self.v_reset

        return spikes
