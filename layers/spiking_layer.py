import numpy as np


class SpikingLayer:
    def __init__(self, num_inputs: int, num_neurons: int, tau, v_rest, v_th, v_reset):
        self.num_neurons = num_neurons

        self.input_current = np.ndarray(shape=(1,))

        self.last_spikes = [0] * num_neurons

        self.v = np.full(num_neurons, v_rest)
        self.tau = tau
        self.v_rest = v_rest
        self.v_th = v_th
        self.v_reset = v_reset

    def step(self, dt: float):

        dv = (-(self.v - self.v_rest) / self.tau + self.input_current) * dt
        self.v += dv

        spikes = (self.v >= self.v_th).astype(float)

        self.v[spikes > 0] = self.v_reset

        self.last_spikes = spikes

        print(self.input_current)
        print()

        return spikes
