from neurons.lif import LIFNeuron


class SpikingLayer:

    def __init__(
        self, num_neurons: int, tau: float, v_rest: float, v_th: float, v_reset: float
    ):
        self.num_neurons = num_neurons
        self.neurons = [
            LIFNeuron(tau, v_rest, v_th, v_reset) for _ in range(num_neurons)
        ]

    def step(self, input_currents: list[float], dt: int):
        spikes = []

        for neuron, current in zip(self.neurons, input_currents):
            spikes.append(neuron.step(current, dt))

        return spikes
