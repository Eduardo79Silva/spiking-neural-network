import numpy as np
from tqdm import tqdm

from utils.recorder import Recorder
from utils.encoding import encode_poisson
from layers.spiking_layer import SpikingLayer
from synapse import Synapse


class Network:

    def __init__(self, layers: list[SpikingLayer] = [], timesteps: int = 50):
        self.layers = layers
        self.timesteps = timesteps
        self.synapses: list[Synapse] = []
        self.spikes = []
        self.weight_snapshots = {}
        self.firing_rate_history = []

    def add_layer(self, layer: SpikingLayer):
        self.layers.append(layer)

    def create_synapse(self, pre_layer_id: int, post_layer_id: int, learning_rule=None):
        self.synapses.append(
            Synapse(
                self.layers[pre_layer_id], self.layers[post_layer_id], learning_rule
            )
        )

    def reset_layers(self):
        for layer in self.layers[1:]:
            layer.last_spikes = np.zeros(layer.num_neurons, dtype=np.float32)
            layer.input_current = np.zeros(layer.num_neurons, dtype=np.float32)
            layer.reset_neurons()

    def run(self, inputs, record_every: int = 1000, predict=False):
        recorder = Recorder(
            synapses=self.synapses, snapshot_every_n_samples=record_every
        )
        outer = tqdm(inputs, desc="Training", unit="sample")
        self.spikes = []

        for sample_idx, (frame, _) in enumerate(outer):
            self.reset_layers()
            data = encode_poisson(frame, self.timesteps).squeeze().numpy()

            sample_spikes = np.zeros(self.layers[-1].num_neurons)

            for t in range(self.timesteps):
                self.layers[0].last_spikes = data[t]
                for synapse in self.synapses:
                    synapse.compute_current()
                for layer in self.layers[1:]:
                    layer.step(t)

                if not predict:
                    for synapse in self.synapses:
                        synapse.update_weights(t)
                else:
                    sample_spikes = np.sum(
                        [sample_spikes, self.layers[-1].last_spikes], axis=0
                    )

            if predict:
                self.spikes.append(sample_spikes)

            recorder.record_sample(self.layers, sample_idx)
            alpha = 0.01

            if not hasattr(self, "_ema"):
                self._ema = {"in": 0.0, "hid": 0.0, "out": 0.0}

            self._ema["in"] = (1 - alpha) * self._ema["in"] + alpha * float(
                np.mean(self.layers[0].last_spikes)
            )
            self._ema["hid"] = (1 - alpha) * self._ema["hid"] + alpha * float(
                np.mean(self.layers[1].last_spikes)
            )
            self._ema["out"] = (1 - alpha) * self._ema["out"] + alpha * float(
                np.mean(self.layers[-1].last_spikes)
            )

            outer.set_postfix(
                {
                    "in": f"{self._ema['in']:.3f}",
                    "hid": f"{self._ema['hid']:.3f}",
                    "out": f"{self._ema['out']:.3f}",
                    "w0": f"{recorder.weight_stats[-1][0]['mean']:.3f}",
                    "w1": f"{recorder.weight_stats[-1][1]['mean']:.3f}",
                }
            )

        recorder.save()
        return recorder

    def get_output_spikes(self):
        return np.array(self.spikes).T
