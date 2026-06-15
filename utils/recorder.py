import time
import json
import numpy as np


class Recorder:
    def __init__(self, synapses, w_min=0.0, w_max=1.0, snapshot_every_n_samples=1000):
        self.synapses = synapses
        self.w_min = w_min
        self.w_max = w_max
        self.snapshot_every_n_samples = snapshot_every_n_samples

        self.firing_rates = []
        self.silent_fraction = []
        self.saturated_fraction = []
        self.weight_stats = []

        self.weight_snapshots = {}

        self.start_time = time.time()
        self.sample_count = 0

    def record_sample(self, layers, sample_idx):

        rates = [float(np.mean(layer.last_spikes)) for layer in layers]
        silent = [float(np.mean(layer.last_spikes == 0)) for layer in layers]
        saturated = [float(np.mean(layer.last_spikes == 1)) for layer in layers]

        self.firing_rates.append(rates)
        self.silent_fraction.append(silent)
        self.saturated_fraction.append(saturated)

        stats = []
        for i, syn in enumerate(self.synapses):
            w = syn.weights
            stats.append(
                {
                    "mean": float(np.mean(w)),
                    "std": float(np.std(w)),
                    "clipped_low": float(np.mean(w <= self.w_min + 1e-6)),
                    "clipped_high": float(np.mean(w >= self.w_max - 1e-6)),
                }
            )
        self.weight_stats.append(stats)

        if sample_idx % self.snapshot_every_n_samples == 0:
            self.weight_snapshots[sample_idx] = [
                syn.weights.copy() for syn in self.synapses
            ]

        self.sample_count += 1

    def save(self, path="run_record.npz"):
        elapsed = time.time() - self.start_time
        np.savez_compressed(
            path,
            firing_rates=np.array(self.firing_rates),
            silent_fraction=np.array(self.silent_fraction),
            saturated_fraction=np.array(self.saturated_fraction),
            weight_stats=np.array(
                [
                    [
                        [s["mean"], s["std"], s["clipped_low"], s["clipped_high"]]
                        for s in sample
                    ]
                    for sample in self.weight_stats
                ]
            ),
            elapsed_seconds=elapsed,
            sample_count=self.sample_count,
        )

        snap_path = path.replace(".npz", "_snapshots.npz")
        if self.weight_snapshots:
            np.savez_compressed(
                snap_path,
                **{
                    f"sample_{k}_synapse_{i}": np.array(w)
                    for k, v in self.weight_snapshots.items()
                    for i, w in enumerate(v)
                },
            )
        print(f"Saved to {path} ({elapsed:.1f}s, {self.sample_count} samples)")
