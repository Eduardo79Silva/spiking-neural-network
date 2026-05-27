import argparse
import numpy as np
import random

from loaders.mnist import load_mnist
from layers.spiking_layer import SpikingLayer
from rules.stdp import STDP
from utils.plotting import (
    plot_raster,
    plot_firing_rates,
    plot_weight_distributions,
    plot_weight_heatmaps,
    plot_firing_rate_history,
)
from network import Network


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Spiking Neural Network Framework")

    parser.add_argument(
        "--experiment",
        type=str,
        choices=["single", "population", "pattern"],
        help="Select experiment to run",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--timesteps", type=int, default=100, help="Number of simulation steps"
    )

    parser.add_argument(
        "--record-every",
        type=int,
        default=1000,
        help="Snapshot weights every N timesteps",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    train_loader, test_loader, train_data = load_mnist(batch_size=64)

    input_layer = SpikingLayer(
        num_neurons=784, tau=10.0, v_rest=-70.0, v_th=-55.0, v_reset=-75.0
    )

    hidden_layer = SpikingLayer(
        num_neurons=64, tau=10.0, v_rest=-70.0, v_th=-55.0, v_reset=-75.0
    )

    output_layer = SpikingLayer(
        num_neurons=10, tau=10.0, v_rest=-70.0, v_th=-58.0, v_reset=-75.0
    )

    network = Network(
        layers=[input_layer, hidden_layer, output_layer], timesteps=args.timesteps
    )

    stdp01 = STDP(784, 64)

    network.create_synapse(0, 1, stdp01)
    network.create_synapse(1, 2)

    network.run(inputs=train_data, record_every=args.record_every)

    spikes = network.get_output_spikes()

    plot_raster(spikes)
    plot_firing_rates(spikes, T=args.timesteps)

    plot_weight_distributions(network.weight_snapshots, synapse_idx=0)
    plot_weight_distributions(network.weight_snapshots, synapse_idx=1)
    plot_weight_heatmaps(network.weight_snapshots, synapse_idx=0)
    plot_weight_heatmaps(network.weight_snapshots, synapse_idx=1)
    plot_firing_rate_history(network.firing_rate_history)


if __name__ == "__main__":
    main()
