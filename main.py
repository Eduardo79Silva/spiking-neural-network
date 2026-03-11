import argparse
import numpy as np
import random

from layers.spiking_layer import SpikingLayer
from rules.stdp import STDP
from utils.plotting import plot_raster, plot_firing_rates
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

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    input_layer = SpikingLayer(
        num_inputs=5, num_neurons=5, tau=10.0, v_rest=-70.0, v_th=-55.0, v_reset=-75.0
    )

    hidden_layer = SpikingLayer(
        num_inputs=5, num_neurons=5, tau=10.0, v_rest=-70.0, v_th=-55.0, v_reset=-75.0
    )

    output_layer = SpikingLayer(
        num_inputs=5, num_neurons=2, tau=10.0, v_rest=-70.0, v_th=-55.0, v_reset=-75.0
    )

    network = Network(layers=[input_layer, hidden_layer, output_layer])

    stdp01 = STDP(5, 5)
    stdp12 = STDP(5, 2)

    network.create_synapse(0, 1, stdp01)
    network.create_synapse(1, 2, stdp12)
    input_layer.input_current = np.dot(
        network.synapses[0].weights, np.linspace(0.0, 1.0, input_layer.num_neurons)
    )

    network.run()

    # experiment = ConstantCurrentPopulationExperiment(
    #     layer=layer, T=args.timesteps, k=5.0
    # )

    spikes = network.get_output_spikes()

    plot_raster(spikes)
    plot_firing_rates(spikes, T=args.timesteps)


if __name__ == "__main__":
    main()
