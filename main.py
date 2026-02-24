from layers.spiking_layer import SpikingLayer
from experiments.constant_current_population import ConstantCurrentPopulationExperiment
from utils.plotting import plot_raster, plot_firing_rates


def main():
    layer = SpikingLayer(
        num_neurons=5, tau=10.0, v_rest=-70.0, v_th=-55.0, v_reset=-75.0
    )

    experiment = ConstantCurrentPopulationExperiment(layer=layer, T=50, k=5.0)

    spikes = experiment.run()

    plot_raster(spikes)
    plot_firing_rates(spikes, T=50)


if __name__ == "__main__":
    main()
