import matplotlib.pyplot as plt
import numpy as np


def plot_raster(spikes):
    N = len(spikes)

    for n in range(N):
        spike_times = [t for t, s in enumerate(spikes[n]) if s == 1]
        plt.vlines(spike_times, n - 0.4, n + 0.4)

    plt.xlabel("Time")
    plt.ylabel("Neuron index")
    plt.title("Spike Raster Plot")
    plt.ylim(-1, N)
    plt.show()


def plot_firing_rates(spikes, T):
    rates = [sum(neuron_spikes) / T for neuron_spikes in spikes]

    plt.bar(range(len(spikes)), rates)
    plt.xlabel("Neuron index")
    plt.ylabel("Firing rate")
    plt.title("Firing Rate per Neuron")
    plt.show()


def plot_weight_distributions(
    weight_snapshots: dict, synapse_idx: int = 0, w_min: float = 0.0, w_max: float = 1.0
):
    """Histogram of weight distribution at each recorded timestep for a given synapse."""
    timesteps = sorted(weight_snapshots.keys())
    n_plots = len(timesteps)

    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4), sharey=True)
    if n_plots == 1:
        axes = [axes]

    for ax, t in zip(axes, timesteps):
        weights = weight_snapshots[t][synapse_idx].flatten()
        ax.hist(
            weights, bins=20, range=(w_min, w_max), color="steelblue", edgecolor="white"
        )
        ax.set_title(f"t={t}")
        ax.set_xlabel("Weight")
        ax.set_xlim(w_min, w_max)

    axes[0].set_ylabel("Count")
    fig.suptitle(f"Weight Distribution Over Time (Synapse {synapse_idx})")
    plt.tight_layout()
    plt.show()


def plot_weight_heatmaps(
    weight_snapshots: dict, synapse_idx: int = 0, w_min: float = 0.0, w_max: float = 1.0
):
    """Heatmap of the weight matrix at each recorded timestep for a given synapse."""
    timesteps = sorted(weight_snapshots.keys())
    n_plots = len(timesteps)

    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    for ax, t in zip(axes, timesteps):
        weights = weight_snapshots[t][synapse_idx]
        im = ax.imshow(weights, vmin=w_min, vmax=w_max, cmap="viridis", aspect="auto")
        ax.set_title(f"t={t}")
        ax.set_xlabel("Pre neuron")
        ax.set_ylabel("Post neuron")

    fig.colorbar(im, ax=axes[-1], label="Weight")
    fig.suptitle(f"Weight Matrix Over Time (Synapse {synapse_idx})")
    plt.tight_layout()
    plt.show()


def plot_firing_rate_history(firing_rate_history: list):
    """Average firing rate per layer over time."""
    history = np.array(firing_rate_history)  # shape: (timesteps, num_layers)
    n_layers = history.shape[1]

    plt.figure(figsize=(10, 4))
    for layer_idx in range(n_layers):
        plt.plot(history[:, layer_idx], label=f"Layer {layer_idx}")

    plt.xlabel("Timestep")
    plt.ylabel("Mean firing rate")
    plt.title("Average Firing Rate per Layer Over Time")
    plt.legend()
    plt.tight_layout()
    plt.show()
