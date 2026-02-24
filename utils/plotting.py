import matplotlib.pyplot as plt


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
