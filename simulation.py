from lif import lif_step
import matplotlib.pyplot as plt

inputs = [0.0, 0.2, 0.5, 1.0, 0.8]

N = len(inputs)
k = 5
T = 50

# One voltage per neuron
v = [-70.0] * N

# Store spike trains separately per neuron
spikes = [[] for _ in range(N)]

for t in range(T):
    for n in range(N):
        current = k * inputs[n]
        v[n], spike = lif_step(v[n], current, dt=1.0, tau=10.0, v_rest=-70)
        spikes[n].append(spike)


for n in range(N):
    spike_times = [t for t, s in enumerate(spikes[n]) if s == 1]
    plt.vlines(spike_times, n - 0.4, n + 0.4)

plt.xlabel("Time")
plt.ylabel("Neuron index")
plt.title("Spike Raster Plot")
plt.ylim(-1, N)
plt.show()

rates = [sum(spikes[n]) / T for n in range(N)]

plt.bar(range(N), rates)
plt.xlabel("Neuron index")
plt.ylabel("Firing rate")
plt.title("Firing Rate per Neuron")
plt.show()
