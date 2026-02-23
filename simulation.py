from lif import lif_step
import matplotlib.pyplot as plt

v = -70  # start at resting potential
v_trace = []
spikes = []

for t in range(100):  # 100 time steps
    current = 2.0  # constant input
    v, spike = lif_step(v, current, dt=1.0, tau=10.0, v_rest=-70)
    v_trace.append(v)
    spikes.append(spike)

plt.scatter(range(len(v_trace)), v_trace)
plt.show()

plt.scatter(range(len(spikes)), spikes)
plt.show()
