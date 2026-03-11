import numpy as np


def poisson_input(
    num_neurons: int, firing_rate: float, dt: float, timesteps: int
) -> np.ndarray:
    """firing_rate in Hz, dt in ms"""
    prob = firing_rate * dt / 1000.0
    return np.random.rand(timesteps, num_neurons) < prob
