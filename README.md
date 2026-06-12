# Spiking Neural Network Framework

A biologically-inspired spiking neural network (SNN) simulator built from scratch in Python, implementing the **Leaky Integrate-and-Fire (LIF)** neuron model. Designed for EEG signal classification and motor imagery decoding — part of an ongoing project targeting brain-computer interface (BCI) applications.

## Overview

This framework provides a modular, extensible foundation for simulating spiking neural networks. Unlike conventional artificial neural networks, SNNs communicate via discrete spike events and operate over continuous time — making them both more biologically plausible and naturally suited for processing temporal signals such as EEG.

**Current scope:** LIF neuron dynamics, population-level simulations, raster visualization, and firing rate analysis.

**Planned:** Adaptive Exponential Integrate-and-Fire (AdEx) neurons, Spike-Timing-Dependent Plasticity (STDP), and integration with the [PhysioNet EEG Motor Imagery dataset](https://physionet.org/content/eegmmidb/1.0.0/) for motor imagery classification.

## Getting Started

### Requirements

```bash
pip install numpy matplotlib
```

### Run a simulation

```bash
python main.py --experiment population --timesteps 100 --seed 42
```

**Arguments:**

| Flag | Options | Default | Description |
|------|---------|---------|-------------|
| `--experiment` | `single`, `population`, `pattern` | — | Experiment type |
| `--timesteps` | int | `100` | Number of simulation time steps |
| `--seed` | int | `42` | Random seed for reproducibility |

### Output

Running the population experiment produces:
- **Raster plot** — spike times per neuron across the simulation window
- **Firing rate plot** — mean spike rate per neuron over time

## Experiment: Constant Current Population

`ConstantCurrentPopulationExperiment` drives a population of LIF neurons with linearly spaced constant currents (from 0.0 to 1.0), allowing observation of the input–output gain function across the population. Neurons receiving stronger input cross threshold more frequently, producing higher firing rates — the fundamental f-I curve relationship.

```python
layer = SpikingLayer(
    num_inputs=5, num_neurons=5,
    tau=10.0, v_rest=-70.0, v_th=-55.0, v_reset=-75.0
)
experiment = ConstantCurrentPopulationExperiment(layer=layer, T=100, k=5.0)
spikes = experiment.run()  # shape: (num_neurons, T)
```

## References

- Dayan, P. & Abbott, L.F. (2001). *Theoretical Neuroscience*. MIT Press.
- Gerstner, W. et al. (2014). *Neuronal Dynamics*. Cambridge University Press. [Free online](https://neuronaldynamics.epfl.ch/)
- PhysioNet EEG Motor Imagery Database: https://physionet.org/content/eegmmidb/1.0.0/
- MNE-Python EEG analysis library: https://mne.tools

## License

MIT
