import logging

import numpy as np
import torch
from sklearn import model_selection
from torch.utils.data import TensorDataset

from layers.spiking_layer import SpikingLayer
from loaders.eeg import load_eeg_motor_imagery
from network import Network
from rules.stdp import STDP
from utils.plotting import (
    plot_firing_rate_history,
    plot_firing_rates,
    plot_raster,
    plot_weight_distributions,
    plot_weight_heatmaps,
)

EEG_DATASET_PATH = "./data/eeg"
OUTPUT_DIR = "./data/loaded_eeg"
AMOUNT_OF_SUBJECTS = 10

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_eeg():
    psd_data_all_subjects, labels_all_subjects = [], []

    for subject_idx in range(1, AMOUNT_OF_SUBJECTS + 1):
        subject_id = f"S{subject_idx:03d}"

        for reading_idx in [4, 8, 12]:
            reading_id = f"{subject_id}R{reading_idx:02d}"

            loaded_eeg = load_eeg_motor_imagery(
                f"{EEG_DATASET_PATH}/{subject_id}/{reading_id}.edf", OUTPUT_DIR
            )

            if loaded_eeg is None:
                continue

            epochs, psd_data, _ = loaded_eeg
            psd_data_all_subjects.append(psd_data)
            event_labels = epochs.events[:, 2]
            labels_all_subjects.append(event_labels)

    psd_data_all_subjects = np.concatenate(psd_data_all_subjects)
    labels_all_subjects = np.concatenate(labels_all_subjects)
    labels_all_subjects -= np.min(labels_all_subjects)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        psd_data_all_subjects, labels_all_subjects, test_size=0.2, random_state=42
    )

    X_train = torch.from_numpy(X_train).float().unsqueeze(1)
    X_test = torch.from_numpy(X_test).float().unsqueeze(1)
    y_train = torch.from_numpy(y_train).float().unsqueeze(1)
    y_test = torch.from_numpy(y_test).float().unsqueeze(1)

    train_dataset = TensorDataset(X_train, y_train)

    logger.info("Dataset shape: %s", train_dataset.tensors[0][0].shape)

    input_layer = SpikingLayer(
        num_neurons=264, tau=10.0, v_rest=-70.0, v_th=-55.0, v_reset=-75.0
    )

    hidden_layer = SpikingLayer(
        num_neurons=64, tau=10.0, v_rest=-70.0, v_th=-55.0, v_reset=-75.0
    )

    output_layer = SpikingLayer(
        num_neurons=2, tau=10.0, v_rest=-70.0, v_th=-58.0, v_reset=-75.0
    )

    network = Network(layers=[input_layer, hidden_layer, output_layer], timesteps=1000)

    stdp01 = STDP(264, 64)

    network.create_synapse(0, 1, stdp01)
    network.create_synapse(1, 2)

    network.run(inputs=train_dataset, record_every=1000)

    spikes = network.get_output_spikes()

    logger.info("Spikes shape: %s", spikes.shape)


if __name__ == "__main__":
    run_eeg()
