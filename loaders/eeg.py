import os
import warnings
import numpy as np
import mne
from mne.preprocessing.ica import ICA
import matplotlib.pyplot as plt
from scipy import signal

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
plt.rcParams["figure.dpi"] = 150

data_path = r"C:\Users\eduar\Documents\Datasets\eeg-motor-movementimagery-dataset-1.0.0\files\S001\S001R04.edf"
raw = mne.io.read_raw_edf(data_path, preload=True)

raw.rename_channels(lambda x: x.replace(".", ""))

montage = mne.channels.make_standard_montage("standard_1020")
raw.set_montage(montage, match_case=False, on_missing="ignore")

chan_types_dict = {"Fp1": "eog", "Fp2": "eog", "Af3": "eog", "Af4": "eog"}
raw.set_channel_types(chan_types_dict)

events, event_dict = mne.events_from_annotations(raw)
print(event_dict)

target_event_id = {
    "left_hand": event_dict.get("T1"),
    "right_hand": event_dict.get("T2"),
}

sfreq = raw.info["sfreq"]
nyquist = 0.5 * sfreq
low_norm = 1.0 / nyquist
high_norm = 40.0 / nyquist

sos = signal.butter(N=4, Wn=[low_norm, high_norm], btype="bandpass", output="sos")
raw_data_matrix = raw.get_data()
filtered_data = signal.sosfiltfilt(sos, x=raw_data_matrix, axis=-1)
raw._data = filtered_data  # pylint: disable=protected-access

ica = ICA(max_iter="auto", random_state=42)
ica.fit(raw)

eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name="Fp1", threshold=3.0)
print(eog_scores)

ica.exclude = eog_indices
ica.apply(raw)

plt.close("all")

if not target_event_id.get("left_hand") and not target_event_id.get("right_hand"):
    print("Missing target events")
else:
    target_event_id = {k: v for k, v in target_event_id.items() if v is not None}

    epochs = mne.Epochs(
        raw,
        events,
        event_id=target_event_id,
        tmin=0.0,
        tmax=4.0,
        baseline=(0.0, 0.5),
        preload=True,
    )

    spectrum = epochs.compute_psd(fmin=1.0, fmax=40.0)
    psd_data, frequencies = spectrum.get_data(return_freqs=True)
    print(f"SNN-ready PSD matrix shape: {psd_data.shape}")
    print(f"Frequency steps resolved: {frequencies}")

    output_dir = os.path.join(os.getcwd(), "data", "preprocessed_data")
    os.makedirs(output_dir, exist_ok=True)

    full_output_path = os.path.join(output_dir, "S001R04-epo.fif")
    epochs.save(full_output_path, overwrite=True)

    print("SUCCESS: Preprocessing complete!")
    print(f"File safely serialized to: {full_output_path}")
    print("=" * 50)
