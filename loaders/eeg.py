import numpy as np
import mne
from mne.preprocessing.ica import ICA
import matplotlib.pyplot as plt
from scipy import signal


import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
plt.rcParams["figure.dpi"] = 150


data_path = r"C:\Users\eduar\Documents\Datasets\eeg-motor-movementimagery-dataset-1.0.0\files\S001\S001R04.edf"

raw = mne.io.read_raw_edf(data_path, preload=True)

eeg_picks = mne.pick_types(raw.info, eeg=True)
freqs = 60
raw_notch = raw.copy().notch_filter(freqs=freqs, picks=eeg_picks)
for title, data in zip(["Un", "Notch "], [raw, raw_notch]):
    fig = data.compute_psd(fmax=64).plot(
        average=True, amplitude=False, picks="data", exclude="bads"
    )
    fig.suptitle(f"{title}filtered", size="xx-large", weight="bold")


def clean_channel(name):
    name = name.replace(".", "")

    if name.endswith("Z"):
        name = name[:-1].upper() + "z"
    else:
        if len(name) >= 2 and name[1].isalpha():
            name = name[0].upper() + name[1].lower() + name[2:]
        else:
            name = name.upper()

    return name


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

raw._data = filtered_data

w, h = signal.sosfreqz(sos, worN=2000)


frequencies = w * (sfreq / (2.0 * np.pi))


magnitude_db = 20 * np.log10(np.abs(h))


plt.figure(figsize=(10, 5))
plt.plot(frequencies, magnitude_db, label="Filter Response", color="crimson", lw=2)


plt.axvline(1, color="gray", linestyle="--", label="Low Cutoff (1 Hz)")
plt.axvline(40, color="gray", linestyle="--", label="High Cutoff (40 Hz)")


plt.xlim(0, 60)
plt.ylim(-60, 5)

plt.title("Frequency Response of 4th-Order 1-40Hz Butterworth Filter", weight="bold")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Gain (dB)")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.show()


input()

if not target_event_id.get("left_hand") and not target_event_id.get("right_hand"):
    print("Missing target events")
else:
    epochs = mne.Epochs(
        raw,
        events,
        event_id=target_event_id,
        tmin=0,
        tmax=4,
        baseline=None,
        preload=True,
    )

    ica = ICA(max_iter="auto")
    raw_for_ica = raw.copy().filter(l_freq=1, h_freq=None)
    ica.fit(raw_for_ica)

    ica.exclude = [1]

    ica.apply(raw)
