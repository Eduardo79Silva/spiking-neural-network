import numpy as np
import mne
from mne.preprocessing import ICA
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
plt.rcParams["figure.dpi"] = 150


data_path = r"C:\Users\eduar\Documents\Datasets\eeg-motor-movementimagery-dataset-1.0.0\files\S001\S001R01.edf"

raw = mne.io.read_raw_edf(data_path, preload=True)

print(raw)
print(raw.info)
print(raw.ch_names)

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
print(events[:10])
input()

raw.info["bads"].append("Fc5")

raw = raw.interpolate_bads(on_bad_position="ignore")

epochs = mne.Epochs(
    raw,
    events,
    event_id=event_dict,
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

raw.plot(duration=5, n_channels=32, clipping=None)


epochs.plot(n_epochs=4)

epochs.compute_psd().plot(picks="eeg")


bands = [(4, 8, "Theta"), (8, 12, "Alpha"), (12, 30, "Beta")]
epochs.plot_psd_topomap(bands=bands, vlim="joint")

evoked = epochs.average()
evoked.plot()

times = np.linspace(0, 2, 5)
evoked.plot_topomap(times=times, colorbar=True)

evoked.plot_joint()

evoked.plot_image()

freqs = np.logspace(*np.log10([4, 30]), num=10)
n_cycles = freqs / 2.0
power, itc = epochs.compute_tfr(
    freqs=freqs,
    n_cycles=n_cycles,
    use_fft=True,
    method="morlet",
    return_itc=True,
    average=True,
)

power.plot(picks=["O1", "Oz", "O2"], baseline=(-0.5, 0), mode="logratio", title="auto")
power.plot_topo(baseline=(-0.5, 0), mode="logratio", title="Average power")
