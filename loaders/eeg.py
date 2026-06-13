import logging
import os
import sys
import warnings

import mne
import numpy as np
from mne.preprocessing.ica import ICA
from scipy import signal

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_EOG_CHANNELS = {"Fp1": "eog", "Fp2": "eog", "Af3": "eog", "Af4": "eog"}
_MOTOR_CHANNELS = ["C3", "Cz", "C4"]
_MONTAGE = "standard_1020"
_FILTER_LOW_HZ = 8
_FILTER_HIGH_HZ = 30
_FILTER_ORDER = 4
_TARGET_EVENTS = {"left_hand": "T1", "right_hand": "T2"}
_EPOCH_TMIN = 0.0
_EPOCH_TMAX = 4.0
_EPOCH_BASELINE = (0.0, 0.5)


def _epochs_cached(output_dir: str, file_path: str) -> str | None:
    """Return the cached .fif path if it exists on disk, else None."""
    subject_id = os.path.splitext(os.path.basename(file_path))[0]
    fif_path = os.path.join(output_dir, f"{subject_id}-epo.fif")
    return fif_path if os.path.exists(fif_path) else None


def _load_raw(file_path: str) -> mne.io.Raw:
    logger.info("Loading subject file: %s", os.path.basename(file_path))
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    raw.rename_channels(lambda x: x.replace(".", ""))
    montage = mne.channels.make_standard_montage(_MONTAGE)
    raw.set_montage(montage, match_case=False, on_missing="ignore")
    raw.set_channel_types(_EOG_CHANNELS)
    return raw


def _apply_bandpass(raw: mne.io.Raw) -> mne.io.Raw:
    logger.info(
        "Applying bandpass filter: %.1f–%.1f Hz (order %d)",
        _FILTER_LOW_HZ,
        _FILTER_HIGH_HZ,
        _FILTER_ORDER,
    )
    nyquist = 0.5 * raw.info["sfreq"]
    sos = signal.butter(
        N=_FILTER_ORDER,
        Wn=[_FILTER_LOW_HZ / nyquist, _FILTER_HIGH_HZ / nyquist],
        btype="bandpass",
        output="sos",
    )
    raw._data = signal.sosfiltfilt(
        sos, raw.get_data(), axis=-1
    )  # pylint: disable=protected-access
    return raw


def _remove_eog_artifacts(raw: mne.io.Raw) -> mne.io.Raw:
    logger.info("Fitting ICA for EOG artifact removal...")
    ica = ICA(max_iter="auto", random_state=42)
    ica.fit(raw)
    eog_indices, _ = ica.find_bads_eog(raw, ch_name="Fp1", threshold=3.0)
    logger.warning("ICA flagged %d component(s) for exclusion.", len(eog_indices))
    ica.exclude = eog_indices
    ica.apply(raw)
    return raw


def _extract_epochs(
    raw: mne.io.Raw, events: np.ndarray, event_dict: dict
) -> mne.Epochs | None:
    target_event_id = {
        label: event_dict[code]
        for label, code in _TARGET_EVENTS.items()
        if code in event_dict
    }

    if not target_event_id:
        logger.error("No target motor imagery events found in annotations.")
        return None

    epochs = mne.Epochs(
        raw,
        events,
        event_id=target_event_id,
        tmin=_EPOCH_TMIN,
        tmax=_EPOCH_TMAX,
        baseline=_EPOCH_BASELINE,
        preload=True,
        verbose=False,
    )
    return epochs


def _per_epoch_scaler(psd_data):
    normalized_psd_data = np.zeros(psd_data.shape)
    for idx, epoch in enumerate(psd_data):
        min_val = np.min(epoch)
        max_val = np.max(epoch)

        if min_val == max_val:
            normalized_epoch = np.full(shape=epoch.shape, fill_value=0.5)
            normalized_psd_data[idx] = normalized_epoch
            continue

        normalized_epoch = (epoch - min_val) / (max_val - min_val)
        normalized_psd_data[idx] = normalized_epoch

    return normalized_psd_data


def load_eeg_motor_imagery(
    file_path: str,
    output_dir: str | None = None,
) -> tuple[mne.Epochs, np.ndarray, np.ndarray] | None:
    """
    Load, preprocess, and epoch a single PhysioNet Motor Imagery .edf file.

    Steps: raw loading → channel formatting → bandpass filter →
    ICA artifact removal → epoch extraction → PSD computation.

    Args:
        file_path:  Absolute path to the subject .edf file.
        output_dir: If provided, saves the epoched .fif file here.

    Returns:
        (epochs, psd_data, frequencies) on success, None if target
        events are missing. psd_data shape: (n_epochs, n_channels, n_freqs).
    """
    if output_dir is not None:
        cached = _epochs_cached(output_dir, file_path)
        if cached:
            logger.info("Loading cached epochs from: %s", cached)
            epochs = mne.read_epochs(cached, verbose=False, preload=True).pick(
                _MOTOR_CHANNELS
            )
            spectrum = epochs.compute_psd(fmin=_FILTER_LOW_HZ, fmax=_FILTER_HIGH_HZ)
            psd_data, frequencies = spectrum.get_data(return_freqs=True)
            psd_data_scaled = _per_epoch_scaler(psd_data)
            return epochs, psd_data_scaled, frequencies

    raw = _load_raw(file_path)
    events, event_dict = mne.events_from_annotations(raw, verbose=False)

    raw = _apply_bandpass(raw)
    raw = _remove_eog_artifacts(raw)

    epochs = _extract_epochs(raw, events, event_dict)

    if epochs is None:
        return None

    epochs = epochs.pick(_MOTOR_CHANNELS)

    spectrum = epochs.compute_psd(fmin=_FILTER_LOW_HZ, fmax=_FILTER_HIGH_HZ)
    psd_data, frequencies = spectrum.get_data(return_freqs=True)
    logger.info("PSD matrix shape: %s", psd_data.shape)

    psd_data_scaled = _per_epoch_scaler(psd_data)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        subject_id = os.path.splitext(os.path.basename(file_path))[0]
        out_path = os.path.join(output_dir, f"{subject_id}-epo.fif")
        epochs.save(out_path, overwrite=True)
        logger.info("Epochs saved to: %s", out_path)

    return epochs, psd_data_scaled, frequencies


if __name__ == "__main__":
    _, psd_data_scaled, _ = load_eeg_motor_imagery(sys.argv[1])
    logger.info("PSD matrix shape: %s", psd_data_scaled.shape)
    logger.info("PSD min: %s", np.min(psd_data_scaled))
    logger.info("PSD max: %s", np.max(psd_data_scaled))
