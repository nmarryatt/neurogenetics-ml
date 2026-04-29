from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np


DEFAULT_REST_VHDR = Path("data/ds004796/sub-01/eeg/sub-01_task-rest_eeg.vhdr")
POSTERIOR_CHANNELS = ["P3", "Pz", "P4", "O1", "Oz", "O2"]
EEG_BANDS = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 12),
    "Beta": (13, 30),
}


def load_rest_eeg(vhdr_path: str | Path = DEFAULT_REST_VHDR, *, preload: bool = True) -> mne.io.BaseRaw:
    """Load a resting-state BrainVision EEG file."""
    return mne.io.read_raw_brainvision(vhdr_path, preload=preload)


def get_rest_marker_times(raw: mne.io.BaseRaw) -> dict[str, float]:
    """Extract rest marker timings from BrainVision/MNE annotations.

    Uses documented start markers directly and falls back to S1 markers if needed.
    End markers remain inferred from event order.
    """
    descriptions = list(raw.annotations.description)
    onsets = raw.annotations.onset

    def find_onset(desc: str) -> float | None:
        return onsets[descriptions.index(desc)] if desc in descriptions else None

    s1_onsets = [onsets[i] for i, desc in enumerate(descriptions) if desc == "Stimulus/S  1"]

    eyes_open_start = find_onset("Stimulus/S  2")
    eyes_closed_start = find_onset("Stimulus/S  4")
    if eyes_open_start is None and s1_onsets:
        eyes_open_start = s1_onsets[0]
    if eyes_closed_start is None and len(s1_onsets) > 1:
        eyes_closed_start = s1_onsets[1]

    eyes_open_end = find_onset("Stimulus/S 10")
    eyes_closed_end = find_onset("Stimulus/S 11")

    if None in (eyes_open_start, eyes_open_end, eyes_closed_start, eyes_closed_end):
        raise ValueError("Could not resolve all rest condition markers from annotations.")

    return {
        "eyes_open_start": float(eyes_open_start),
        "eyes_open_end": float(eyes_open_end),
        "eyes_closed_start": float(eyes_closed_start),
        "eyes_closed_end": float(eyes_closed_end),
    }


def crop_rest_conditions(
    raw: mne.io.BaseRaw,
) -> tuple[mne.io.BaseRaw, mne.io.BaseRaw, dict[str, float]]:
    """Crop a resting EEG recording into eyes-open and eyes-closed segments."""
    markers = get_rest_marker_times(raw)
    raw_eyes_open = raw.copy().crop(
        tmin=markers["eyes_open_start"], tmax=markers["eyes_open_end"]
    )
    raw_eyes_closed = raw.copy().crop(
        tmin=markers["eyes_closed_start"], tmax=markers["eyes_closed_end"]
    )
    return raw_eyes_open, raw_eyes_closed, markers


def compute_mean_psd_db(
    raw: mne.io.BaseRaw,
    *,
    fmin: float = 1,
    fmax: float = 45,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Welch PSD and average across channels, returned in dB."""
    psd = raw.compute_psd(method="welch", fmin=fmin, fmax=fmax)
    freqs = psd.freqs
    mean_db = 10 * np.log10(psd.get_data().mean(axis=0))
    return freqs, mean_db


def add_eeg_band_annotations(ax: plt.Axes, bands: dict[str, tuple[float, float]] = EEG_BANDS) -> None:
    """Shade and label standard EEG frequency bands on an axis."""
    for label, (start, end) in bands.items():
        ax.axvspan(start, end, color="grey", alpha=0.08)
        ax.text(
            (start + end) / 2,
            ax.get_ylim()[1] - 0.5,
            label,
            ha="center",
            va="top",
            fontsize=9,
            color="dimgray",
        )


def plot_psd_comparison(
    raw_open: mne.io.BaseRaw,
    raw_closed: mne.io.BaseRaw,
    *,
    fmin: float = 1,
    fmax: float = 45,
    title: str = "rest EEG: average PSD across channels",
    save_path: str | Path | None = None,
    posterior_only: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot eyes-open vs eyes-closed PSD comparison and optionally save it."""
    if posterior_only:
        raw_open = raw_open.copy().pick(POSTERIOR_CHANNELS)
        raw_closed = raw_closed.copy().pick(POSTERIOR_CHANNELS)

    freqs_open, open_mean_db = compute_mean_psd_db(raw_open, fmin=fmin, fmax=fmax)
    freqs_closed, closed_mean_db = compute_mean_psd_db(raw_closed, fmin=fmin, fmax=fmax)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(freqs_open, open_mean_db, label="Eyes open")
    ax.plot(freqs_closed, closed_mean_db, label="Eyes closed")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power spectral density (dB)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    add_eeg_band_annotations(ax)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax
