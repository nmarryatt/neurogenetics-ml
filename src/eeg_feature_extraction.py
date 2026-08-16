from __future__ import annotations

from importlib import util
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.stats import kurtosis, skew

from src import eeg_preprocessing


EEG_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma_low": (30.0, 40.0),
}

CONNECTIVITY_BANDS = {
    "theta": EEG_BANDS["theta"],
    "alpha": EEG_BANDS["alpha"],
    "beta": EEG_BANDS["beta"],
    "gamma_low": EEG_BANDS["gamma_low"],
}

REGIONS = {
    "frontal": ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"],
    "central": ["C3", "C4", "Cz"],
    "temporal": ["T7", "T8", "P7", "P8"],
    "parietal": ["P3", "P4", "Pz"],
    "occipital": ["O1", "O2", "Oz"],
}

PSD_FMIN = 1.0
PSD_FMAX = 40.0
WELCH_WINDOW_S = 2.0


def dependency_table() -> pd.DataFrame:
    """Return availability of required and optional feature-extraction packages."""
    packages = {
        "mne": "core EEG I/O and spectral estimation",
        "numpy": "numerical arrays",
        "pandas": "feature tables",
        "scipy": "signal summaries and integration",
        "mne_connectivity": "connectivity matrices",
        "specparam": "aperiodic / 1/f spectral parameterisation",
        "antropy": "entropy and fractal-complexity features",
        "pycrostates": "microstate clustering and backfitting",
    }
    return pd.DataFrame(
        [
            {
                "package": package,
                "available": util.find_spec(package) is not None,
                "purpose": purpose,
            }
            for package, purpose in packages.items()
        ]
    )


def feature_family_table() -> pd.DataFrame:
    """Return a plain-language summary of the feature families in this notebook."""
    return pd.DataFrame(
        [
            {
                "family": "Spectral power",
                "what_it_summarises": "How EEG power is distributed across frequency bands.",
                "main_metrics": "absolute power, relative power, regional power, peak alpha frequency, theta/beta ratio",
                "interpretation": "Higher values mean stronger activity in that band or region; relative power controls for total 1-40 Hz power.",
            },
            {
                "family": "Aperiodic / 1/f",
                "what_it_summarises": "The broadband background shape of the power spectrum, separated from narrow oscillatory peaks.",
                "main_metrics": "aperiodic offset, aperiodic exponent, oscillatory peak frequency/power/bandwidth",
                "interpretation": "The exponent reflects spectral slope; peak metrics describe rhythmic components after modelling the background.",
            },
            {
                "family": "Time-domain",
                "what_it_summarises": "Basic properties of the cleaned EEG waveform.",
                "main_metrics": "standard deviation, peak-to-peak amplitude, line length, skew, kurtosis, Hjorth parameters",
                "interpretation": "Useful signal summaries and QC-adjacent features; large amplitudes can reflect neural signal or residual artifact.",
            },
            {
                "family": "Connectivity",
                "what_it_summarises": "Synchronisation between EEG channels within each frequency band.",
                "main_metrics": "mean wPLI, standard deviation of wPLI",
                "interpretation": "wPLI reduces zero-lag volume-conduction effects; with 2-second epochs this notebook estimates theta and above, not delta.",
            },
            {
                "family": "Graph theory",
                "what_it_summarises": "Network organisation after thresholding the connectivity matrix.",
                "main_metrics": "density, mean strength, mean clustering",
                "interpretation": "These compress pairwise connectivity into network-level summaries.",
            },
            {
                "family": "Complexity",
                "what_it_summarises": "Irregularity and fractal structure of the resting EEG signal.",
                "main_metrics": "permutation entropy, spectral entropy, Higuchi fractal dimension",
                "interpretation": "Higher entropy generally means less predictable signal structure.",
            },
            {
                "family": "Microstates",
                "what_it_summarises": "Short-lived, recurring scalp topographies and their transitions.",
                "main_metrics": "duration, occurrence, coverage, transition probabilities, global explained variance",
                "interpretation": "Fit templates on training subjects only, then backfit held-out subjects to avoid leakage.",
            },
        ]
    )


def parse_epoch_path(path: Path) -> dict[str, str | Path]:
    """Parse subject id and rest condition from a cleaned epoch path."""
    subject_id = eeg_preprocessing.subject_id_from_vhdr(path)
    if "eyes_open" in path.name:
        condition = "eyes_open"
    elif "eyes_closed" in path.name:
        condition = "eyes_closed"
    else:
        condition = "unknown"
    return {"subject_id": subject_id, "condition": condition, "path": path}


def find_clean_epoch_files(processed_eeg_dir: str | Path) -> pd.DataFrame:
    """Find cleaned resting-state epoch files saved by the preprocessing notebook."""
    processed_eeg_dir = Path(processed_eeg_dir)
    epoch_paths = sorted(processed_eeg_dir.glob("sub-*/eeg/*_clean-epo.fif"))
    return pd.DataFrame([parse_epoch_path(path) for path in epoch_paths])


def load_epochs(path: str | Path) -> mne.Epochs:
    """Load one cleaned Epochs FIF file."""
    return mne.read_epochs(path, preload=True, verbose=False)


def summarize_epoch_files(epoch_index: pd.DataFrame) -> pd.DataFrame:
    """Summarize available cleaned EEG per subject and condition."""
    rows = []
    for _, row in epoch_index.iterrows():
        epochs = load_epochs(row["path"])
        eeg_epochs = epochs.copy().pick("eeg", exclude="bads")
        rows.append(
            {
                "subject_id": row["subject_id"],
                "condition": row["condition"],
                "n_epochs": len(epochs),
                "n_channels": len(eeg_epochs.ch_names),
                "sfreq": float(epochs.info["sfreq"]),
                "duration_minutes": len(epochs) * (epochs.tmax - epochs.tmin) / 60,
            }
        )
    return pd.DataFrame(rows)


def available_channels(epochs: mne.Epochs, requested: list[str]) -> list[str]:
    """Return requested channels present in the epochs object."""
    return [ch for ch in requested if ch in epochs.ch_names]


def bandpower_from_psd(
    psd: np.ndarray,
    freqs: np.ndarray,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """Integrate PSD values over a frequency band."""
    mask = (freqs >= fmin) & (freqs < fmax)
    if not np.any(mask):
        return np.full(psd.shape[:-1], np.nan)
    return simpson(psd[..., mask], x=freqs[mask], axis=-1)


def compute_spectral_features(epochs: mne.Epochs) -> dict[str, float]:
    """Compute global and regional bandpower features from cleaned epochs."""
    n_per_seg = int(round(epochs.info["sfreq"] * WELCH_WINDOW_S))
    spectrum = epochs.compute_psd(
        method="welch",
        fmin=PSD_FMIN,
        fmax=PSD_FMAX,
        picks="eeg",
        exclude="bads",
        n_per_seg=n_per_seg,
        n_fft=n_per_seg,
    )
    psd = spectrum.get_data()
    freqs = spectrum.freqs
    ch_names = spectrum.ch_names
    total_power = bandpower_from_psd(psd, freqs, PSD_FMIN, PSD_FMAX)

    features = {}
    for band_name, (fmin, fmax) in EEG_BANDS.items():
        bp = bandpower_from_psd(psd, freqs, fmin, fmax)
        features[f"power_abs_{band_name}_global"] = float(np.nanmean(bp))
        features[f"power_rel_{band_name}_global"] = float(np.nanmean(bp / total_power))

        for region_name, region_channels in REGIONS.items():
            picks = [
                ch_names.index(ch)
                for ch in available_channels(epochs, region_channels)
                if ch in ch_names
            ]
            if picks:
                features[f"power_abs_{band_name}_{region_name}"] = float(
                    np.nanmean(bp[:, picks])
                )
                features[f"power_rel_{band_name}_{region_name}"] = float(
                    np.nanmean((bp / total_power)[:, picks])
                )

    posterior_channels = available_channels(
        epochs,
        REGIONS["parietal"] + REGIONS["occipital"],
    )
    posterior_picks = [ch_names.index(ch) for ch in posterior_channels if ch in ch_names]
    alpha_mask = (freqs >= EEG_BANDS["alpha"][0]) & (freqs < EEG_BANDS["alpha"][1])
    if posterior_picks and np.any(alpha_mask):
        posterior_alpha = psd[:, posterior_picks][:, :, alpha_mask].mean(axis=(0, 1))
        features["peak_alpha_frequency_posterior"] = float(
            freqs[alpha_mask][np.argmax(posterior_alpha)]
        )

    if all(
        key in features
        for key in ["power_abs_theta_global", "power_abs_beta_global"]
    ):
        features["theta_beta_ratio_global"] = (
            features["power_abs_theta_global"] / features["power_abs_beta_global"]
        )

    return features


def compute_time_domain_features(epochs: mne.Epochs) -> dict[str, float]:
    """Compute simple waveform and Hjorth features."""
    data = epochs.copy().pick("eeg", exclude="bads").get_data() * 1e6
    diff = np.diff(data, axis=-1)
    variance = np.var(data, axis=-1)
    diff_variance = np.var(diff, axis=-1)
    mobility = np.sqrt(diff_variance / variance)
    complexity = np.sqrt(np.var(np.diff(diff, axis=-1), axis=-1) / diff_variance)
    complexity = complexity / mobility

    return {
        "td_mean_uv": float(np.mean(data)),
        "td_sd_uv": float(np.std(data)),
        "td_peak_to_peak_uv": float(np.mean(np.ptp(data, axis=-1))),
        "td_line_length_uv": float(np.mean(np.sum(np.abs(diff), axis=-1))),
        "td_skew": float(skew(data.reshape(-1), nan_policy="omit")),
        "td_kurtosis": float(kurtosis(data.reshape(-1), nan_policy="omit")),
        "hjorth_activity": float(np.nanmean(variance)),
        "hjorth_mobility": float(np.nanmean(mobility)),
        "hjorth_complexity": float(np.nanmean(complexity)),
    }


def compute_aperiodic_features(epochs: mne.Epochs) -> dict[str, float]:
    """Compute aperiodic and oscillatory peak features with specparam."""
    if util.find_spec("specparam") is None:
        return {}

    from specparam import SpectralModel

    n_per_seg = int(round(epochs.info["sfreq"] * WELCH_WINDOW_S))
    spectrum = epochs.compute_psd(
        method="welch",
        fmin=PSD_FMIN,
        fmax=PSD_FMAX,
        picks="eeg",
        exclude="bads",
        n_per_seg=n_per_seg,
        n_fft=n_per_seg,
    )
    freqs = spectrum.freqs
    mean_psd = spectrum.get_data().mean(axis=(0, 1))

    model = SpectralModel(peak_width_limits=(1.0, 8.0), max_n_peaks=6, verbose=False)
    model.fit(freqs, mean_psd, [PSD_FMIN, PSD_FMAX])

    if hasattr(model, "aperiodic_params_"):
        aperiodic_params = np.asarray(model.aperiodic_params_)
        peak_params = np.asarray(model.peak_params_)
    else:
        param_dict = model.results.params.asdict()
        aperiodic_params = np.asarray(
            param_dict.get("aperiodic_fit", model.get_params("aperiodic"))
        )
        peak_params = np.asarray(param_dict.get("peak_fit", np.empty((0, 3))))
        if peak_params.ndim == 0:
            peak_params = np.empty((0, 3))

    features = {
        "aperiodic_offset": float(aperiodic_params[0]),
        "aperiodic_exponent": float(aperiodic_params[1]),
        "n_oscillatory_peaks": int(len(peak_params)),
    }
    if len(peak_params) > 0:
        strongest_peak = peak_params[np.argmax(peak_params[:, 1])]
        features.update(
            {
                "strongest_peak_frequency": float(strongest_peak[0]),
                "strongest_peak_power": float(strongest_peak[1]),
                "strongest_peak_bandwidth": float(strongest_peak[2]),
            }
        )
    return features


def compute_connectivity_graph_features(
    epochs: mne.Epochs,
    method: str = "wpli",
) -> dict[str, float]:
    """Compute band-wise connectivity and simple graph summaries."""
    if util.find_spec("mne_connectivity") is None:
        return {}

    from mne_connectivity import spectral_connectivity_epochs

    eeg_epochs = epochs.copy().pick("eeg", exclude="bads")
    features = {}
    for band_name, (fmin, fmax) in CONNECTIVITY_BANDS.items():
        con = spectral_connectivity_epochs(
            eeg_epochs,
            method=method,
            mode="multitaper",
            sfreq=epochs.info["sfreq"],
            fmin=fmin,
            fmax=fmax,
            faverage=True,
            verbose=False,
        )
        matrix = con.get_data(output="dense")[:, :, 0]
        matrix = np.maximum(matrix, matrix.T)
        np.fill_diagonal(matrix, 0.0)
        upper = matrix[np.triu_indices_from(matrix, k=1)]
        features[f"connectivity_{method}_{band_name}_mean"] = float(np.nanmean(upper))
        features[f"connectivity_{method}_{band_name}_sd"] = float(np.nanstd(upper))

        threshold = np.nanpercentile(upper, 75)
        adjacency = (matrix >= threshold).astype(float)
        np.fill_diagonal(adjacency, 0.0)
        weighted_adjacency = np.where(adjacency, matrix, 0.0)

        n_channels = matrix.shape[0]
        possible_edges = n_channels * (n_channels - 1)
        density = adjacency.sum() / possible_edges if possible_edges else np.nan
        strength = weighted_adjacency.sum(axis=1)
        degree = adjacency.sum(axis=1)
        triangles = np.diag(adjacency @ adjacency @ adjacency) / 2
        denominator = degree * (degree - 1)
        clustering = np.divide(
            2 * triangles,
            denominator,
            out=np.full_like(triangles, np.nan, dtype=float),
            where=denominator > 0,
        )

        features[f"graph_{method}_{band_name}_density"] = float(density)
        features[f"graph_{method}_{band_name}_strength_mean"] = float(np.mean(strength))
        features[f"graph_{method}_{band_name}_clustering_mean"] = float(
            np.nanmean(clustering)
        )
    return features


def compute_complexity_features(
    epochs: mne.Epochs,
    max_channels: int = 10,
) -> dict[str, float]:
    """Compute lightweight entropy and fractal features."""
    if util.find_spec("antropy") is None:
        return {}

    import antropy as ant

    data = epochs.copy().pick("eeg", exclude="bads").get_data()
    channel_data = data.mean(axis=0)[:max_channels]
    return {
        "complexity_perm_entropy_mean": float(
            np.mean([ant.perm_entropy(x, normalize=True) for x in channel_data])
        ),
        "complexity_spectral_entropy_mean": float(
            np.mean(
                [
                    ant.spectral_entropy(x, sf=epochs.info["sfreq"], normalize=True)
                    for x in channel_data
                ]
            )
        ),
        "complexity_higuchi_fd_mean": float(
            np.mean([ant.higuchi_fd(x) for x in channel_data])
        ),
    }


def compute_feature_groups(epochs: mne.Epochs) -> dict[str, dict[str, float]]:
    """Compute feature families separately for easier notebook inspection."""
    return {
        "spectral": compute_spectral_features(epochs),
        "time_domain": compute_time_domain_features(epochs),
        "aperiodic": compute_aperiodic_features(epochs),
        "complexity": compute_complexity_features(epochs),
        "connectivity_graph": compute_connectivity_graph_features(epochs, method="wpli"),
    }


def extract_epoch_file_features(row: pd.Series) -> dict[str, float | int | str]:
    """Extract all features for one row from the epoch index."""
    epochs = load_epochs(row["path"])
    eeg_epochs = epochs.copy().pick("eeg", exclude="bads")
    base = {
        "subject_id": row["subject_id"],
        "condition": row["condition"],
        "n_epochs": len(epochs),
        "n_channels": len(eeg_epochs.ch_names),
        "sfreq": float(epochs.info["sfreq"]),
    }
    features = {}
    for group_features in compute_feature_groups(epochs).values():
        features.update(group_features)
    return {**base, **features}


def extract_all_features(epoch_index: pd.DataFrame) -> pd.DataFrame:
    """Extract features for each cleaned epoch file."""
    feature_rows = []
    for _, row in epoch_index.iterrows():
        print(f"Extracting {row['subject_id']} {row['condition']}")
        feature_rows.append(extract_epoch_file_features(row))
    return pd.DataFrame(feature_rows)


def save_features(features: pd.DataFrame, features_dir: str | Path) -> Path:
    """Save the feature table as a TSV file."""
    features_dir = Path(features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)
    feature_path = features_dir / "rest_eeg_features.tsv"
    features.to_csv(feature_path, sep="\t", index=False)
    return feature_path


def microstate_note() -> str:
    """Return the current microstate implementation note."""
    if util.find_spec("pycrostates") is None:
        return "Install pycrostates before running microstate extraction."
    return (
        "Microstate extraction should be added after defining train/test or CV folds. "
        "Fit templates on training subjects only, then backfit held-out subjects to avoid leakage."
    )
