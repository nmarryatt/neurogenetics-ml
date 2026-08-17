from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mne
import pandas as pd
from mne.preprocessing import ICA
from pyprep.prep_pipeline import PrepPipeline

from src.eeg_utils import crop_rest_conditions, load_rest_eeg


Condition = Literal["eyes_open", "eyes_closed"]
REST_CONDITIONS: tuple[Condition, Condition] = ("eyes_open", "eyes_closed")


DEFAULT_PREP_PARAMS = {
    "ref_chs": "eeg",
    "reref_chs": "eeg",
    "line_freqs": [],
    "max_iterations": 4,
}

DEFAULT_ARTIFACT_THRESHOLDS = {
    "eye blink": 0.85,
    "muscle artifact": 0.85,
    "heart beat": 0.85,
    "line noise": 0.85,
    "channel noise": 0.85,
}


def _to_str_list(values) -> list[str]:
    """Convert channel-like values, including numpy strings, to plain Python strings."""
    return [str(value) for value in values]


@dataclass
class RestPreprocessingResult:
    """Outputs and QC summaries from one subject's resting-state EEG preprocessing."""

    subject_id: str
    vhdr_path: Path
    markers: dict[str, float]
    prep_qc: pd.DataFrame
    ica_qc: pd.DataFrame
    epoch_qc: pd.DataFrame
    epochs: dict[Condition, mne.Epochs]
    raw_clean: dict[Condition, mne.io.BaseRaw]
    icas: dict[Condition, ICA]


def find_rest_vhdrs(data_root: str | Path = "data/ds004796") -> list[Path]:
    """Find downloaded BIDS resting-state BrainVision header files."""
    return sorted(Path(data_root).glob("sub-*/eeg/sub-*_task-rest_eeg.vhdr"))


def subject_id_from_vhdr(vhdr_path: str | Path) -> str:
    """Extract a BIDS subject id such as 'sub-01' from a rest EEG path."""
    path = Path(vhdr_path)
    for part in path.parts:
        if part.startswith("sub-"):
            return part
    return path.name.split("_")[0]


def subject_output_dir(output_dir: str | Path, subject_id: str) -> Path:
    """Return the processed EEG output directory for one subject."""
    return Path(output_dir) / subject_id / "eeg"


def expected_subject_outputs(output_dir: str | Path, subject_id: str) -> list[Path]:
    """Return the files expected when one subject has finished preprocessing."""
    out_dir = subject_output_dir(output_dir, subject_id)
    return [
        out_dir / f"{subject_id}_task-rest_{condition}_clean-epo.fif"
        for condition in REST_CONDITIONS
    ] + [
        out_dir / f"{subject_id}_task-rest_prep_qc.tsv",
        out_dir / f"{subject_id}_task-rest_ica_qc.tsv",
        out_dir / f"{subject_id}_task-rest_epoch_qc.tsv",
    ]


def subject_preprocessing_complete(output_dir: str | Path, subject_id: str) -> bool:
    """Return True if all expected saved outputs exist for one subject."""
    return all(path.exists() for path in expected_subject_outputs(output_dir, subject_id))


def get_completed_subject_ids(output_dir: str | Path = "data/processed/eeg") -> list[str]:
    """Find subjects with complete saved preprocessing outputs."""
    output_dir = Path(output_dir)
    subject_ids = sorted(path.name for path in output_dir.glob("sub-*") if path.is_dir())
    return [
        subject_id
        for subject_id in subject_ids
        if subject_preprocessing_complete(output_dir, subject_id)
    ]


def _read_subject_qc(output_dir: str | Path, subject_id: str, qc_name: str) -> pd.DataFrame | None:
    """Read one per-subject QC file if it exists."""
    qc_path = subject_output_dir(output_dir, subject_id) / f"{subject_id}_task-rest_{qc_name}_qc.tsv"
    if not qc_path.exists():
        return None
    return pd.read_csv(qc_path, sep="\t")


def rebuild_rest_preprocessing_qc(
    output_dir: str | Path = "data/processed/eeg",
    *,
    data_root: str | Path | None = "data/ds004796",
) -> dict[str, pd.DataFrame]:
    """Rebuild aggregate preprocessing QC tables from saved per-subject outputs.

    This is useful after a kernel restart because it does not depend on in-memory
    variables from the original batch run.
    """
    output_dir = Path(output_dir)
    if data_root is not None:
        expected_subject_ids = [
            subject_id_from_vhdr(path) for path in find_rest_vhdrs(data_root)
        ]
    else:
        expected_subject_ids = sorted(
            path.name for path in output_dir.glob("sub-*") if path.is_dir()
        )

    run_rows = []
    qc_tables: dict[str, list[pd.DataFrame]] = {
        "prep": [],
        "ica": [],
        "epoch": [],
    }

    for subject_id in expected_subject_ids:
        complete = subject_preprocessing_complete(output_dir, subject_id)
        out_dir = subject_output_dir(output_dir, subject_id)
        any_outputs = out_dir.exists() and any(out_dir.glob("*"))
        if complete:
            status = "ok"
            error = ""
        elif any_outputs:
            status = "partial"
            missing = [
                path.name
                for path in expected_subject_outputs(output_dir, subject_id)
                if not path.exists()
            ]
            error = "Missing outputs: " + ", ".join(missing)
        else:
            status = "missing"
            error = ""
        run_rows.append({"subject_id": subject_id, "status": status, "error": error})

        for qc_name in qc_tables:
            qc = _read_subject_qc(output_dir, subject_id, qc_name)
            if qc is not None:
                qc_tables[qc_name].append(qc)

    run_qc = pd.DataFrame(run_rows, columns=["subject_id", "status", "error"])
    prep_qc = (
        pd.concat(qc_tables["prep"], ignore_index=True)
        if qc_tables["prep"]
        else pd.DataFrame()
    )
    ica_qc = (
        pd.concat(qc_tables["ica"], ignore_index=True)
        if qc_tables["ica"]
        else pd.DataFrame()
    )
    epoch_qc = (
        pd.concat(qc_tables["epoch"], ignore_index=True)
        if qc_tables["epoch"]
        else pd.DataFrame()
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    run_qc.to_csv(output_dir / "rest_preprocessing_run_qc.tsv", sep="\t", index=False)
    prep_qc.to_csv(output_dir / "rest_preprocessing_prep_qc.tsv", sep="\t", index=False)
    ica_qc.to_csv(output_dir / "rest_preprocessing_ica_qc.tsv", sep="\t", index=False)
    epoch_qc.to_csv(output_dir / "rest_preprocessing_epoch_qc.tsv", sep="\t", index=False)

    return {
        "run_qc": run_qc,
        "prep_qc": prep_qc,
        "ica_qc": ica_qc,
        "epoch_qc": epoch_qc,
    }


def parse_saved_list(value) -> list:
    """Parse list-like values stored as strings in QC TSV files."""
    if isinstance(value, list):
        return value
    if pd.isna(value) or value == "":
        return []
    try:
        parsed = ast.literal_eval(str(value))
        return parsed if isinstance(parsed, list) else [parsed]
    except (ValueError, SyntaxError):
        return [str(value)]


def make_qc_review_tables(
    run_qc: pd.DataFrame,
    prep_qc: pd.DataFrame,
    ica_qc: pd.DataFrame,
    epoch_qc: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build review-friendly preprocessing QC tables."""
    status_summary = (
        run_qc["status"]
        .value_counts(dropna=False)
        .rename_axis("status")
        .to_frame("n_subjects")
    )
    problem_subjects = run_qc.loc[run_qc["status"] != "ok"].copy()

    prep_review = prep_qc.copy()
    prep_review["still_noisy_channels_list"] = prep_review[
        "still_noisy_channels"
    ].apply(parse_saved_list)
    prep_review["interpolated_channels_list"] = prep_review[
        "interpolated_channels"
    ].apply(parse_saved_list)

    prep_most_noisy = prep_review.sort_values(
        ["n_still_noisy_channels", "n_interpolated_channels"],
        ascending=False,
    )[
        [
            "subject_id",
            "condition",
            "n_still_noisy_channels",
            "still_noisy_channels",
            "n_interpolated_channels",
            "interpolated_channels",
        ]
    ]

    prep_still_noisy_channels_long = (
        prep_review[["subject_id", "condition", "still_noisy_channels_list"]]
        .explode("still_noisy_channels_list")
        .dropna(subset=["still_noisy_channels_list"])
        .rename(columns={"still_noisy_channels_list": "channel"})
    )
    prep_still_noisy_channels_long = prep_still_noisy_channels_long.loc[
        prep_still_noisy_channels_long["channel"] != ""
    ]

    prep_still_noisy_channel_frequency = (
        prep_still_noisy_channels_long["channel"]
        .value_counts()
        .rename_axis("channel")
        .to_frame("n_subject_conditions")
    )

    ica_compact = ica_qc.drop(columns=["icalabel_summary"], errors="ignore").copy()
    ica_most_removed = ica_compact.sort_values(
        "n_excluded_components",
        ascending=False,
    )

    epoch_review = epoch_qc.copy()
    epoch_review["excluded_bad_channels_list"] = epoch_review[
        "excluded_bad_channels"
    ].apply(parse_saved_list)

    epoch_most_rejected = epoch_review.sort_values(
        "percent_epochs_dropped",
        ascending=False,
    )[
        [
            "subject_id",
            "condition",
            "n_epochs_before_rejection",
            "n_epochs_after_rejection",
            "n_epochs_dropped",
            "percent_epochs_dropped",
            "excluded_bad_channels",
        ]
    ]

    epoch_excluded_channels_long = (
        epoch_review[
            [
                "subject_id",
                "condition",
                "percent_epochs_dropped",
                "excluded_bad_channels_list",
            ]
        ]
        .explode("excluded_bad_channels_list")
        .dropna(subset=["excluded_bad_channels_list"])
        .rename(columns={"excluded_bad_channels_list": "channel"})
    )
    epoch_excluded_channels_long = epoch_excluded_channels_long.loc[
        epoch_excluded_channels_long["channel"] != ""
    ]

    return {
        "status_summary": status_summary,
        "problem_subjects": problem_subjects,
        "prep_most_noisy": prep_most_noisy,
        "prep_still_noisy_channel_frequency": prep_still_noisy_channel_frequency,
        "prep_still_noisy_channels_long": prep_still_noisy_channels_long,
        "ica_compact": ica_compact,
        "ica_most_removed": ica_most_removed,
        "epoch_most_rejected": epoch_most_rejected,
        "epoch_excluded_channels_long": epoch_excluded_channels_long,
    }


def save_qc_review_tables(
    qc_review_tables: dict[str, pd.DataFrame],
    output_dir: str | Path = "data/processed/eeg",
) -> Path:
    """Save review-friendly QC tables to disk."""
    qc_review_dir = Path(output_dir) / "qc_review"
    qc_review_dir.mkdir(parents=True, exist_ok=True)
    for name, table in qc_review_tables.items():
        table.to_csv(qc_review_dir / f"{name}.tsv", sep="\t", index=True)
    return qc_review_dir


def _run_prep(
    raw_condition: mne.io.BaseRaw,
    *,
    montage: mne.channels.DigMontage | None,
    prep_params: dict,
    random_state: int,
) -> tuple[mne.io.BaseRaw, PrepPipeline]:
    prep = PrepPipeline(
        raw_condition.copy(),
        prep_params=prep_params,
        montage=montage,
        ransac=True,
        random_state=random_state,
    )
    prep.fit()
    return prep.raw_eeg, prep


def _fit_ica(
    raw: mne.io.BaseRaw,
    *,
    n_components: int | float | None,
    method: str,
    random_state: int,
    fit_params: dict | None,
) -> ICA:
    ica = ICA(
        n_components=n_components,
        method=method,
        max_iter="auto",
        random_state=random_state,
        fit_params=fit_params,
    )
    ica.fit(raw)
    return ica


def _find_eog_components_from_proxies(
    ica: ICA,
    raw: mne.io.BaseRaw,
    *,
    proxy_channels: tuple[str, ...] = ("Fp1", "Fp2", "AFp1", "AFp2"),
    threshold: float = 3.0,
) -> list[int]:
    candidates: set[int] = set()
    for ch_name in proxy_channels:
        if ch_name not in raw.ch_names:
            continue
        component_indices, _ = ica.find_bads_eog(
            raw,
            ch_name=ch_name,
            threshold=threshold,
            measure="zscore",
        )
        candidates.update(component_indices)
    return sorted(candidates)


def _get_icalabel_artifacts(
    raw: mne.io.BaseRaw,
    ica: ICA,
    *,
    thresholds: dict[str, float],
) -> tuple[list[int], pd.DataFrame]:
    try:
        from mne_icalabel import label_components
    except ImportError as exc:
        raise ImportError(
            "mne-icalabel is required for ICLabel cleaning. "
            "Install it with `pip install mne-icalabel`, or run with use_icalabel=False."
        ) from exc

    labels = label_components(raw, ica, method="iclabel")
    rows = []
    artifact_components = []
    for idx, (label, prob) in enumerate(zip(labels["labels"], labels["y_pred_proba"])):
        probability = float(prob)
        rows.append({"component": idx, "label": label, "probability": probability})
        threshold = thresholds.get(label)
        if threshold is not None and probability >= threshold:
            artifact_components.append(idx)

    return artifact_components, pd.DataFrame(rows)


def make_clean_rest_epochs(
    raw_clean: mne.io.BaseRaw,
    *,
    epoch_duration_s: float = 2.0,
    epoch_overlap_s: float = 0.0,
    reject_eeg_uv: float = 150.0,
    flat_eeg_uv: float = 1.0,
) -> tuple[mne.Epochs, dict]:
    """Create fixed-length clean rest epochs, excluding channels marked bad."""
    raw_for_epoch = raw_clean.copy().pick("eeg", exclude="bads")
    excluded_bads = sorted(set(raw_clean.info["bads"]) - set(raw_for_epoch.ch_names))

    epochs = mne.make_fixed_length_epochs(
        raw_for_epoch,
        duration=epoch_duration_s,
        overlap=epoch_overlap_s,
        preload=True,
        reject_by_annotation=True,
    )
    n_before = len(epochs)
    epochs.drop_bad(
        reject={"eeg": reject_eeg_uv * 1e-6},
        flat={"eeg": flat_eeg_uv * 1e-6},
    )
    n_after = len(epochs)

    qc = {
        "n_epochs_before_rejection": n_before,
        "n_epochs_after_rejection": n_after,
        "n_epochs_dropped": n_before - n_after,
        "percent_epochs_dropped": 100 * (n_before - n_after) / n_before
        if n_before
        else 0.0,
        "epoch_duration_s": epoch_duration_s,
        "epoch_overlap_s": epoch_overlap_s,
        "reject_eeg_uv": reject_eeg_uv,
        "flat_eeg_uv": flat_eeg_uv,
        "excluded_bad_channels": excluded_bads,
    }
    return epochs, qc


def preprocess_rest_subject(
    vhdr_path: str | Path,
    *,
    output_dir: str | Path | None = "data/processed/eeg",
    save_epochs: bool = True,
    resample_sfreq: float = 250.0,
    line_freq: float = 50.0,
    filter_l_freq: float = 1.0,
    filter_h_freq: float = 40.0,
    prep_params: dict | None = None,
    prep_random_state: int = 42,
    ica_method: str = "fastica",
    ica_n_components: int | float | None = None,
    ica_random_state: int = 7,
    ica_fit_params: dict | None = None,
    use_icalabel: bool = True,
    artifact_thresholds: dict[str, float] | None = None,
    epoch_duration_s: float = 2.0,
    epoch_overlap_s: float = 0.0,
    reject_eeg_uv: float = 150.0,
    flat_eeg_uv: float = 1.0,
) -> RestPreprocessingResult:
    """Run the resting-state EEG preprocessing pipeline for one subject."""
    vhdr_path = Path(vhdr_path)
    subject_id = subject_id_from_vhdr(vhdr_path)
    prep_params = DEFAULT_PREP_PARAMS if prep_params is None else prep_params
    artifact_thresholds = (
        DEFAULT_ARTIFACT_THRESHOLDS
        if artifact_thresholds is None
        else artifact_thresholds
    )

    raw = load_rest_eeg(vhdr_path, preload=True)
    montage = raw.get_montage()
    raw_open, raw_closed, markers = crop_rest_conditions(raw)
    raw_by_condition = {"eyes_open": raw_open, "eyes_closed": raw_closed}

    prep_rows = []
    ica_rows = []
    epoch_rows = []
    clean_raws: dict[Condition, mne.io.BaseRaw] = {}
    epochs_by_condition: dict[Condition, mne.Epochs] = {}
    icas: dict[Condition, ICA] = {}

    for condition, raw_condition in raw_by_condition.items():
        raw_denoised = raw_condition.copy().notch_filter(
            freqs=[line_freq],
            method="spectrum_fit",
            filter_length="10s",
        )
        raw_resampled = raw_denoised.copy().resample(resample_sfreq)

        raw_prep, prep = _run_prep(
            raw_resampled,
            montage=montage,
            prep_params=prep_params,
            random_state=prep_random_state,
        )

        raw_filtered = raw_prep.copy().filter(
            l_freq=filter_l_freq,
            h_freq=filter_h_freq,
        )
        ica = _fit_ica(
            raw_filtered,
            n_components=ica_n_components,
            method=ica_method,
            random_state=ica_random_state,
            fit_params=ica_fit_params,
        )

        eog_components = _find_eog_components_from_proxies(ica, raw_filtered)
        icalabel_components: list[int] = []
        icalabel_summary = pd.DataFrame()
        if use_icalabel:
            icalabel_components, icalabel_summary = _get_icalabel_artifacts(
                raw_filtered,
                ica,
                thresholds=artifact_thresholds,
            )

        excluded_components = sorted(set(eog_components) | set(icalabel_components))
        ica_clean = ica.copy()
        ica_clean.exclude = excluded_components
        raw_clean = raw_filtered.copy()
        ica_clean.apply(raw_clean)

        epochs, epoch_qc = make_clean_rest_epochs(
            raw_clean,
            epoch_duration_s=epoch_duration_s,
            epoch_overlap_s=epoch_overlap_s,
            reject_eeg_uv=reject_eeg_uv,
            flat_eeg_uv=flat_eeg_uv,
        )

        prep_rows.append(
            {
                "subject_id": subject_id,
                "condition": condition,
                "interpolated_channels": _to_str_list(prep.interpolated_channels),
                "still_noisy_channels": _to_str_list(prep.still_noisy_channels),
                "n_interpolated_channels": len(prep.interpolated_channels),
                "n_still_noisy_channels": len(prep.still_noisy_channels),
            }
        )
        ica_rows.append(
            {
                "subject_id": subject_id,
                "condition": condition,
                "method": ica_method,
                "n_components": ica.n_components_,
                "eog_components": eog_components,
                "icalabel_components": icalabel_components,
                "excluded_components": excluded_components,
                "n_excluded_components": len(excluded_components),
                "icalabel_summary": icalabel_summary.to_dict("records"),
            }
        )
        epoch_rows.append({"subject_id": subject_id, "condition": condition, **epoch_qc})

        clean_raws[condition] = raw_clean
        epochs_by_condition[condition] = epochs
        icas[condition] = ica_clean

        if output_dir is not None and save_epochs:
            out_dir = Path(output_dir) / subject_id / "eeg"
            out_dir.mkdir(parents=True, exist_ok=True)
            epochs.save(
                out_dir / f"{subject_id}_task-rest_{condition}_clean-epo.fif",
                overwrite=True,
            )

    prep_qc = pd.DataFrame(prep_rows)
    ica_qc = pd.DataFrame(ica_rows)
    epoch_qc = pd.DataFrame(epoch_rows)

    if output_dir is not None:
        out_dir = Path(output_dir) / subject_id / "eeg"
        out_dir.mkdir(parents=True, exist_ok=True)
        prep_qc.to_csv(
            out_dir / f"{subject_id}_task-rest_prep_qc.tsv",
            sep="\t",
            index=False,
        )
        ica_qc.to_csv(
            out_dir / f"{subject_id}_task-rest_ica_qc.tsv",
            sep="\t",
            index=False,
        )
        epoch_qc.to_csv(
            out_dir / f"{subject_id}_task-rest_epoch_qc.tsv",
            sep="\t",
            index=False,
        )

    return RestPreprocessingResult(
        subject_id=subject_id,
        vhdr_path=vhdr_path,
        markers=markers,
        prep_qc=prep_qc,
        ica_qc=ica_qc,
        epoch_qc=epoch_qc,
        epochs=epochs_by_condition,
        raw_clean=clean_raws,
        icas=icas,
    )


def preprocess_all_rest_subjects(
    data_root: str | Path = "data/ds004796",
    *,
    output_dir: str | Path | None = "data/processed/eeg",
    subject_ids: list[str] | None = None,
    continue_on_error: bool = True,
    run_all: bool = False,
    **kwargs,
) -> tuple[list[RestPreprocessingResult], pd.DataFrame]:
    """Run the rest EEG preprocessing pipeline across downloaded subjects.

    If run_all=False, subjects with complete saved outputs are skipped so an
    interrupted batch can resume from missing or partial subjects.
    """
    vhdr_paths = find_rest_vhdrs(data_root)
    if subject_ids is not None:
        if not subject_ids:
            raise ValueError("subject_ids=[] selects no subjects. Use subject_ids=None to process all subjects.")
        wanted = set(subject_ids)
        vhdr_paths = [path for path in vhdr_paths if subject_id_from_vhdr(path) in wanted]

    results = []
    rows = []
    for vhdr_path in vhdr_paths:
        subject_id = subject_id_from_vhdr(vhdr_path)
        if (
            output_dir is not None
            and not run_all
            and subject_preprocessing_complete(output_dir, subject_id)
        ):
            rows.append(
                {
                    "subject_id": subject_id,
                    "status": "already_processed",
                    "error": "",
                }
            )
            continue

        try:
            result = preprocess_rest_subject(
                vhdr_path,
                output_dir=output_dir,
                **kwargs,
            )
            results.append(result)
            rows.append({"subject_id": subject_id, "status": "ok", "error": ""})
        except Exception as exc:
            rows.append({"subject_id": subject_id, "status": "error", "error": str(exc)})
            if not continue_on_error:
                raise

    if output_dir is not None:
        rebuilt_qc = rebuild_rest_preprocessing_qc(output_dir, data_root=data_root)
        rebuilt_run_qc = rebuilt_qc["run_qc"]
        attempted_run_qc = pd.DataFrame(rows, columns=["subject_id", "status", "error"])
        run_qc = rebuilt_run_qc.merge(
            attempted_run_qc,
            on="subject_id",
            how="left",
            suffixes=("", "_this_run"),
        )
        run_qc["status_this_run"] = run_qc["status_this_run"].fillna("not_selected")
        run_qc["error_this_run"] = run_qc["error_this_run"].fillna("")
        run_qc.to_csv(
            Path(output_dir) / "rest_preprocessing_run_qc.tsv",
            sep="\t",
            index=False,
        )
    else:
        run_qc = pd.DataFrame(rows, columns=["subject_id", "status", "error"])

    return results, run_qc
