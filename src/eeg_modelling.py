from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import PredefinedSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


ID_COLS = {
    "subject_id",
    "condition",
    "n_epochs",
    "n_channels",
    "sfreq",
    "split",
    "cv_fold",
    "APOE_group",
    "PICALM_group",
    "APOE_risk_dosage",
    "PICALM_risk_dosage",
    "age",
    "sex",
    "education",
}

BASE_METADATA_COLS = [
    "subject_id",
    "split",
    "cv_fold",
    "APOE_group",
    "PICALM_group",
    "APOE_risk_dosage",
    "PICALM_risk_dosage",
    "age",
    "sex",
    "education",
]


def make_subject_level_datasets(features_long: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, list[str]]]:
    """Create eyes-closed, eyes-open, and combined subject-level feature tables."""
    feature_cols = [
        col
        for col in features_long.columns
        if col not in ID_COLS and pd.api.types.is_numeric_dtype(features_long[col])
    ]
    subject_metadata = features_long[BASE_METADATA_COLS].drop_duplicates("subject_id")

    def condition_dataset(condition: str) -> pd.DataFrame:
        cols = BASE_METADATA_COLS + feature_cols
        out = features_long.loc[features_long["condition"] == condition, cols].copy()
        return out.drop_duplicates("subject_id").reset_index(drop=True)

    wide = features_long.pivot(index="subject_id", columns="condition", values=feature_cols)
    wide.columns = [f"{feature}_{condition}" for feature, condition in wide.columns]
    combined = subject_metadata.merge(wide.reset_index(), on="subject_id", how="inner")

    datasets = {
        "eyes_closed": condition_dataset("eyes_closed"),
        "eyes_open": condition_dataset("eyes_open"),
        "combined": combined,
    }
    feature_columns = {
        name: [col for col in df.columns if col not in ID_COLS and pd.api.types.is_numeric_dtype(df[col])]
        for name, df in datasets.items()
    }
    return datasets, feature_columns


def dataset_summary(datasets: dict[str, pd.DataFrame], feature_columns: dict[str, list[str]]) -> pd.DataFrame:
    """Summarize sample size and feature count for each modelling dataset."""
    return pd.DataFrame(
        [
            {
                "dataset": name,
                "n_subjects": df["subject_id"].nunique(),
                "n_train_subjects": df.loc[df["split"] == "train", "subject_id"].nunique(),
                "n_test_subjects": df.loc[df["split"] == "test", "subject_id"].nunique(),
                "n_features": len(feature_columns[name]),
            }
            for name, df in datasets.items()
        ]
    )


def make_first_attempt_models(random_state: int = 7) -> dict[str, Pipeline]:
    """Return first-pass models for full-feature APOE carrier prediction."""
    return {
        "logistic_regression_l2_c0.1": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        penalty="l2",
                        C=0.1,
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=4,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "xgboost_gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=150,
                        max_depth=2,
                        learning_rate=0.05,
                        subsample=0.80,
                        colsample_bytree=0.80,
                        reg_lambda=3.0,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }


def prepare_xy(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_col: str = "APOE_group",
    positive_label: str = "e4_carrier",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, list[str]]:
    """Create X/y data while dropping all-missing feature columns."""
    modelling_df = df.loc[df[target_col].notna()].copy()
    y = (modelling_df[target_col] == positive_label).astype(int)
    x_cols = [
        col
        for col in feature_columns
        if col in modelling_df.columns and pd.api.types.is_numeric_dtype(modelling_df[col])
    ]
    X = modelling_df[x_cols]
    all_missing_cols = X.columns[X.isna().all()].tolist()
    if all_missing_cols:
        X = X.drop(columns=all_missing_cols)
        x_cols = [col for col in x_cols if col not in all_missing_cols]
    return modelling_df, X, y, x_cols


def evaluate_cv_and_test(
    datasets: dict[str, pd.DataFrame],
    feature_columns: dict[str, list[str]],
    models: dict[str, Pipeline],
    target_col: str = "APOE_group",
    positive_label: str = "e4_carrier",
) -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, object]]]:
    """Evaluate each dataset/model with CV balanced accuracy and test balanced accuracy."""
    rows = []
    fitted_candidates: dict[tuple[str, str], dict[str, object]] = {}

    for dataset_name, df in datasets.items():
        train_df = df.loc[df["split"] == "train"].copy()
        test_df = df.loc[df["split"] == "test"].copy()

        train_df, X_train, y_train, x_cols = prepare_xy(
            train_df, feature_columns[dataset_name], target_col, positive_label
        )
        test_df, X_test, y_test, _ = prepare_xy(test_df, feature_columns[dataset_name], target_col, positive_label)
        X_test = X_test.reindex(columns=x_cols)
        cv = PredefinedSplit(test_fold=train_df["cv_fold"].astype(int).to_numpy() - 1)

        for model_name, model in models.items():
            cv_scores = cross_validate(
                clone(model),
                X_train,
                y_train,
                cv=cv,
                scoring=["balanced_accuracy"],
                n_jobs=1,
                error_score="raise",
            )

            fitted = clone(model).fit(X_train, y_train)
            y_pred = fitted.predict(X_test)

            rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "n_subjects_train": train_df["subject_id"].nunique(),
                    "n_subjects_test": test_df["subject_id"].nunique(),
                    "n_features_input": X_train.shape[1],
                    "cv_balanced_accuracy_mean": cv_scores["test_balanced_accuracy"].mean(),
                    "cv_balanced_accuracy_sd": cv_scores["test_balanced_accuracy"].std(),
                    "test_balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                }
            )

            fitted_candidates[(dataset_name, model_name)] = {
                "model": fitted,
                "x_cols": x_cols,
                "y_test": y_test,
                "y_pred": y_pred,
            }

    results = pd.DataFrame(rows).sort_values("cv_balanced_accuracy_mean", ascending=False)
    return results, fitted_candidates


def feature_subtype_columns(feature_columns: list[str]) -> dict[str, list[str]]:
    """Group EEG features into interpretable feature families."""
    subtype_prefixes = {
        "spectral": ("power_", "theta_beta_ratio_"),
        "aperiodic_periodic": ("aperiodic_", "peak_", "strongest_peak_", "n_oscillatory_peaks_"),
        "time_complexity": ("td_", "hjorth_", "complexity_"),
        "connectivity_graph": ("connectivity_", "graph_"),
        "microstate": ("microstate_",),
    }
    return {
        subtype: [col for col in feature_columns if col.startswith(prefixes)]
        for subtype, prefixes in subtype_prefixes.items()
    }


def evaluate_feature_subtypes(
    datasets: dict[str, pd.DataFrame],
    feature_columns: dict[str, list[str]],
    models: dict[str, Pipeline],
    datasets_to_include: tuple[str, ...] = ("eyes_open", "eyes_closed"),
    target_col: str = "APOE_group",
    positive_label: str = "e4_carrier",
) -> pd.DataFrame:
    """Evaluate each feature family separately in each requested resting condition."""
    rows = []
    for dataset_name in datasets_to_include:
        df = datasets[dataset_name]
        subtype_cols = feature_subtype_columns(feature_columns[dataset_name])
        train_df = df.loc[df["split"] == "train"].copy()
        test_df = df.loc[df["split"] == "test"].copy()

        for subtype, cols in subtype_cols.items():
            if not cols:
                continue

            train_df_prepared, X_train, y_train, x_cols = prepare_xy(train_df, cols, target_col, positive_label)
            test_df_prepared, X_test, y_test, _ = prepare_xy(test_df, cols, target_col, positive_label)
            X_test = X_test.reindex(columns=x_cols)
            cv = PredefinedSplit(test_fold=train_df_prepared["cv_fold"].astype(int).to_numpy() - 1)

            for model_name, model in models.items():
                cv_scores = cross_validate(
                    clone(model),
                    X_train,
                    y_train,
                    cv=cv,
                    scoring=["balanced_accuracy"],
                    n_jobs=1,
                    error_score="raise",
                )
                fitted = clone(model).fit(X_train, y_train)
                y_pred = fitted.predict(X_test)
                rows.append(
                    {
                        "dataset": dataset_name,
                        "feature_subtype": subtype,
                        "model": model_name,
                        "n_subjects_train": train_df_prepared["subject_id"].nunique(),
                        "n_subjects_test": test_df_prepared["subject_id"].nunique(),
                        "n_features_input": X_train.shape[1],
                        "cv_balanced_accuracy_mean": cv_scores["test_balanced_accuracy"].mean(),
                        "cv_balanced_accuracy_sd": cv_scores["test_balanced_accuracy"].std(),
                        "test_balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                    }
                )

    return pd.DataFrame(rows).sort_values(["dataset", "feature_subtype", "cv_balanced_accuracy_mean"])


def plot_cv_test_balanced_accuracy(
    results: pd.DataFrame,
    output_path: str | Path,
    title: str,
    group_col: str = "dataset",
    height: float = 4.8,
) -> plt.Figure:
    """Plot CV and held-out test balanced accuracy side by side."""
    plot_df = results.melt(
        id_vars=[group_col, "model"],
        value_vars=["cv_balanced_accuracy_mean", "test_balanced_accuracy"],
        var_name="evaluation",
        value_name="balanced_accuracy",
    )
    plot_df["evaluation"] = plot_df["evaluation"].map(
        {
            "cv_balanced_accuracy_mean": "CV",
            "test_balanced_accuracy": "Test fold",
        }
    )
    plot_df["analysis"] = plot_df[group_col] + " | " + plot_df["model"]
    order = (
        results.assign(analysis=results[group_col] + " | " + results["model"])
        .sort_values("cv_balanced_accuracy_mean", ascending=False)["analysis"]
    )

    fig, ax = plt.subplots(figsize=(10, height))
    sns.barplot(
        data=plot_df,
        y="analysis",
        x="balanced_accuracy",
        hue="evaluation",
        order=order,
        ax=ax,
    )
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("Balanced accuracy")
    ax.set_ylabel("")
    ax.set_title(title)
    ax.legend(title="", frameon=False)
    sns.despine()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_feature_subtype_results(
    results: pd.DataFrame,
    dataset: str,
    output_path: str | Path,
    title: str,
) -> plt.Figure:
    """Plot CV and held-out test balanced accuracy by model for one condition."""
    dataset_results = results.loc[results["dataset"] == dataset].copy()
    plot_df = dataset_results.melt(
        id_vars=["feature_subtype", "model"],
        value_vars=["cv_balanced_accuracy_mean", "test_balanced_accuracy"],
        var_name="evaluation",
        value_name="balanced_accuracy",
    )
    plot_df["evaluation"] = plot_df["evaluation"].map(
        {
            "cv_balanced_accuracy_mean": "CV",
            "test_balanced_accuracy": "Test fold",
        }
    )
    subtype_labels = {
        "spectral": "Spectral",
        "aperiodic_periodic": "Aperiodic/periodic",
        "time_complexity": "Time/complexity",
        "connectivity_graph": "Connectivity/graph",
        "microstate": "Microstate",
    }
    model_labels = {
        "logistic_regression_l2_c0.1": "Logistic regression",
        "random_forest": "Random forest",
        "xgboost_gradient_boosting": "XGBoost",
    }
    plot_df["feature_family"] = plot_df["feature_subtype"].map(subtype_labels)
    plot_df["model_label"] = plot_df["model"].map(model_labels)
    subtype_order = [
        "Spectral",
        "Aperiodic/periodic",
        "Time/complexity",
        "Connectivity/graph",
        "Microstate",
    ]
    model_order = [
        "Logistic regression",
        "Random forest",
        "XGBoost",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharex=True, sharey=True)
    for ax, evaluation in zip(axes, ["CV", "Test fold"], strict=True):
        sns.barplot(
            data=plot_df.loc[plot_df["evaluation"] == evaluation],
            y="feature_family",
            x="balanced_accuracy",
            hue="model_label",
            order=[name for name in subtype_order if name in plot_df["feature_family"].unique()],
            hue_order=[name for name in model_order if name in plot_df["model_label"].unique()],
            ax=ax,
        )
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.6)
        ax.set_title(evaluation)
        ax.set_xlabel("Balanced accuracy")
        ax.set_ylabel("")
        ax.legend_.remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(title, y=0.98)
    sns.despine()
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def random_forest_feature_importances(
    datasets: dict[str, pd.DataFrame],
    feature_columns: dict[str, list[str]],
    view_specs: list[dict[str, str]],
    random_state: int = 7,
    target_col: str = "APOE_group",
    positive_label: str = "e4_carrier",
) -> pd.DataFrame:
    """Fit selected Random Forest view models and return training-set feature importances."""
    model = make_first_attempt_models(random_state=random_state)["random_forest"]
    rows = []

    for view in view_specs:
        dataset_name = view["dataset"]
        feature_subtype = view["feature_subtype"]
        view_label = view["view_label"]
        cols = feature_subtype_columns(feature_columns[dataset_name])[feature_subtype]
        train_df = datasets[dataset_name].loc[datasets[dataset_name]["split"] == "train"].copy()
        _, X_train, y_train, x_cols = prepare_xy(train_df, cols, target_col, positive_label)
        fitted = clone(model).fit(X_train, y_train)
        importances = fitted.named_steps["model"].feature_importances_

        rows.extend(
            {
                "view": view_label,
                "dataset": dataset_name,
                "feature_subtype": feature_subtype,
                "feature": feature,
                "importance": importance,
            }
            for feature, importance in zip(x_cols, importances, strict=True)
        )

    out = pd.DataFrame(rows)
    out["rank_within_view"] = out.groupby("view")["importance"].rank(method="first", ascending=False).astype(int)
    return out.sort_values(["view", "rank_within_view"])


def plot_top_feature_importances(
    importances: pd.DataFrame,
    output_path: str | Path,
    top_n: int = 15,
) -> plt.Figure:
    """Plot the top Random Forest feature importances within each selected view."""
    top = importances.loc[importances["rank_within_view"] <= top_n].copy()
    views = top["view"].drop_duplicates().tolist()
    fig, axes = plt.subplots(len(views), 1, figsize=(10, 3.4 * len(views)), squeeze=False)

    for ax, view in zip(axes.ravel(), views, strict=True):
        view_df = top.loc[top["view"] == view].sort_values("importance", ascending=True)
        sns.barplot(data=view_df, y="feature", x="importance", color="#377eb8", ax=ax)
        ax.set_title(view)
        ax.set_xlabel("Random Forest feature importance")
        ax.set_ylabel("")

    sns.despine()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def evaluate_three_view_probability_ensemble(
    datasets: dict[str, pd.DataFrame],
    feature_columns: dict[str, list[str]],
    view_specs: list[dict[str, str]],
    random_state: int = 7,
    target_col: str = "APOE_group",
    positive_label: str = "e4_carrier",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average Random Forest predicted probabilities from selected EEG feature views."""
    model = make_first_attempt_models(random_state=random_state)["random_forest"]
    reference_df = datasets[view_specs[0]["dataset"]]
    train_subjects = reference_df.loc[reference_df["split"] == "train", ["subject_id", target_col, "cv_fold"]]
    test_subjects = reference_df.loc[reference_df["split"] == "test", ["subject_id", target_col]]
    train_subjects = train_subjects.loc[train_subjects[target_col].notna()].copy()
    test_subjects = test_subjects.loc[test_subjects[target_col].notna()].copy()

    y_train_all = (train_subjects[target_col] == positive_label).astype(int).to_numpy()
    cv_fold = train_subjects["cv_fold"].astype(int).to_numpy()

    cv_predictions = []
    for fold in sorted(np.unique(cv_fold)):
        val_subject_ids = train_subjects.loc[cv_fold == fold, "subject_id"]
        fold_train_subject_ids = train_subjects.loc[cv_fold != fold, "subject_id"]
        fold_probs = []

        for view in view_specs:
            dataset_name = view["dataset"]
            feature_subtype = view["feature_subtype"]
            cols = feature_subtype_columns(feature_columns[dataset_name])[feature_subtype]
            view_df = datasets[dataset_name]
            fold_train_df = view_df.loc[view_df["subject_id"].isin(fold_train_subject_ids)].copy()
            fold_val_df = view_df.loc[view_df["subject_id"].isin(val_subject_ids)].copy()
            _, X_fold_train, y_fold_train, x_cols = prepare_xy(fold_train_df, cols, target_col, positive_label)
            val_prepared, X_fold_val, y_fold_val, _ = prepare_xy(fold_val_df, cols, target_col, positive_label)
            X_fold_val = X_fold_val.reindex(columns=x_cols)
            fitted = clone(model).fit(X_fold_train, y_fold_train)
            fold_probs.append(fitted.predict_proba(X_fold_val)[:, list(fitted.classes_).index(1)])

        mean_prob = np.mean(fold_probs, axis=0)
        cv_predictions.append(
            pd.DataFrame(
                {
                    "subject_id": val_prepared["subject_id"].to_numpy(),
                    "evaluation": "CV",
                    "fold": fold,
                    "true_label": y_fold_val.to_numpy(),
                    "predicted_probability": mean_prob,
                    "predicted_label": (mean_prob >= 0.5).astype(int),
                }
            )
        )

    cv_predictions_df = pd.concat(cv_predictions, ignore_index=True)
    cv_bal_acc = balanced_accuracy_score(cv_predictions_df["true_label"], cv_predictions_df["predicted_label"])

    test_probs = []
    for view in view_specs:
        dataset_name = view["dataset"]
        feature_subtype = view["feature_subtype"]
        cols = feature_subtype_columns(feature_columns[dataset_name])[feature_subtype]
        view_df = datasets[dataset_name]
        train_df = view_df.loc[view_df["subject_id"].isin(train_subjects["subject_id"])].copy()
        test_df = view_df.loc[view_df["subject_id"].isin(test_subjects["subject_id"])].copy()
        _, X_train, y_train, x_cols = prepare_xy(train_df, cols, target_col, positive_label)
        test_prepared, X_test, y_test, _ = prepare_xy(test_df, cols, target_col, positive_label)
        X_test = X_test.reindex(columns=x_cols)
        fitted = clone(model).fit(X_train, y_train)
        test_probs.append(fitted.predict_proba(X_test)[:, list(fitted.classes_).index(1)])

    test_mean_prob = np.mean(test_probs, axis=0)
    test_predictions_df = pd.DataFrame(
        {
            "subject_id": test_prepared["subject_id"].to_numpy(),
            "evaluation": "Test fold",
            "fold": np.nan,
            "true_label": y_test.to_numpy(),
            "predicted_probability": test_mean_prob,
            "predicted_label": (test_mean_prob >= 0.5).astype(int),
        }
    )
    predictions = pd.concat([cv_predictions_df, test_predictions_df], ignore_index=True)
    test_bal_acc = balanced_accuracy_score(test_predictions_df["true_label"], test_predictions_df["predicted_label"])

    summary = pd.DataFrame(
        [
            {
                "model": "three_view_random_forest_probability_ensemble",
                "views": " + ".join(view["view_label"] for view in view_specs),
                "n_subjects_train": train_subjects["subject_id"].nunique(),
                "n_subjects_test": test_subjects["subject_id"].nunique(),
                "cv_balanced_accuracy": cv_bal_acc,
                "test_balanced_accuracy": test_bal_acc,
            }
        ]
    )
    return summary, predictions


def evaluate_random_forest_feature_combo(
    datasets: dict[str, pd.DataFrame],
    feature_columns: dict[str, list[str]],
    dataset_name: str,
    feature_subtypes: tuple[str, ...],
    combo_label: str,
    random_state: int = 7,
    target_col: str = "APOE_group",
    positive_label: str = "e4_carrier",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit one Random Forest using a selected combination of feature families."""
    model = make_first_attempt_models(random_state=random_state)["random_forest"]
    subtype_cols = feature_subtype_columns(feature_columns[dataset_name])
    combo_cols = [col for subtype in feature_subtypes for col in subtype_cols[subtype]]

    df = datasets[dataset_name]
    train_df = df.loc[df["split"] == "train"].copy()
    test_df = df.loc[df["split"] == "test"].copy()
    train_df, X_train, y_train, x_cols = prepare_xy(train_df, combo_cols, target_col, positive_label)
    test_df, X_test, y_test, _ = prepare_xy(test_df, combo_cols, target_col, positive_label)
    X_test = X_test.reindex(columns=x_cols)
    cv = PredefinedSplit(test_fold=train_df["cv_fold"].astype(int).to_numpy() - 1)

    cv_scores = cross_validate(
        clone(model),
        X_train,
        y_train,
        cv=cv,
        scoring=["balanced_accuracy"],
        n_jobs=1,
        error_score="raise",
    )
    fitted = clone(model).fit(X_train, y_train)
    y_pred = fitted.predict(X_test)

    summary = pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "feature_set": combo_label,
                "model": "random_forest",
                "n_subjects_train": train_df["subject_id"].nunique(),
                "n_subjects_test": test_df["subject_id"].nunique(),
                "n_features_input": X_train.shape[1],
                "cv_balanced_accuracy_mean": cv_scores["test_balanced_accuracy"].mean(),
                "cv_balanced_accuracy_sd": cv_scores["test_balanced_accuracy"].std(),
                "test_balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            }
        ]
    )

    importances = pd.DataFrame(
        {
            "feature_set": combo_label,
            "feature": x_cols,
            "importance": fitted.named_steps["model"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importances["rank"] = range(1, len(importances) + 1)
    return summary, importances


def evaluate_random_forest_feature_combo_with_metrics(
    datasets: dict[str, pd.DataFrame],
    feature_columns: dict[str, list[str]],
    dataset_name: str,
    feature_subtypes: tuple[str, ...],
    combo_label: str,
    random_state: int = 7,
    target_col: str = "APOE_group",
    positive_label: str = "e4_carrier",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit one selected Random Forest feature combo and return held-out test metrics."""
    model = make_first_attempt_models(random_state=random_state)["random_forest"]
    subtype_cols = feature_subtype_columns(feature_columns[dataset_name])
    combo_cols = [col for subtype in feature_subtypes for col in subtype_cols[subtype]]

    df = datasets[dataset_name]
    train_df = df.loc[df["split"] == "train"].copy()
    test_df = df.loc[df["split"] == "test"].copy()
    train_df, X_train, y_train, x_cols = prepare_xy(train_df, combo_cols, target_col, positive_label)
    test_df, X_test, y_test, _ = prepare_xy(test_df, combo_cols, target_col, positive_label)
    X_test = X_test.reindex(columns=x_cols)

    fitted = clone(model).fit(X_train, y_train)
    y_pred = fitted.predict(X_test)
    y_prob = fitted.predict_proba(X_test)[:, list(fitted.classes_).index(1)]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    metrics = pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "feature_set": combo_label,
                "model": "random_forest",
                "n_subjects_train": train_df["subject_id"].nunique(),
                "n_subjects_test": test_df["subject_id"].nunique(),
                "n_features_input": len(x_cols),
                "accuracy": accuracy_score(y_test, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                "sensitivity_recall": recall_score(y_test, y_pred, zero_division=0),
                "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_prob) if y_test.nunique() == 2 else np.nan,
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn,
                "true_positives": tp,
            }
        ]
    )

    predictions = test_df[["subject_id", target_col]].copy()
    predictions["true_label"] = y_test.to_numpy()
    predictions["predicted_label"] = y_pred
    predictions["predicted_probability"] = y_prob
    return metrics, predictions
