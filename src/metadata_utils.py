from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_METADATA_PATH = Path("data/participant_metadata/participants.tsv")


def load_full_cohort(
    path: str | Path = DEFAULT_METADATA_PATH,
    na_values: tuple[str, ...] = ("n/a",),
) -> pd.DataFrame:
    """Load the full participant metadata table."""
    return pd.read_csv(path, sep="\t", na_values=list(na_values))


def get_imaging_cohort(full_cohort: pd.DataFrame) -> pd.DataFrame:
    """Subset to the second-phase neuroimaging cohort."""
    return full_cohort.loc[full_cohort["second_phase"] == 1].copy()


def clean_genotype_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from genotype string columns used in this project."""
    out = df.copy()
    for col in ("APOE_haplotype", "PICALM_rs3851179", "APOE_rs429358", "APOE_rs7412"):
        if col in out.columns:
            out[col] = out[col].astype("string").str.strip()
    return out


def add_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add human-readable label columns while preserving coded originals."""
    out = df.copy()

    if "education" in out.columns:
        out["education_label"] = out["education"].replace(
            {
                0: "primary education",
                1: "secondary education",
                2: "partial higher education",
                3: "higher education",
            }
        )

    if "sex" in out.columns:
        out["sex_label"] = out["sex"].replace({0: "male", 1: "female"})

    if "diabetes" in out.columns:
        out["diabetes_label"] = out["diabetes"].replace({0: "no", 1: "yes"})

    if "hypertension" in out.columns:
        out["hypertension_label"] = out["hypertension"].replace({0: "no", 1: "yes"})

    if "smoking_status" in out.columns:
        out["smoking_status_label"] = out["smoking_status"].replace(
            {0: "no", 1: "yes", 2: "in past"}
        )

    return out


def allele_dosage(genotype: str | pd.NA, allele: str):
    """Return 0/1/2 dosage for a requested allele in slash-delimited genotypes."""
    if pd.isna(genotype):
        return pd.NA
    a1, a2 = str(genotype).strip().split("/")
    return int(a1 == allele) + int(a2 == allele)


def add_risk_dosage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add APOE e4 dosage and PICALM G-risk dosage columns."""
    out = clean_genotype_strings(df)

    if "APOE_haplotype" in out.columns:
        out["APOE_risk_dosage"] = out["APOE_haplotype"].apply(
            lambda x: str(x).split("/").count("e4") if pd.notna(x) else pd.NA
        )

    if "PICALM_rs3851179" in out.columns:
        out["PICALM_risk_dosage"] = out["PICALM_rs3851179"].apply(
            lambda x: allele_dosage(x, "G")
        )

    return out


def add_genotype_group_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple APOE and PICALM grouping columns used in exploratory tables."""
    out = clean_genotype_strings(df)

    if "APOE_haplotype" in out.columns:
        out["APOE_group"] = out["APOE_haplotype"].apply(
            lambda x: "e4_carrier" if pd.notna(x) and "e4" in str(x) else "non_e4"
        )

    if "PICALM_rs3851179" in out.columns:
        out["PICALM_group"] = out["PICALM_rs3851179"].replace(
            {"A/A": "AA_AG", "A/G": "AA_AG", "G/A": "AA_AG", "G/G": "GG"}
        )

    return out


def make_dosage_table(
    df: pd.DataFrame,
    dosage_col: str,
    *,
    sex_col: str = "sex_label",
    diabetes_col: str = "diabetes_label",
    hypertension_col: str = "hypertension_label",
    education_col: str = "education_label",
) -> pd.DataFrame:
    """Create a compact descriptive table stratified by 0/1/2 dosage."""

    grouped = df.groupby(dosage_col)
    rows: dict[str, pd.Series] = {}

    rows["N"] = grouped.size()
    rows["Age mean (SD)"] = grouped["age"].apply(lambda x: f"{x.mean():.2f} ({x.std():.2f})")

    def n_pct(x: pd.Series, value: str) -> str:
        n = (x == value).sum()
        pct = (x == value).mean() * 100
        return f"{n} ({pct:.1f}%)"

    rows["Female, n (%)"] = grouped[sex_col].apply(lambda x: n_pct(x, "female"))
    rows["Diabetes, n (%)"] = grouped[diabetes_col].apply(lambda x: n_pct(x, "yes"))
    rows["Hypertension, n (%)"] = grouped[hypertension_col].apply(
        lambda x: n_pct(x, "yes")
    )
    rows["Higher education, n (%)"] = grouped[education_col].apply(
        lambda x: n_pct(x, "higher education")
    )

    table = pd.DataFrame(rows).T
    return table.reindex(columns=[0, 1, 2])


def prepare_imaging_metadata(path: str | Path = DEFAULT_METADATA_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load full cohort and return a cleaned imaging cohort ready for EDA."""
    full_cohort = load_full_cohort(path=path)
    participants = get_imaging_cohort(full_cohort)
    participants = add_label_columns(participants)
    participants = add_risk_dosage_columns(participants)
    participants = add_genotype_group_columns(participants)
    return full_cohort, participants
