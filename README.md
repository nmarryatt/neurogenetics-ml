# Neural Signatures of Alzheimer's Disease Genetic Risk

> Predicting APOE ε4 and PICALM carrier status from resting-state EEG and fMRI in cognitively healthy middle-aged adults, to explore pre-symptomatic neural mechanisms underlying Alzheimers Disease risk.
---

## Project Status

**In progress** — currently: data exploration (`01_data_exploration.ipynb`)

| Notebook | Status |
|---|---|
| 01 — Data Exploration | In progress |
| 02 — Preprocessing | Pending |
| 03 — Feature Extraction | Pending |
| 04 — Classification | Pending |
| 05 — Interpretability & Discussion | Pending |

---

## Scientific Background

Alzheimer's disease (AD) is the leading cause of dementia, affecting more than 50 million people worldwide. While no single gene is diagnostic of AD, genetic predisposition plays a significant role in disease onset and progression. The APOE ε4 allele is the strongest known genetic risk factor for late-onset AD, heterozygous carriers face an approximately 3-fold increased risk, while homozygous carriers face up to 14-fold increased risk. PICALM variants represent a secondary, mechanistically distinct risk factor.

Despite their well-established association with AD risk, the mechanisms by which these genes confer susceptibility remain incompletely understood. Critically, their biological effects are thought to manifest decades before any clinical symptoms emerge, suggesting a window in which pre-symptomatic neural changes may be detectable.

This project investigates whether such changes are measurable through non-invasive brain recordings in cognitively healthy adults. Specifically, it asks:
1. **Can resting-state EEG and fMRI features predict APOE ε4 carrier status** in cognitively healthy middle-aged adults?
2. **Which neural features drive that prediction** and are they consistent with known Alzheimer's biomarkers?
3. **Do the same signatures appear in PICALM risk carriers**, or do these genetically distinct risk pathways produce distinct neural phenotypes?
4. **Are the findings consistent with current understanding of AD pathology** and do they support the hypothesis that genetic risk manifests as measurable neural phenotypes decades before clinical onset?

The approach is motivated by prior work (Dzianok & Kublik, 2025) showing AD-like EEG/fMRI features in asymptomatic APOE/PICALM carriers. This project extends that work by applying machine learning classification with SHAP-based interpretability, enabling feature-level interrogation rather than group-level comparison alone.

---

## Dataset

**PEARL-Neuro Database**
*Polish Electroencephalography, Alzheimer's Risk-genes, Lifestyle and Neuroimaging*

| Property | Detail |
|---|---|
| Source | OpenNeuro `ds004796` |
| DOI | [10.18112/openneuro.ds004796.v1.0.4](https://doi.org/10.18112/openneuro.ds004796.v1.0.4) |
| N (neuroimaging subset) | ~79 participants |
| Age range | 50–63 years |
| EEG channels | 128 electrodes |
| Modalities | EEG, fMRI, genetics, psychometrics, blood tests |
| Format | BIDS |
| Access | Open access |

Participants are cognitively healthy adults stratified by APOE and PICALM genotype. The dataset includes resting-state recordings (eyes open and eyes closed) and two cognitive tasks: the Multi-Source Interference Task (MSIT) and Sternberg's memory task.

> **Note:** Raw data is not included in this repository. Download instructions are provided below.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/neurogenetics-ml.git
cd neurogenetics-ml
```

### 2. Set up the environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name neurogenetics-ml --display-name "Python (neurogenetics-ml)"
```

In VS Code, open the notebook and select the `Python (neurogenetics-ml)` kernel, or choose the interpreter at `.venv/bin/python`.

### 3. Download the data

Data is hosted on OpenNeuro and downloaded separately. Using AWS CLI (no account required):

```bash
# Start with one subject to explore the structure
aws s3 sync --no-sign-request s3://openneuro.org/ds004796/sub-01 data/raw/sub-01/

# Download all EEG data when ready
aws s3 sync --no-sign-request s3://openneuro.org/ds004796 data/raw/ \
  --exclude "*" \
  --include "*/eeg/*" \
  --include "participants.tsv" \
  --include "dataset_description.json"
```

### 4. Run the notebooks

Launch Jupyter and run notebooks in order (01 → 05):

```bash
jupyter notebook notebooks/
```

---

## Key Dependencies

| Package | Purpose |
|---|---|
| `mne` | EEG loading, preprocessing, visualisation |
| `nilearn` | fMRI processing and connectivity |
| `scikit-learn` | Classification models, cross-validation |
| `shap` | Model interpretability |
| `numpy` / `scipy` | Numerical computing, signal processing |
| `matplotlib` / `seaborn` | Visualisation |

Full dependency list in `environment.yml`.

---

## Results

*To be updated as analysis progresses.*

---

## References

Dzianok, P., Wojciechowski, J., Wolak, T., & Kublik, E. (2025). Alzheimer's disease-like features in resting state EEG/fMRI of cognitively intact and healthy middle-aged APOE/PICALM risk carriers. *Journal of Alzheimer's Disease*, 104(2), 509–524. https://doi.org/10.1177/13872877251317489

Dzianok, P. & Kublik, E. (2024). PEARL-Neuro Database: EEG, fMRI, health and lifestyle data of middle-aged people at risk of dementia. *Scientific Data*, 11, 276. https://doi.org/10.1038/s41597-024-03106-5

---

## Acknowledgements

Data from the PEARL-Neuro Database (OpenNeuro ds004796), collected at the Laboratory of Emotions Neurobiology, Nencki Institute of Experimental Biology PAS, Warsaw, Poland. Funded by the Polish National Science Centre (NCN).

---

## License

Code in this repository is released under the MIT License. Data belongs to the original dataset authors and is subject to its own terms — see the [OpenNeuro dataset page](https://openneuro.org/datasets/ds004796) for details.
