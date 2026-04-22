# **HALO**

**HALO** is a research codebase for **bioactivity-driven prediction of antibacterial drug synergy** using machine learning and multi-scale chemical and biological features.

The repository contains the full pipeline used in the study, including:
- data preprocessing and curation,
- feature construction (Chemical Checker and strain-space),
- model training and evaluation,
- figure and table generation for the manuscript.

---

## Repository structure
```text
HALO_repo/
├── src/
├── data/
├── feature_pipeline/
├── figure_pipeline/
├── scripts/
├── notebooks/
├── results/
├── figures/
├── paper/
└── README.md
```

### Directory overview

- **`src/`**  
  Core reusable source code: mappers, shared utilities, and helper functions used across scripts and notebooks.

- **`data/`**  
  Data at different stages of the pipeline:
  - raw and extracted datasets,
  - curated reference tables (e.g. drug lists),
  - final feature matrices used for modeling.

- **`feature_pipeline/`**  
  Feature construction pipelines:
  - `chemicalchecker/`: fetching and assembling Chemical Checker features  
  - `strain_space/`: construction of strain-space (S-space) features following the Chemical Checker protocol

- **`scripts/`**  
  Scripted entry points to reproduce main experiments, model training, and external validation runs.

- **`notebooks/`**  
  Exploratory, preprocessing, and intermediate analysis notebooks used during development and data curation.

- **`figure_pipeline/`**  
  Scripts used to generate individual figure panels and tables programmatically.

- **`figures/`**  
  Exported final figures used in the manuscript (generated outputs).

- **`results/`**  
  Model outputs, logs, metrics, and intermediate experiment artifacts (generated).

- **`paper/`**  
  Manuscript source files and supplementary materials.

---

## Feature construction overview

HALO uses two complementary feature spaces:

1. **Chemical Checker (CC) features**  
Multi-level chemical and biological descriptors spanning levels **A–E**, with **128 dimensions per sublevel**.

- Features are fetched via the Chemical Checker API.
- Raw CC signatures are assembled into model-ready matrices.
- Final CC feature files are stored under:
`data/features/chemicalchecker_cc/`
including:
- `cc_features_raw.csv` (raw per-level signatures)
- `cc_features_concat_25x128.csv`
- `cc_features_concat_15x128.csv`

2. **Strain-space (S-space) features**  
Data-driven embeddings derived from **drug–strain fitness profiles**.

- Constructed using the Chemical Checker signature protocol:
- type-0 (sign0) → type-I (sign1) → type-II (sign2)
- The final 128-dimensional strain-space embeddings are used directly in modeling.
- Intermediate and final artifacts are managed under:
`feature_pipeline/strain_space/`

Detailed documentation for both pipelines is provided in:
- `feature_pipeline/chemicalchecker/README.md`
- `feature_pipeline/strain_space/README.md`

---

## Setup

### 1) Create environment
```bash
conda create -n halo python=3.10
conda activate halo
```

### 2) Install HALO
from repository root:
```bash
python -m pip install -e .
```
after installation, import such as:
```python
from halo.paths import CC_FEATURES
```
should work without notifying `sys.path`.

### 3) Install core dependencies
```bash
python -m pip install pandas numpy scikit-learn lightgbm requests 
```

### Chemical Checker
HALO uses the `Chemical Checker` for two purposes:
1) Fetching Chemical Checker (CC) signatures via the CC API as drug features
2) Building the **Strain-space (S-space)** feature set by using the python `chemicalchecker` package: uses the Chemical Checker pipeline to construct an additional feature set: Strain-space, derived from single-drug bacterial fitness profiles.

#### Install
You must install it separately:
```bash
python -m pip install chemicalchecker
```

**Configure CC (required):**
Set `CC_CONFIG` to point to a valid config before importing `chemicalchecker`:
```bash
export CC_CONFIG=$(pwd)/feature_pipeline/chemicalchecker/cc_config.json
```

Validate the config is valid JSON:
```bash
python -m json.tool "$CC_CONFIG"
```

A reference configuration used in this study is included in:
`feature_pipeline/chemicalchecker/cc_config.json`
Users must adapt paths to their local setup.

---

## Reproducibility notes
- Generated outputs (cached Chemical Checker artifacts, model results, figures) are not required to run the pipeline and may be excluded from version control.

- Scripts and notebooks assume paths relative to the repository root.

- Large external datasets and proprietary databases are not redistributed and must be obtained independently.

---

## Citation
If you use this codebase, please cite the associated manuscript (details to be added upon publication).

---

## Contact
For questions or issues related to this repository, please open a GitHub issue or contact the authors.
