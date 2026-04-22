# Dataflow

This document describes how raw source data becomes the final modeling datasets and feature matrices in HALO.

---

## Sources

- **source_a_brochado**: Brochado et al. (training)
- **source_b_cacace**: Cacace et al. (training)
- **source_c_acdb**: ACDB (training)
- **source_d_chandrasekaran**: Chandrasekaran et al. (external validation)

---

## Data stages (data/)

HALO uses a simple staged layout:

- **a_raw/**  
  Original downloads (as obtained from the source).

- **b_extracted/**  
  Manually/locally extracted tables from raw (same content, smaller/cleaner format).

- **c_interim/**  
  Per-source cleaned datasets (standardized columns, cleaned labels, mapped identifiers).

- **d_processed/**  
  Final integrated datasets used by models.

---

## Training dataset (pair synergy)

### Notebooks
Located under:
- `notebooks/preprocessing/`

Recommended run order:
1. `00_curate_antibacterial_references.ipynb`  
   Curates reference lists used for mapping and filtering (e.g., antibacterial drug lists).

2. `01_build_interim_datasets.ipynb`  
   Cleans each source independently and writes per-source outputs to `data/c_interim/`.

3. `02_build_processed_dataset.ipynb`  
   Merges the interim sources, resolves duplicates/inconsistencies, assigns labels, and writes the final training dataset.

### Final output
- `data/d_processed/halo_training_dataset.csv`  
  Canonical merged dataset (pair × strain samples) used for modeling.

---

## Chemical Checker (CC) features

Pipeline:
- `feature_pipeline/chemicalchecker/00_fetch_cc_features.py`  
  Writes: `data/features/chemicalchecker_cc/cc_features_raw.csv`

- `feature_pipeline/chemicalchecker/01_prepare_cc_features.ipynb`  
  Writes:
  - `data/features/chemicalchecker_cc/cc_features_concat_25x128.csv`
  - `data/features/chemicalchecker_cc/cc_features_concat_15x128.csv`

---

## Strain-space (S-space) features

Pipeline notebooks:
- `feature_pipeline/strain_space/notebooks/00_clean_brochado.ipynb`
- `feature_pipeline/strain_space/notebooks/01_clean_cacace.ipynb`
- `feature_pipeline/strain_space/notebooks/02_preprocess_fitness_data.ipynb`
- `feature_pipeline/strain_space/notebooks/03_build_sspace.ipynb`
- `feature_pipeline/strain_space/notebooks/04_prepare_strain_space_features.ipynb`

Final outputs:
- `data/features/strain_space_ss/S_sign2.tsv` (raw CC output)
- `data/features/strain_space_ss/sspace.csv` (final 128-d features, `inchikey` + `s_0..s_127`)

---

## Results and figures

- **Model artifacts (selected, figure-critical):** `results/models/...`
- **Figure panels (working assets):** `figure_pipeline/**/fig*_panels/` (may include schematic components)
- **Final assembled figures:** `figures/main/` and `figures/supplementary/`
