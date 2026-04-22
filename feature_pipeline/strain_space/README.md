# Strain-space (S-space) feature pipeline

This directory contains the pipeline used to construct **strain-space (S-space)** drug embeddings from single-drug bacterial fitness profiles, following the Chemical Checker signature protocol (sign0 → sign1 → sign2).

The final output is a **128-dimensional embedding per compound**, used as an additional feature space in HALO modeling.

---

## Directory layout

- `inputs/`  
  Staged input tables used to build S-space (raw exports → cleaned matrices → merged fitness table).

- `cache/`  
  Chemical Checker working directory and intermediate artifacts generated during S-space construction.

- `notebooks/`  
  Notebooks used to clean fitness datasets and run S-space construction (recommended order below).

---

## Outputs

- `data/features/strain_space_ss/S_sign2.tsv`  
  Final S-space embeddings (sign2), one row per compound (InChIKey) with 128 features.

- `data/features/strain_space_ss/sspace.csv`  
  Post-processed version of `S_sign2.tsv` with standardized column names (`s_0` … `s_127`) and numeric scaling checks (produced by `04_prepare_strain_space_features.ipynb`).

---

## Run order (recommended)

### 00_clean_brochado.ipynb
**Goal:** Standardize the Brochado single-treatment fitness table and export a cleaned wide matrix.  
**Reads:** `inputs/stage0/before_raw_brochado.csv`  
**Writes:** `inputs/stage1/raw_brochado_fitness.csv`

### 01_clean_cacace.ipynb
**Goal:** Aggregate and harmonize the Cacace donor/recipient fitness tables and export a cleaned wide matrix.  
**Reads:**
- `inputs/stage0/before_raw_cacace_donor.csv`
- `inputs/stage0/before_raw_cacace_recipient.csv`  
**Writes:** `inputs/stage1/raw_cacace_fitness.csv`  
**Notes:** Averages fitness across concentrations/replicates per (strain, drug). Merges donor/recipient views and uses the mean where available.

### 02_preprocess_fitness_data.ipynb
**Goal:** Merge Brochado + Cacace fitness tables into one unified fitness table keyed by `inchikey`.  
**Reads:**
- `inputs/stage1/raw_brochado_fitness.csv`
- `inputs/stage1/raw_cacace_fitness.csv`
- `data/reference/drug_lists/list_antibacterial.csv`
- `data/reference/drug_lists/3_letter_code_cacace.csv`  
**Writes:** `inputs/stage1/raw_fitness.csv`  
**Notes:** Maps Cacace 3-letter codes → drug names → InChIKeys using curated reference tables.

### 03_build_sspace.ipynb
**Goal:** Build S-space signatures using Chemical Checker and export sign2 embeddings.  
**Reads:** `inputs/stage1/raw_fitness.csv`  
**Writes (cache):** `cache/S1.001/` (intermediate CC artifacts)  
**Writes (final):** `data/features/strain_space_ss/S_sign2.tsv`  
**Notes:** Uses dataset identifier `S1.001`. Some CC diagnostics may be skipped for small graphs; sign2 generation still proceeds.

### 04_prepare_strain_space_features.ipynb
**Goal:** Convert `S_sign2.tsv` into a model-ready table and validate scaling / formatting.  
**Reads:** `data/features/strain_space_ss/S_sign2.tsv`  
**Writes:** `data/features/strain_space_ss/sspace.csv`  
**Notes:** Renames feature columns to `s_0` … `s_127` and standardizes InChIKey formatting.

---

## Prerequisites (Chemical Checker)

This pipeline requires the Python `chemicalchecker` package and a valid `CC_CONFIG`.

- Install: `python -m pip install chemicalchecker`
- Set config before importing `chemicalchecker`:
  ```bash
  export CC_CONFIG=$(pwd)/feature_pipeline/chemicalchecker/cc_config.json
  ```

The reference config lives at `feature_pipeline/chemicalchecker/cc_config.json`.
See the repository root README for full setup details.

