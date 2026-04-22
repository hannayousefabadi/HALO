# ChemicalChecker dependency

This directory documents the external **ChemicalChecker** dependency used in this project.

ChemicalChecker is **not included** in this repository and must be installed separately.

---

## Installation

Install ChemicalChecker from the official repository:

https://github.com/sbnb-irb/chemical_checker/

Follow the installation instructions provided by the ChemicalChecker authors, including database setup.

---

## Contents of this directory

This directory contains two files related to ChemicalChecker usage in this project:

- `00_fetch_cc_features.py`  
  Script used to fetch **original ChemicalChecker (CC) features** via the CC API.  
  The output is stored in `data/features/chemicalchecker_cc/cc_features_raw.csv`.

  This file has the following structure:
  - rows: drugs (indexed by InChIKey)
  - columns:
    - `level`: CC sublevel identifier (`A1`–`E5`, 25 total)
    - `dim_0` … `dim_127`: 128-dimensional signature for that level

- `01_prepare_cc_features.ipynb`
  Notebook that **filters and reshapes** the raw CC export (`cc_features_raw.csv`) into fixed-length feature tables used by the HALO models.

  What it does:
  - Drops rows with missing values in critical fields (`drug`, `inchikey`, `level`, and the 128 signature dimensions).
  - Keeps only compounds with **complete coverage** of the required CC sublevels:
    - **25/25 sublevels** for the full CC feature set (A1–E5)
    - **15/15 sublevels** for the reduced CC feature set (A1–C5)
  - Concatenates 128-d vectors across sublevels to produce one vector per compound.

  Outputs:
  - `cc_features_concat_25x128.csv`  
    One row per compound; **3200-d** feature vector (25 × 128), concatenated in sublevel order **A1 → E5**.
  - `cc_features_concat_15x128.csv`  
    One row per compound; **1920-d** feature vector (15 × 128), concatenated in sublevel order **A1 → C5**.

  Notes:
  - InChIKeys are standardized to **uppercase**.
  - These files are treated as **reference feature inputs** for modeling (they are not learned artifacts).

- `cc_config.json`  
Reference ChemicalChecker configuration file used during strain-space construction  
in `feature_pipeline/strain_space/notebooks/03_build_sspace.ipynb`.

---

## `cc_config.json`

ChemicalChecker requires a configuration file (`cc_config.json`) that defines local paths to:
- ChemicalChecker databases
- temporary files
- log directories

These paths are **machine-specific**.

We include `cc_config.json` **only as a reference configuration used in this study**.  
Users must adapt the paths and set `CC_CONFIG` to point to a local copy.

---

## Required environment variable

Before running any ChemicalChecker-based step, set:

```bash
export CC_CONFIG=/absolute/path/to/your/cc_config.json
```
