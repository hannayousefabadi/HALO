# Drug reference lists

This directory contains manually curated reference tables used during
data preprocessing, drug name normalization, and feature construction.

These files are **not model outputs**. They encode curation decisions
made to harmonize drug identifiers across heterogeneous sources.

---

## Files

### `3_letter_code_cacace.csv`
Mapping table for the Cacace et al. dataset.

- Maps the **3-letter drug codes** used in the Cacace study
  to full drug names and/or standard identifiers.
- Used during preprocessing to normalize drug identifiers
  before integration with other datasets.

---

### `alternative_names.csv`
Table of alternative / synonymous drug names.

- Maps multiple generic names, aliases, or spelling variants to the same **InChIKey**.
- Used to normalize drug names across datasets where the same compound appears under different names.

---

### `list_antibacterial.csv`
Curated list of antibacterial drugs.

- Constructed manually based on FDA-approved drugs and
  literature-supported antibacterials.
- May contain **multiple rows referring to the same compound**
  under different generic names.
- Used when mapping antibacterial annotations to training datasets,
  preserving the original naming used by each source.

---

### `list_antibacterial_for_cc.csv`
Deduplicated antibacterial list for Chemical Checker queries.

- Derived from `list_antibacterial.csv` by de-duplicating by InChIKey 
  in ` 00_curate_antibacterial_references.ipynb`.
- Each compound appears **only once**, with a unique identifier.
- Used specifically to fetch Chemical Checker features,
  which require unique compound identifiers.

---

## Notes
- These files encode **domain knowledge and curation choices**.
- They are treated as reference inputs and are not regenerated automatically.
- Any changes to these lists may affect downstream preprocessing
  and feature construction steps.
