# Feature pipelines

This directory contains the feature construction workflows used in HALO.
Each subdirectory is self-contained and documented with its own README.

## Pipelines

### `chemicalchecker/`
Fetches raw Chemical Checker (CC) signatures via the CC API and assembles model-ready feature matrices.

Outputs (under `data/features/chemicalchecker_cc/`):
- `cc_features_raw.csv`
- `cc_features_concat_25x128.csv`
- `cc_features_concat_15x128.csv`

See: `feature_pipeline/chemicalchecker/README.md`

### `strain_space/`
Constructs strain-space (S-space) embeddings from drug–strain fitness profiles using the Chemical Checker protocol (sign0 → sign1 → sign2).

Outputs (under `data/features/strain_space_ss/`):
- `S_sign2.tsv`
- `sspace.csv`

See: `feature_pipeline/strain_space/README.md`

## Notes
Both pipelines require the `chemicalchecker` Python package and a valid `CC_CONFIG`.
Installation and configuration are described in the repository root README.
