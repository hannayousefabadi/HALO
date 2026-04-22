# src/paths.py
from pathlib import Path

# HALO root
def _find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for parent in (p, *p.parents):
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    return p.parents[2]

ROOT = _find_repo_root(Path(__file__))

DATA = ROOT / "data"
RAW = DATA / "a_raw"
EXTRACTED = DATA / "b_extracted"
INTERIM = DATA / "c_interim"
PROCESSED = DATA / "d_processed"

FEATURES = DATA / "features"
CC_FEATURES = FEATURES / "chemicalchecker_cc"
SS_FEATURES = FEATURES / "strain_space_ss"

REFERENCE = DATA / "reference"
DRUG_LISTS = REFERENCE / "drug_lists"

RESULTS = ROOT / "results"
MODEL_RESULTS = RESULTS / "models"

FEATURE_PIPELINE = ROOT / "feature_pipeline"
FIGURE_PIPELINE = ROOT / "figure_pipeline"
FIGURES = ROOT / "figures"
