import runpy
from halo.paths import ALL_MODELS

TARGET = (
    ALL_MODELS
    / "exp06e_lgbm_bin_nosspace_elementwise_preCV_reduced_nestedcv"
    / "exp06e_lgbm_bin_nosspace_elementwise_preCV_reduced_nestedcv.py"
)

if __name__ == "__main__":
    if not TARGET.exists():
        raise FileNotFoundError(f"Target script not found: {TARGET}")
    runpy.run_path(str(TARGET), run_name="__main__")


