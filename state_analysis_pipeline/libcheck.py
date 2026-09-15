#!/usr/bin/env python
# libcheck.py
"""
Checks the environment before running the pipeline: standard packages
installed, and custom modules present and importable in this folder.
Not wired into run_pipeline.py, run by hand.

Only imports each module, doesn't touch inputs/outputs.

Run from the folder with the pipeline scripts:
    python libcheck.py
"""

import importlib
import sys

# (import name, pip install name)
STANDARD_PACKAGES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("tifffile", "tifffile"),
    ("skimage", "scikit-image"),
    ("cv2", "opencv-python"),
    ("trackpy", "trackpy"),
    ("hmmlearn", "hmmlearn"),
    ("sklearn", "scikit-learn"),
    ("tqdm", "tqdm"),
    ("plotly", "plotly"),
]

# custom modules, base/shared ones first so a failure higher up is
# likely the real cause
CUSTOM_MODULES = [
    "progress_tracker",
    "frame_intensity_utils",
    "cell_data_io",
    "tv_denoise",
    "atrouse_algorithm",
    "dogmasks_v3",
    "filtering_dff",
    "graphing_funcs",
    "tifffun",
    "tifffun_log2",
    "calc_diff_cube",
    "reticulum_mask_utils",
    "simple_markov",
    "prob_an",
    "state_probs_io",
    "fiji_export_utils",
    "get_data_and_artifacts",
    "deconvolution",
    "lysosome_tracking",
    "reticulum_state_classification",
    "lysosome_state_classification",
    "export_state_videos",
    "run_pipeline",
]

# standalone, not required by run_pipeline.py
OPTIONAL_MODULES = [
    "lysosome_reticulum_overlay",
]


def check_standard_packages():
    print("Standard/third-party packages")
    print("-" * 60)
    missing = []
    for import_name, pip_name in STANDARD_PACKAGES:
        try:
            mod = importlib.import_module(import_name)
        except ImportError as e:
            print(f"  MISS  {import_name:<12} ({pip_name}) -- {e}")
            missing.append(pip_name)
            continue

        version = getattr(mod, "__version__", None)
        if version is None:
            try:
                from importlib.metadata import version as get_version
                version = get_version(pip_name)
            except Exception:
                version = "unknown version"
        print(f"  OK    {import_name:<12} ({pip_name}) {version}")

    return missing


def check_modules(module_names, label):
    print(f"\n{label}")
    print("-" * 60)
    missing_files = []
    broken = []

    for name in module_names:
        try:
            importlib.import_module(name)
            print(f"  OK    {name}")
        except ModuleNotFoundError as e:
            if e.name == name:
                print(f"  MISS  {name}  -- {name}.py not found in this folder")
                missing_files.append(name)
            else:
                print(f"  FAIL  {name}  -- needs '{e.name}', which is missing")
                broken.append((name, f"needs '{e.name}', which is missing"))
        except Exception as e:
            print(f"  FAIL  {name}  -- {type(e).__name__}: {e}")
            broken.append((name, f"{type(e).__name__}: {e}"))

    return missing_files, broken


def main():
    print("=" * 60)
    print("Pipeline environment check")
    print("=" * 60)

    missing_packages = check_standard_packages()
    missing_custom, broken_custom = check_modules(CUSTOM_MODULES, "Custom pipeline modules")
    missing_optional, broken_optional = check_modules(
        OPTIONAL_MODULES, "Optional modules (not required by run_pipeline.py)"
    )

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    if not missing_packages and not missing_custom and not broken_custom:
        print("Everything the pipeline needs is present and importable.")
    else:
        if missing_packages:
            print("\nMissing package(s), install with:")
            print(f"  pip install {' '.join(missing_packages)}")
        if missing_custom:
            print("\nMissing custom module file(s) -- copy these into this folder:")
            for name in missing_custom:
                print(f"  {name}.py")
        if broken_custom:
            print("\nCustom module(s) present but failed to import:")
            for name, reason in broken_custom:
                print(f"  {name}: {reason}")

    if missing_optional or broken_optional:
        print("\n(optional, not required by run_pipeline.py)")
        for name in missing_optional:
            print(f"  missing: {name}.py")
        for name, reason in broken_optional:
            print(f"  broken: {name}: {reason}")

    any_problem = bool(missing_packages or missing_custom or broken_custom)
    sys.exit(1 if any_problem else 0)


if __name__ == "__main__":
    main()
