#!/usr/bin/env python
# run_pipeline.py
"""
Main pipeline driver. Runs steps 0-4 (and optionally step 5) for one or
more cells, in order, with resume support via each step's progress.json.

Must live in the same folder as the other step scripts (get_data_and_
artifacts.py, deconvolution.py, lysosome_tracking.py, reticulum_state_
classification.py, lysosome_state_classification.py, export_state_
videos.py, progress_tracker.py, frame_intensity_utils.py,
treatment_frames.json, pipeline_config.json), run from the folder
containing inputs/.

Edit pipeline_config.json, then run this as a single cell, or:
    python run_pipeline.py                 # cells from pipeline_config.json
    python run_pipeline.py 010519-Cell2     # one cell, overrides the config file
"""

import json
import sys
from pathlib import Path

import get_data_and_artifacts as step0_data_and_artifacts
import deconvolution as step1_deconvolution
import lysosome_tracking as step2_lysosome_tracking
import reticulum_state_classification as step3_reticulum_states
import lysosome_state_classification as step4_lysosome_states
import export_state_videos as step5_export_videos

CONFIG_PATH = Path("./pipeline_config.json")

DEFAULT_CONFIG = {
    "cell_names": None,               # null/None = every cell found, or a list of names
    "force_rerun_all": False,         # ignore progress.json, redo every step
    "run_deconvolution": True,
    "deconv_c1_iterations": 30,
    "deconv_c2_iterations": 20,
    "deconv_c2_denoise_weight": 0.01,
    "reticulum_normalization_method": "dff",       # "dff" or "log2"
    "reticulum_outlier_handle": "remove_percentile",  # clip, remove_iqr, remove_percentile, remove_zscore, none
    "lysosome_normalization_method": "dff",        # "dff" or "log2"
    "run_export_videos": False,
}


def load_config():
    """Reads pipeline_config.json, filling in any missing key from
    DEFAULT_CONFIG. Falls back to DEFAULT_CONFIG entirely if the file
    doesn't exist."""
    if not CONFIG_PATH.exists():
        print(f"no {CONFIG_PATH} found, using built-in defaults")
        return dict(DEFAULT_CONFIG)

    with open(CONFIG_PATH) as f:
        user_config = json.load(f)

    config = dict(DEFAULT_CONFIG)
    config.update(user_config)

    unknown_keys = set(user_config) - set(DEFAULT_CONFIG)
    if unknown_keys:
        print(f"warning: {CONFIG_PATH} has unrecognized key(s), ignored: {sorted(unknown_keys)}")

    return config


def configure_steps(config):
    steps = [
        step0_data_and_artifacts,
        step1_deconvolution,
        step2_lysosome_tracking,
        step3_reticulum_states,
        step4_lysosome_states,
        step5_export_videos,
    ]
    if config["force_rerun_all"]:
        for step in steps:
            step.FORCE_RERUN = True

    step1_deconvolution.RUN_DECONVOLUTION = config["run_deconvolution"]
    step1_deconvolution.C1_ITERATIONS = config["deconv_c1_iterations"]
    step1_deconvolution.C2_ITERATIONS = config["deconv_c2_iterations"]
    step1_deconvolution.C2_DENOISE_WEIGHT = config["deconv_c2_denoise_weight"]

    step3_reticulum_states.NORMALIZATION_METHOD = config["reticulum_normalization_method"]
    step3_reticulum_states.OUTLIER_HANDLE = config["reticulum_outlier_handle"]

    step4_lysosome_states.NORMALIZATION_METHOD = config["lysosome_normalization_method"]


def run_pipeline(cell_names=None):
    """cell_names overrides pipeline_config.json's cell_names if given
    (e.g. from the command line)."""
    config = load_config()
    configure_steps(config)

    if cell_names is None:
        cell_names = config["cell_names"]

    print("\n=== Step 0: metadata, dark/bright frame detection ===")
    step0_data_and_artifacts.run(cell_names)

    print("\n=== Step 1: deconvolution ===")
    step1_deconvolution.run(cell_names)

    print("\n=== Step 2: lysosome tracking ===")
    step2_lysosome_tracking.run(cell_names)

    print("\n=== Step 3: reticulum state classification ===")
    step3_reticulum_states.run(cell_names)

    print("\n=== Step 4: lysosome state classification ===")
    step4_lysosome_states.run(cell_names)

    if config["run_export_videos"]:
        print("\n=== Step 5: export state videos ===")
        step5_export_videos.run(cell_names)

    print("\nPipeline complete.")


if __name__ == "__main__":
    run_pipeline(sys.argv[1:] or None)
