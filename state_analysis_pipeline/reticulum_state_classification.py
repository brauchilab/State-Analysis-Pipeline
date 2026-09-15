#!/usr/bin/env python
# reticulum_state_classification.py
"""
Step 3: ER (C1 channel) state classification.

For each cell:
  - builds the ER binary mask from the deconvolved C1 image
  - ANDs in the hand-drawn cell membrane mask (cell_coords.csv, from a
    separate manual tool), if present
  - computes normalized intensity (dF/F0 or log2(F/F0), configurable),
    excluding dark frames from the baseline window
  - handles outliers (configurable: clip, or trim by IQR/percentile/
    zscore), using the log2-safe trimming functions when normalizing
    with log2
  - samples points of interest, excluding dark frames from both the
    valid-frame-fraction threshold and the training data
  - trains the HMM and classifies each pixel's state probabilities
  - saves the state probabilities to an npz

Stops here -- ring/lysosome correlation analysis from the original
pipeline is not part of this step.

Bright frames are logged (bright_frames.json) but not given any special
treatment here, same as lysosome tracking. See future_considerations.md.

Run from the folder containing inputs/ and outputs/:
    python reticulum_state_classification.py                 # all cells
    python reticulum_state_classification.py 010519-Cell2     # one cell
"""

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from matplotlib.path import Path as MplPath

from reticulum_mask_utils import c1_dec_mask_raw, _greedy_maximal_spacing
from calc_diff_cube import calcDiffCube as calcdiff_dff, calcDiffCube_log2 as calcdiff_log2
from tifffun import (
    remove_positive_outliers_iqr as remove_outliers_iqr_dff,
    remove_positive_outliers_percentile as remove_outliers_percentile_dff,
    remove_positive_outliers_zscore as remove_outliers_zscore_dff,
)
from tifffun_log2 import (
    remove_positive_outliers_iqr_log2,
    remove_positive_outliers_percentile_log2,
    remove_positive_outliers_zscore_log2,
)
from simple_markov import run_estimation
from prob_an import predict_states_probabilities_vectorized
from state_probs_io import quick_save
from cell_data_io import find_cell_data
from progress_tracker import update_step, is_step_complete

OUTPUTS_DIR = Path("./outputs")

STEP_NAME = "reticulum_state_classification"
FORCE_RERUN = False

# "dff" -> (F-F0)/F0. "log2" -> log2(F/F0).
NORMALIZATION_METHOD = "dff"

# "clip": np.clip(result, -1, 1). Arbitrary for log2 data, since log2 has
# no real floor at -1 the way dF/F0 does -- treat those bounds as
# arbitrary if used with log2.
# "remove_iqr" / "remove_percentile" / "remove_zscore": trim extreme
# values instead of hard clipping. "none": leave as-is.
OUTLIER_HANDLE = "remove_percentile"

MAX_POIS = 1000
MIN_VALID_FRACTION = 0.8
POI_FALLBACK_FRACTIONS = [0.7, 0.6, 0.5]
POI_RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# cell membrane mask
# ---------------------------------------------------------------------------

def load_cell_coords(celloutdir):
    """Manually drawn cell-border polygon from cell_coords.csv, or None."""
    coord_file = Path(celloutdir) / "cell_coords.csv"
    if not coord_file.exists():
        return None
    return np.loadtxt(coord_file, delimiter=",", skiprows=1)  # shape (N, 2), x, y


def coords_to_mask(coords, image_shape):
    height, width = image_shape
    ys, xs = np.mgrid[0:height, 0:width]
    pixel_coords = np.column_stack([xs.ravel(), ys.ravel()])
    inside = MplPath(coords).contains_points(pixel_coords)
    return inside.reshape(height, width)


def apply_cell_mask_to_er_mask(celloutdir, image_shape):
    """
    Zero out pixels outside the cell membrane in c1_raw_masked.tif, which
    c1_dec_mask_raw already produced from the ER-only mask. Zeroed pixels
    are treated as invalid by the POI sampler downstream, so they're
    excluded from HMM training and classification automatically.

    Returns the cell mask, or None if cell_coords.csv wasn't found (the
    ER-only mask is left as-is in that case).
    """
    coords = load_cell_coords(celloutdir)
    if coords is None:
        print("  no cell_coords.csv found, cell membrane not excluded")
        return None

    cell_mask = coords_to_mask(coords, image_shape)  # True = inside cell

    raw_masked_path = Path(celloutdir) / "c1_raw_masked.tif"
    raw_masked = tifffile.imread(raw_masked_path)  # (T, H, W)
    raw_masked[:, ~cell_mask] = 0
    tifffile.imwrite(raw_masked_path, raw_masked, imagej=True)

    inside_px = cell_mask.sum()
    print(f"  cell mask applied: {inside_px}/{cell_mask.size} pixels inside "
          f"({100 * inside_px / cell_mask.size:.1f}%)")
    return cell_mask


# ---------------------------------------------------------------------------
# dark frame lookup (bright frames are logged elsewhere but not used here,
# see future_considerations.md)
# ---------------------------------------------------------------------------

def get_dark_frame_indices(cell_name):
    dark_path = OUTPUTS_DIR / cell_name / "dark_frames.json"
    if not dark_path.exists():
        return None
    with open(dark_path) as f:
        data = json.load(f)
    indices = data.get("dark_frame_indices", [])
    return indices if indices else None


# ---------------------------------------------------------------------------
# normalization + outlier handling
# ---------------------------------------------------------------------------

def compute_normalized_intensity(c1_img_path, dark_frame_indices):
    if NORMALIZATION_METHOD == "log2":
        return calcdiff_log2(c1_img_path, dark_frame_indices=dark_frame_indices)
    return calcdiff_dff(c1_img_path, dark_frame_indices=dark_frame_indices)


def apply_outlier_handling(dff_result):
    if OUTLIER_HANDLE == "clip":
        return np.clip(dff_result, -1, 1)

    if OUTLIER_HANDLE == "none":
        return dff_result

    if NORMALIZATION_METHOD == "log2":
        removers = {
            "remove_iqr": lambda d: remove_positive_outliers_iqr_log2(d, iqr_multiplier=1.5),
            "remove_percentile": lambda d: remove_positive_outliers_percentile_log2(d, upper_percentile=99),
            "remove_zscore": lambda d: remove_positive_outliers_zscore_log2(d, z_threshold=3),
        }
    else:
        removers = {
            "remove_iqr": lambda d: remove_outliers_iqr_dff(d, iqr_multiplier=1.5),
            "remove_percentile": lambda d: remove_outliers_percentile_dff(d, upper_percentile=99),
            "remove_zscore": lambda d: remove_outliers_zscore_dff(d, z_threshold=3),
        }

    remover = removers.get(OUTLIER_HANDLE)
    return remover(dff_result) if remover else dff_result


# ---------------------------------------------------------------------------
# POI sampling, dark frames excluded from thresholds and training data
# ---------------------------------------------------------------------------

def sample_pois_quality_distributed(dff_results, output_csv, dark_frame_indices=None):
    frames, height, width = dff_results.shape

    dark_set = set(dark_frame_indices) if dark_frame_indices else set()
    valid_frame_mask = np.array([f not in dark_set for f in range(frames)])
    n_valid_frames = int(valid_frame_mask.sum())

    if dark_set:
        print(f"  excluding {len(dark_set)} dark frame(s) from POI thresholds and training CSV "
              f"({frames} -> {n_valid_frames} valid frames)")

    if POI_RANDOM_SEED is not None:
        np.random.seed(POI_RANDOM_SEED)

    # frames-per-pixel still runs over the full stack -- dark frames
    # contribute 0 either way since they're all-zero. what changes is the
    # threshold below, now relative to n_valid_frames
    valid_frames_per_pixel = np.sum((dff_results != 0) & ~np.isnan(dff_results), axis=0)

    thresholds_to_try = [MIN_VALID_FRACTION] + POI_FALLBACK_FRACTIONS
    candidate_coords = None
    for threshold in thresholds_to_try:
        required_frames = int(n_valid_frames * threshold)
        valid_pixels_mask = valid_frames_per_pixel >= required_frames
        candidate_coords = np.argwhere(valid_pixels_mask)
        print(f"  threshold {threshold * 100:.0f}% (>={required_frames} frames): "
              f"{len(candidate_coords)} candidates")
        if len(candidate_coords) >= MAX_POIS:
            break

    if len(candidate_coords) == 0:
        print("  ERROR: no valid candidates found even with relaxed thresholds")
        return []

    if len(candidate_coords) <= MAX_POIS:
        selected_pois = candidate_coords.tolist()
    else:
        selected_pois = _greedy_maximal_spacing(candidate_coords, MAX_POIS)
    print(f"  selected {len(selected_pois)} POI(s)")

    # dark frame rows are dropped entirely, not just zeroed, so the
    # synthetic all-zero block never reaches HMM training
    out_frame_indices = np.where(valid_frame_mask)[0] if dark_set else np.arange(frames)

    data = {"frame": out_frame_indices}
    for (y, x) in selected_pois:
        data[f"({y},{x})"] = dff_results[out_frame_indices, y, x]
    pd.DataFrame(data).to_csv(output_csv, index=False)
    print(f"  saved POIs to {output_csv}")

    return selected_pois


# ---------------------------------------------------------------------------
# per-cell pipeline
# ---------------------------------------------------------------------------

def classify_cell_states(celloutdir, dark_frame_indices):
    c1_img_path = str(Path(celloutdir) / "c1_raw_masked.tif")

    dff_result = compute_normalized_intensity(c1_img_path, dark_frame_indices)
    dff_result = np.nan_to_num(dff_result, nan=0.0)
    dff_result_used = apply_outlier_handling(dff_result)

    pois_csv = str(Path(celloutdir) / "POIs.csv")
    sample_pois_quality_distributed(dff_result_used, pois_csv, dark_frame_indices)

    params = run_estimation(pois_csv, output_file=str(Path(celloutdir) / "trained_params.txt"),
                             version="C1")
    state_means = params["trained_params_C1"]["emission_means"]
    state_stds = params["trained_params_C1"]["emission_stds"]

    return predict_states_probabilities_vectorized(dff_result_used, state_means, state_stds)


def process_cell(currcell):
    cell_name = currcell.get("cell_name", "UNKNOWN_CELL")

    if not FORCE_RERUN and is_step_complete(OUTPUTS_DIR, cell_name, STEP_NAME):
        print(f"[skip] {cell_name}: state classification already done")
        return None

    print(f"[cell] {cell_name}")

    c1_img_path = currcell["c1_file_path"]
    deconvs = find_cell_data(currcell["cell_folder_path"])
    c1_deconv_path = deconvs[0]["c1_file_path"]

    celloutdir = OUTPUTS_DIR / cell_name
    celloutdir.mkdir(parents=True, exist_ok=True)

    c1_dec_mask_raw(c1_deconv_path, c1_img_path, str(celloutdir))

    raw_shape_hw = tifffile.imread(c1_img_path).shape[-2:]
    apply_cell_mask_to_er_mask(str(celloutdir), raw_shape_hw)

    dark_frame_indices = get_dark_frame_indices(cell_name)
    state_probs = classify_cell_states(str(celloutdir), dark_frame_indices)

    quick_save(state_probs, cell_name, str(celloutdir), method="global")

    return {
        "normalization_method": NORMALIZATION_METHOD,
        "outlier_handle": OUTLIER_HANDLE,
        "n_dark_frames_excluded": len(dark_frame_indices) if dark_frame_indices else 0,
    }


def run(cell_names=None):
    """Runs this step for the given cell names, or every cell find_cell_data()
    returns if cell_names is None."""
    OUTPUTS_DIR.mkdir(exist_ok=True)
    logfile = OUTPUTS_DIR / "error_log_reticulum_state_classification.txt"
    with open(logfile, "w") as f:
        f.write(f"Error log started at {datetime.now()}\n\n")

    cells2process = find_cell_data()
    if cell_names:
        requested = set(cell_names)
        run_queue = [c for c in cells2process if c["cell_name"] in requested]
        missing = requested - {c["cell_name"] for c in cells2process}
        if missing:
            print(f"WARNING: cell name(s) not found: {sorted(missing)}")
    else:
        run_queue = cells2process

    print(f"Processing {len(run_queue)} cell(s), normalization={NORMALIZATION_METHOD}, "
          f"outlier_handle={OUTLIER_HANDLE}.")

    error_cells = []
    for currcell in run_queue:
        cell_name = currcell.get("cell_name", "UNKNOWN_CELL")
        try:
            notes = process_cell(currcell)
            if notes is not None:
                update_step(OUTPUTS_DIR, cell_name, STEP_NAME, "complete", notes)
                print(f"  [done] {cell_name}")
        except Exception as e:
            tb = traceback.format_exc()
            error_cells.append(cell_name)
            print(f"  [error] {cell_name}: {e}")
            update_step(OUTPUTS_DIR, cell_name, STEP_NAME, "error", {"message": str(e)})
            with open(logfile, "a") as f:
                f.write(f"Cell: {cell_name}\nError: {e}\n{tb}\n{'-' * 60}\n\n")

    print(f"\nDone. {len(run_queue) - len(error_cells)}/{len(run_queue)} cell(s) succeeded.")
    if error_cells:
        print(f"Failed: {error_cells}\nSee {logfile}")


def main():
    run(sys.argv[1:] or None)


if __name__ == "__main__":
    main()
