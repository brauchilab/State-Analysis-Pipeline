#!/usr/bin/env python
# export_state_videos.py
"""
Step 5 (optional): colored state videos, for visual inspection.

  C1 (ER): one video, 3 colors for 3 states (blue/green/red), background
    or invalid = black. Uses whichever normalization method (dF/F0 or
    log2) was actually used for that cell in reticulum_state_classification,
    read from progress.json -- the pipeline runs one method per cell, not
    both in parallel.
  C2 (lysosome): one video, 2 colors for 2 states (magenta/yellow),
    background = black. Requires the lysosome state classification CSV
    (particle, frame, predicted_state); if missing, that cell's C2 export
    is skipped with a note, without stopping the rest.

Output per cell:
    fiji_export/<cell>/C1_state.tif
    fiji_export/<cell>/C2_state.tif

Run standalone, or with RUN_EXPORT_VIDEOS from the main pipeline script:
    python export_state_videos.py                 # all cells
    python export_state_videos.py 010519-Cell2     # one cell
"""

import os
import sys
from pathlib import Path

import numpy as np

from state_probs_io import quick_load
from fiji_export_utils import (
    create_simple_state_matrix, save_rgb_tiff,
    load_particle_states, get_mask_files, particle_id_from_mask_path,
    safe_read_mask_stack, get_mask_shape,
)
from progress_tracker import update_step, is_step_complete, load_progress

OUTPUTS_DIR = Path("./outputs")
EXPORT_DIR = Path("./fiji_export")

STEP_NAME = "export_state_videos"
FORCE_RERUN = False

NPZ_METHOD = "global"
STATES_CSV_PATTERN = "*C2*tracks_dff_filt_with_states_ordered.csv"

ER_COLORS = {
    -1: (0, 0, 0),
     0: (0, 0, 255),
     1: (0, 255, 0),
     2: (255, 0, 0),
}
LYSO_STATE_COLORS = {
    0: (255, 0, 255),  # magenta
    1: (255, 255, 0),  # yellow
}


def get_step_normalization_method(cell_name, step_name):
    """Reads which normalization method was actually used for a given
    cell's classification step (reticulum_state_classification or
    lysosome_state_classification). Returns None if that step hasn't
    completed or didn't record one."""
    progress = load_progress(OUTPUTS_DIR, cell_name)
    step = progress.get("steps", {}).get(step_name, {})
    if step.get("status") != "complete":
        return None
    return step.get("notes", {}).get("normalization_method")


def export_c1_state(cell_name, out_dir):
    method = get_step_normalization_method(cell_name, "reticulum_state_classification")
    if method is None:
        print("  [skip] C1: reticulum state classification not completed for this cell")
        return

    celloutdir = OUTPUTS_DIR / cell_name
    npz_path = celloutdir / f"{cell_name}_state_probs_{NPZ_METHOD}.npz"
    if not npz_path.exists():
        print(f"  [skip] C1: missing {npz_path}")
        return

    state_probs_float = quick_load(cell_name, str(celloutdir), method=NPZ_METHOD, return_tuples=False)
    simple = create_simple_state_matrix(state_probs_float)
    F, H, W = simple.shape
    rgb = np.zeros((F, H, W, 3), dtype=np.uint8)
    for state, color in ER_COLORS.items():
        rgb[simple == state] = color
    save_rgb_tiff(rgb, str(out_dir / "C1_state.tif"))
    print(f"  saved C1_state.tif (normalization: {method})")


def export_c2_state(cell_name, out_dir):
    method = get_step_normalization_method(cell_name, "lysosome_state_classification")

    celloutdir = OUTPUTS_DIR / cell_name
    masks_dir = celloutdir / "individual_masks"

    particle_states = load_particle_states(str(celloutdir), STATES_CSV_PATTERN)
    if particle_states is None:
        print(f"  [skip] C2: no lysosome state classification CSV found "
              f"(pattern: {STATES_CSV_PATTERN})")
        return

    mask_files = get_mask_files(str(masks_dir))
    if not mask_files:
        print(f"  [skip] C2: no individual_masks/ in {celloutdir}")
        return

    shape, used_path = get_mask_shape(mask_files)
    if shape is None:
        print(f"  [skip] C2: no readable mask in {masks_dir}")
        return
    F, H, W = shape
    rgb = np.zeros((F, H, W, 3), dtype=np.uint8)

    for mask_path in mask_files:
        particle_id = particle_id_from_mask_path(mask_path)
        frame_states = particle_states.get(particle_id)
        if frame_states is None:
            continue
        mm = safe_read_mask_stack(mask_path, expected_shape=(F, H, W))
        if mm is None:
            continue
        n_frames_avail = min(F, mm.shape[0])
        for f in range(n_frames_avail):
            if f not in frame_states:
                continue
            frame_mask = mm[f] > 0
            if not frame_mask.any():
                continue
            color = LYSO_STATE_COLORS.get(int(frame_states[f]))
            if color is None:
                continue
            rgb[f][frame_mask] = color

    save_rgb_tiff(rgb, str(out_dir / "C2_state.tif"))
    method_note = f", normalization: {method}" if method else ""
    print(f"  saved C2_state.tif ({len(particle_states)} particle(s){method_note})")


def process_cell(cell_name):
    if not FORCE_RERUN and is_step_complete(OUTPUTS_DIR, cell_name, STEP_NAME):
        print(f"[skip] {cell_name}: state videos already exported")
        return None

    print(f"[cell] {cell_name}")
    out_dir = EXPORT_DIR / cell_name
    out_dir.mkdir(parents=True, exist_ok=True)

    export_c1_state(cell_name, out_dir)
    export_c2_state(cell_name, out_dir)
    return {}


def run(cell_names=None):
    """Runs this step for the given cell names, or every cell in outputs/
    if cell_names is None."""
    if not OUTPUTS_DIR.is_dir():
        raise SystemExit(f"outputs dir not found: {OUTPUTS_DIR.resolve()}")
    EXPORT_DIR.mkdir(exist_ok=True)

    if not cell_names:
        cell_names = sorted(d.name for d in OUTPUTS_DIR.iterdir() if d.is_dir())

    print(f"Exporting state videos for {len(cell_names)} cell(s).")

    for cell_name in cell_names:
        try:
            notes = process_cell(cell_name)
            if notes is not None:
                update_step(OUTPUTS_DIR, cell_name, STEP_NAME, "complete", notes)
        except Exception as e:
            print(f"  [error] {cell_name}: {e}")
            update_step(OUTPUTS_DIR, cell_name, STEP_NAME, "error", {"message": str(e)})

    print("\nDone.")


def main():
    run(sys.argv[1:] or None)


if __name__ == "__main__":
    main()
