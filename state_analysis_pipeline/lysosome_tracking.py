#!/usr/bin/env python
# lysosome_tracking.py
"""
Step 2: lysosome detection and tracking on the C2 channel.

For each cell:
  - loads the C2 raw stack and its DoG detection mask
  - excises dark-period frames (dark_frames.json) from the working stack
    entirely, before detection, since a blacked-out frame can't yield a
    real detection regardless of what was physically there
  - bright frames (bright_frames.json) are logged but not treated
    specially -- see future_considerations.md
  - runs trackpy detection + linking, scaling diameter/minmass/search_range
    by the cell's pixel size (relative to the size these were calibrated
    at) so physical particle size/displacement stays consistent across
    cells with different pixel sizes
  - filters particles by a minimum frame count, computes df/f0, saves
    per-particle masks and a tracks CSV, and generates earliest-frame
    preview images

Run from the folder containing inputs/ and outputs/:
    python lysosome_tracking.py                 # all cells
    python lysosome_tracking.py 010519-Cell2     # one cell
"""

import gc
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import trackpy as tp
from scipy.ndimage import label

from dogmasks_v3 import MaskProcessor
from filtering_dff import calculate_dff_for_tracks, filter_particles_by_length, clean_particle_tifs
from cell_data_io import find_cell_data
from graphing_funcs import save_particle_earliest_frame_images
from progress_tracker import update_step, is_step_complete

OUTPUTS_DIR = Path("./outputs")

STEP_NAME = "lysosome_tracking"
FORCE_RERUN = False

# trackingpy() was calibrated (diameter=7, minmass=200, search_range=10) at
# this reference pixel size. Cells with finer pixel sizes get these three
# parameters scaled so a lysosome still spans/moves the same PHYSICAL
# amount. memory is left unscaled.
REFERENCE_PIXEL_SIZE_UM = 0.1
BASE_DIAMETER_PX = 7
BASE_MINMASS = 200
BASE_SEARCH_RANGE_PX = 10
BASE_MEMORY_FRAMES = 3

MIN_FRAME_FRACTION = 0.75  # particle must appear in at least this fraction of kept frames
EARLIEST_FRAME_CIRCLE_RADIUS = 7


# ---------------------------------------------------------------------------
# per-cell metadata / artifact lookups
# ---------------------------------------------------------------------------

def get_cell_metadata(cell_name):
    meta_path = OUTPUTS_DIR / cell_name / "cell_metadata.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        return json.load(f)


def get_pixel_size_um(cell_name):
    meta = get_cell_metadata(cell_name)
    return meta.get("pixel_width") if meta else None


def get_dark_frames(cell_name):
    dark_path = OUTPUTS_DIR / cell_name / "dark_frames.json"
    if not dark_path.exists():
        return []
    with open(dark_path) as f:
        data = json.load(f)
    return data.get("dark_frame_indices", [])


def get_valid_frame_indices(cell_name, total_frames):
    """All original frame indices to keep, excluding dark frames. Cells
    with no dark frames get the full range unchanged."""
    dark_frames = set(get_dark_frames(cell_name))
    if not dark_frames:
        return np.arange(total_frames)
    return np.array([i for i in range(total_frames) if i not in dark_frames])


def get_scaled_lyso_params(cell_name):
    """Scale diameter/minmass/search_range by this cell's pixel size, to
    preserve physical particle size/displacement. Cells without pixel size
    metadata use the unscaled base values."""
    pixel_size_um = get_pixel_size_um(cell_name)

    if pixel_size_um is None:
        print(f"  no pixel size for {cell_name}, using base params "
              f"(diameter={BASE_DIAMETER_PX}, minmass={BASE_MINMASS}, "
              f"search_range={BASE_SEARCH_RANGE_PX})")
        return BASE_DIAMETER_PX, BASE_MINMASS, BASE_SEARCH_RANGE_PX, BASE_MEMORY_FRAMES

    scale = REFERENCE_PIXEL_SIZE_UM / pixel_size_um

    diameter = round(BASE_DIAMETER_PX * scale)
    if diameter % 2 == 0:
        diameter += 1
    diameter = max(3, diameter)

    minmass = max(1, round(BASE_MINMASS * (scale ** 2)))  # scales with area
    search_range = max(1, round(BASE_SEARCH_RANGE_PX * scale))
    memory = BASE_MEMORY_FRAMES

    print(f"  {cell_name}: pixel size {pixel_size_um} um/px, scale {scale:.3f} -> "
          f"diameter={diameter}, minmass={minmass}, search_range={search_range}")

    return diameter, minmass, search_range, memory


# ---------------------------------------------------------------------------
# detection + tracking
# ---------------------------------------------------------------------------

def trackingpy(raw_path, mask_path, csv_out, masks_dir, minframenum,
               diameter, minmass, search_range, memory,
               valid_frame_indices, total_original_frames):
    os.makedirs(masks_dir, exist_ok=True)

    frames = tifffile.imread(raw_path).astype(np.float32)
    masks = tifffile.imread(mask_path).astype(bool)

    # Excise dark-period frames before detection, so trackpy's memory
    # gap-tolerance never has to bridge a genuine acquisition blackout.
    # Everything below works in this truncated index space; real frame
    # numbers are restored when writing the CSV and mask tifs.
    valid_frame_indices = np.asarray(valid_frame_indices)
    n_dark = total_original_frames - len(valid_frame_indices)
    if n_dark:
        frames = frames[valid_frame_indices]
        masks = masks[valid_frame_indices]
        print(f"  excised {n_dark} dark frame(s): working stack now {frames.shape[0]} frames")

    masked_frames = frames * masks

    features = []
    for i, (frame, mask_frame) in enumerate(zip(masked_frames, masks)):
        f = tp.locate(frame, diameter=diameter, minmass=minmass, engine="python")
        if f is not None and not f.empty:
            f["frame"] = i
            raw = frames[i]
            r = diameter // 2
            intensities = []
            for _, row in f.iterrows():
                y, x = int(row.y), int(row.x)
                x0, x1 = max(0, x - r), min(raw.shape[1], x + r + 1)
                y0, y1 = max(0, y - r), min(raw.shape[0], y + r + 1)
                roi = raw[y0:y1, x0:x1]
                intensities.append(np.mean(roi))
            f["mean_intensity"] = intensities
            features.append(f)

    features = pd.concat(features, ignore_index=True)
    print(f"  detected {features.shape[0]} feature(s)")

    tracks = tp.link_df(features, search_range=search_range, memory=memory)
    print(f"  linked {tracks['particle'].nunique()} unique particle(s)")

    particles_before = tracks["particle"].nunique()
    counts = tracks.groupby("particle").size()
    valid_particles = counts[counts >= minframenum].index
    tracks = tracks[tracks["particle"].isin(valid_particles)].copy()
    print(f"  kept {len(valid_particles)}/{particles_before} particles with >= {minframenum} frames")

    if len(tracks) == 0:
        print("  WARNING: no particles remained after filtering")
        tracks.to_csv(csv_out, index=False)
        return

    # Map frame numbers back to the original acquisition timeline for the
    # saved CSV; the in-memory tracks/features stay truncated for the mask
    # generation below.
    tracks_out = tracks.copy()
    tracks_out["frame"] = tracks_out["frame"].map(lambda i: int(valid_frame_indices[int(i)]))
    tracks_out.to_csv(csv_out, index=False)
    print(f"  saved tracks to {csv_out}")

    _write_particle_masks(
        features, tracks, tracks_out, masks, frames.shape,
        masks_dir, diameter, valid_frame_indices, total_original_frames, n_dark,
    )


def _write_particle_masks(features, tracks, tracks_out, masks, frames_shape,
                           masks_dir, diameter, valid_frame_indices,
                           total_original_frames, n_dark):
    """Per-particle mask tifs, frame by frame, written via memmap so only
    one frame is held in RAM at a time."""
    num_frames, frame_height, frame_width = frames_shape
    valid_particle_ids = set(tracks["particle"].unique())

    detection_to_particle = {}
    for _, row in tracks.iterrows():
        frame_idx = int(row["frame"])
        frame_features = features[features["frame"] == frame_idx].reset_index(drop=True)
        for det_idx, det_row in frame_features.iterrows():
            if abs(det_row["x"] - row["x"]) < 0.1 and abs(det_row["y"] - row["y"]) < 0.1:
                detection_to_particle[(frame_idx, det_idx)] = int(row["particle"])
                break

    particle_memmaps = {}
    for pid in valid_particle_ids:
        mask_filename = os.path.join(masks_dir, f"particle_{pid:04d}_mask.tif")
        mm = np.memmap(mask_filename + ".tmp", dtype=np.uint8, mode="w+",
                        shape=(num_frames, frame_height, frame_width))
        particle_memmaps[pid] = (mm, mask_filename)

    r = diameter // 2
    for frame_idx, mask_frame in enumerate(masks):
        labeled_mask, _ = label(mask_frame)
        frame_features = features[features["frame"] == frame_idx].reset_index(drop=True)

        for det_idx, row in frame_features.iterrows():
            pid_key = (frame_idx, det_idx)
            if pid_key not in detection_to_particle:
                continue
            particle_id = detection_to_particle[pid_key]
            if particle_id not in particle_memmaps:
                continue

            y, x = int(row.y), int(row.x)
            particle_label = labeled_mask[y, x] if (0 <= y < labeled_mask.shape[0]
                                                      and 0 <= x < labeled_mask.shape[1]) else 0
            if particle_label > 0:
                region = (labeled_mask == particle_label).astype(np.uint8) * 255
            else:
                y_coords, x_coords = np.ogrid[:mask_frame.shape[0], :mask_frame.shape[1]]
                distance = np.sqrt((x_coords - x) ** 2 + (y_coords - y) ** 2)
                region = (distance <= r).astype(np.uint8) * 255

            particle_memmaps[particle_id][0][frame_idx] = region

    # Reinsert blank frames at the excised dark-frame positions, so the
    # saved mask matches the original video length.
    for pid, (mm, mask_filename) in particle_memmaps.items():
        mm.flush()
        truncated_mask = np.array(mm)
        if n_dark:
            full_mask = np.zeros((total_original_frames, frame_height, frame_width), dtype=np.uint8)
            full_mask[valid_frame_indices] = truncated_mask
        else:
            full_mask = truncated_mask
        tifffile.imwrite(mask_filename, full_mask)
        del mm, truncated_mask, full_mask
        os.remove(mask_filename + ".tmp")

    del particle_memmaps
    gc.collect()
    print(f"  saved particle masks to {masks_dir}")

    mask_summary = tracks_out[["particle", "frame"]].groupby("particle").agg(
        {"frame": ["min", "max", "count"]}
    ).reset_index()
    mask_summary.columns = ["particle_id", "first_frame", "last_frame", "total_frames"]
    mask_summary.to_csv(os.path.join(masks_dir, "mask_summary.csv"), index=False)


# ---------------------------------------------------------------------------
# per-cell pipeline
# ---------------------------------------------------------------------------

def c2_process_pipeline(c2_raw_path, c2_deconv_path, celloutdir, minframenum,
                         diameter, minmass, search_range, memory,
                         valid_frame_indices, total_original_frames, pixel_size_um):
    c2_basename = os.path.splitext(os.path.basename(c2_raw_path))[0]
    c2_deconv_basename = os.path.splitext(os.path.basename(c2_deconv_path))[0]
    dog_mask_path = os.path.join(celloutdir, f"{c2_deconv_basename}_mask.tif")
    masks_dir = os.path.join(celloutdir, "individual_masks")
    os.makedirs(masks_dir, exist_ok=True)

    csv_out = os.path.join(celloutdir, f"{c2_basename}_tracks.csv")
    dff_csv = csv_out.replace(".csv", "_dff.csv")
    dff_filt_csv = dff_csv.replace(".csv", "_filt.csv")

    MaskProcessor().process_single(img_path=c2_deconv_path, out_folder=celloutdir,
                                    pixel_size_um=pixel_size_um)

    trackingpy(c2_raw_path, dog_mask_path, csv_out, masks_dir,
               minframenum=minframenum, diameter=diameter, minmass=minmass,
               search_range=search_range, memory=memory,
               valid_frame_indices=valid_frame_indices,
               total_original_frames=total_original_frames)

    dff = calculate_dff_for_tracks(csv_out, baseline_method="robust_mean")
    dff.to_csv(dff_csv, index=False)

    filtered_dff_df = filter_particles_by_length(dff_csv, minframenum, output_file=dff_filt_csv)
    valid_particles = filtered_dff_df["particle"].unique()
    clean_particle_tifs(masks_dir, valid_particles, dry_run=False)

    return dff_filt_csv


def process_cell(currcell):
    cell_name = currcell.get("cell_name", "UNKNOWN_CELL")

    if not FORCE_RERUN and is_step_complete(OUTPUTS_DIR, cell_name, STEP_NAME):
        print(f"[skip] {cell_name}: lysosome tracking already done")
        return None

    print(f"[cell] {cell_name}")

    c1_raw_path = currcell["c1_file_path"]
    c2_raw_path = currcell["c2_file_path"]
    deconvs = find_cell_data(currcell["cell_folder_path"])
    c2_deconv_path = deconvs[0]["c2_file_path"]

    total_frames = tifffile.imread(c1_raw_path).shape[0]
    valid_frame_indices = get_valid_frame_indices(cell_name, total_frames)
    n_dark = total_frames - len(valid_frame_indices)
    minframenum = int(len(valid_frame_indices) * MIN_FRAME_FRACTION)
    if n_dark:
        print(f"  excising {n_dark} dark frame(s) ({total_frames} -> {len(valid_frame_indices)}); "
              f"minframenum={minframenum}")

    celloutdir = OUTPUTS_DIR / cell_name
    celloutdir.mkdir(parents=True, exist_ok=True)

    diameter, minmass, search_range, memory = get_scaled_lyso_params(cell_name)
    pixel_size_um = get_pixel_size_um(cell_name)

    dff_filt_csv = c2_process_pipeline(
        c2_raw_path, c2_deconv_path, str(celloutdir), minframenum,
        diameter, minmass, search_range, memory,
        valid_frame_indices, total_frames, pixel_size_um,
    )

    save_particle_earliest_frame_images(
        df_out=pd.read_csv(dff_filt_csv),
        tif_path=c2_raw_path,
        output_folder=str(celloutdir / "c2_earliest_frames"),
        circle_radius=EARLIEST_FRAME_CIRCLE_RADIUS,
    )

    n_particles = pd.read_csv(dff_filt_csv)["particle"].nunique()
    return {
        "n_dark_frames_excised": int(n_dark),
        "minframenum": minframenum,
        "n_particles_kept": int(n_particles),
    }


def run(cell_names=None):
    """Runs this step for the given cell names, or every cell find_cell_data()
    returns if cell_names is None."""
    OUTPUTS_DIR.mkdir(exist_ok=True)
    logfile = OUTPUTS_DIR / "error_log_lysosome_tracking.txt"
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

    print(f"Processing {len(run_queue)} cell(s).")

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
        finally:
            gc.collect()

    print(f"\nDone. {len(run_queue) - len(error_cells)}/{len(run_queue)} cell(s) succeeded.")
    if error_cells:
        print(f"Failed: {error_cells}\nSee {logfile}")


def main():
    run(sys.argv[1:] or None)


if __name__ == "__main__":
    main()
