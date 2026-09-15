#!/usr/bin/env python
# get_data_and_artifacts.py
"""
Step 0: per-cell setup before deconvolution.

For each cell folder in inputs/:
  - reads acquisition metadata (pixel size, frame interval, etc.) from
    RAW_<cell>.tif
  - flags dark frame runs in C2_RAW_<cell>.tif (camera glitch / treatment
    artifact showing up as a blackout)
  - flags bright frame runs in C2_RAW_<cell>.tif (optical flash / treatment
    artifact showing up as a spike)

If a cell's C1_RAW/C2_RAW split files don't exist yet, they're created
first, since both the artifact check here and deconvolution downstream
need them on disk.

Saves, per cell, under outputs/<cell_name>/:
    cell_metadata.json
    dark_frame_check.png,   dark_frames.json
    bright_frame_check.png, bright_frames.json
and outputs/metadata_summary.csv across all cells.

Run from the folder containing inputs/:
    python get_data_and_artifacts.py                 # all cells
    python get_data_and_artifacts.py 010519-Cell2     # one cell
"""

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from frame_intensity_utils import frame_intensity_profile, detect_intensity_runs
from progress_tracker import update_step, is_step_complete

INPUTS_DIR = Path("./inputs")
OUTPUTS_DIR = Path("./outputs")

FORCE_RERUN = False

# metadata defaults, used when pixel size or frame interval can't be read
DEFAULT_PIXEL_SIZE = 0.1
DEFAULT_FRAME_INTERVAL = 1.0
ERROR_ON_MISSING_METADATA = False  # True: raise instead of falling back to defaults

# dark/bright frame detection
DARK_FRACTION = 0.5      # frame flagged if mean intensity < this * stack median
DARK_MIN_CONSECUTIVE = 2
BRIGHT_FACTOR = 2.0       # frame flagged if mean intensity > this * stack median
BRIGHT_MIN_CONSECUTIVE = 1  # flashes are usually shorter than dark periods


# ---------------------------------------------------------------------------
# channel split (only runs if C1_RAW/C2_RAW aren't already there)
# ---------------------------------------------------------------------------

def split_channels(stack):
    if stack.ndim != 4:
        raise ValueError(f"Expected 4D stack, got shape {stack.shape}")
    if stack.shape[1] == 2:
        return stack[:, 0], stack[:, 1]
    elif stack.shape[0] == 2:
        return stack[0], stack[1]
    else:
        raise ValueError(f"Could not identify channel axis in shape {stack.shape}")


def ensure_channels_split(cell_dir, cell_name):
    """Write C1_RAW/C2_RAW next to RAW if they don't already exist."""
    c1_path = cell_dir / f"C1_RAW_{cell_name}.tif"
    c2_path = cell_dir / f"C2_RAW_{cell_name}.tif"
    if c1_path.exists() and c2_path.exists():
        return

    raw_path = cell_dir / f"RAW_{cell_name}.tif"
    if not raw_path.exists():
        return  # metadata step will report the missing RAW file

    print(f"  [split] {cell_name}: no C1/C2_RAW yet, splitting from RAW")
    stack = tifffile.imread(raw_path)
    try:
        c1, c2 = split_channels(stack)
    except ValueError as e:
        print(f"    [error] {cell_name}: {e}")
        update_step(OUTPUTS_DIR, cell_name, "channel_split", "error", {"message": str(e)})
        return

    tifffile.imwrite(c1_path, c1, imagej=True)
    tifffile.imwrite(c2_path, c2, imagej=True)
    print(f"    saved {c1_path.name}, {c2_path.name}")


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------

def _axis_size(shape, axes, axis_char):
    if axes and axis_char in axes:
        return shape[axes.index(axis_char)]
    return None


def _tag_value(page, tag_name):
    tag = page.tags.get(tag_name)
    if tag is None:
        return None
    val = tag.value
    # tifffile returns resolution tags as (numerator, denominator) rationals
    if isinstance(val, tuple) and len(val) == 2:
        num, den = val
        return (num / den) if den else None
    return val


def extract_metadata(raw_path):
    """Pull the Fiji 'Image > Properties'-style fields out of an ImageJ tif."""
    with tifffile.TiffFile(raw_path) as tf:
        page = tf.pages[0]
        ij_meta = tf.imagej_metadata or {}
        shape = tf.series[0].shape
        axes = tf.series[0].axes

        meta = {
            "source_file": os.path.basename(str(raw_path)),
            "shape": shape,
            "axes": axes,
        }

        meta["channels"] = ij_meta.get("channels") or _axis_size(shape, axes, "C") or 1
        meta["slices"] = ij_meta.get("slices") or _axis_size(shape, axes, "Z") or 1
        meta["frames"] = ij_meta.get("frames") or _axis_size(shape, axes, "T") or 1

        # XResolution/YResolution are pixels-per-unit, invert for unit-per-pixel
        x_res = _tag_value(page, "XResolution")
        y_res = _tag_value(page, "YResolution")
        meta["pixel_width"] = (1 / x_res) if x_res else None
        meta["pixel_height"] = (1 / y_res) if y_res else None
        meta["unit"] = ij_meta.get("unit")

        meta["voxel_depth"] = ij_meta.get("spacing")
        meta["frame_interval_sec"] = ij_meta.get("finterval")

        meta["origin_x"] = ij_meta.get("xorigin", 0)
        meta["origin_y"] = ij_meta.get("yorigin", 0)

    return meta


def apply_defaults(meta, cell_name):
    """Fill in pixel size / frame interval if missing, and report what happened."""
    fallback_notes = {}

    if not meta.get("pixel_width") or not meta.get("pixel_height"):
        if ERROR_ON_MISSING_METADATA:
            raise ValueError(f"{cell_name}: pixel size not found in tif metadata")
        meta["pixel_width"] = meta["pixel_width"] or DEFAULT_PIXEL_SIZE
        meta["pixel_height"] = meta["pixel_height"] or DEFAULT_PIXEL_SIZE
        fallback_notes["pixel_size_defaulted"] = DEFAULT_PIXEL_SIZE

    if not meta.get("frame_interval_sec"):
        if ERROR_ON_MISSING_METADATA:
            raise ValueError(f"{cell_name}: frame interval not found in tif metadata")
        meta["frame_interval_sec"] = DEFAULT_FRAME_INTERVAL
        fallback_notes["frame_interval_defaulted"] = DEFAULT_FRAME_INTERVAL

    return fallback_notes


def run_metadata_step(cell_dir, cell_out_dir):
    cell_name = cell_dir.name
    step = "metadata"

    if not FORCE_RERUN and is_step_complete(OUTPUTS_DIR, cell_name, step):
        print(f"  [skip] {cell_name}: metadata already extracted")
        return None

    raw_path = cell_dir / f"RAW_{cell_name}.tif"
    if not raw_path.exists():
        print(f"  [skip] {cell_name}: no RAW_{cell_name}.tif found")
        return None

    try:
        meta = extract_metadata(raw_path)
        fallback_notes = apply_defaults(meta, cell_name)
    except Exception as e:
        print(f"  [error] {cell_name} metadata: {e}")
        update_step(OUTPUTS_DIR, cell_name, step, "error", {"message": str(e)})
        return None

    json_path = cell_out_dir / "cell_metadata.json"
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    update_step(OUTPUTS_DIR, cell_name, step, "complete", fallback_notes)
    print(f"  [saved] {json_path.name}" + (f"  ({fallback_notes})" if fallback_notes else ""))

    row = {"cell_name": cell_name, **meta}
    row.pop("shape", None)
    row.pop("axes", None)
    return row


# ---------------------------------------------------------------------------
# dark / bright frame detection (shared logic, opposite direction)
# ---------------------------------------------------------------------------

def run_artifact_frame_step(cell_dir, cell_out_dir, mode):
    """mode: 'dark' or 'bright'."""
    cell_name = cell_dir.name
    step = f"{mode}_frames"

    if not FORCE_RERUN and is_step_complete(OUTPUTS_DIR, cell_name, step):
        print(f"  [skip] {cell_name}: {mode} frame check already done")
        return

    c2_path = cell_dir / f"C2_RAW_{cell_name}.tif"
    if not c2_path.exists():
        print(f"  [skip] {cell_name}: no C2_RAW_{cell_name}.tif found")
        return

    if mode == "dark":
        factor, min_consecutive, direction = DARK_FRACTION, DARK_MIN_CONSECUTIVE, "below"
        span_color = "red"
    else:
        factor, min_consecutive, direction = BRIGHT_FACTOR, BRIGHT_MIN_CONSECUTIVE, "above"
        span_color = "gold"

    profile = frame_intensity_profile(c2_path)
    confirmed, threshold, ranges = detect_intensity_runs(profile, factor, min_consecutive, direction)

    if ranges:
        print(f"  [{mode.upper()}] {cell_name}: {len(ranges)} run(s) -> "
              f"{ranges} (of {len(profile)} frames)")
    else:
        print(f"  [ok] {cell_name}: no {mode} runs detected ({len(profile)} frames)")

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(profile, color="steelblue", linewidth=0.8, label="mean frame intensity")
    ax.axhline(threshold, color="gray", linestyle="--", linewidth=0.8,
               label=f"threshold ({factor}x median)")
    for s, e in ranges:
        ax.axvspan(s, e, color=span_color, alpha=0.35)
    ax.set_title(f"{cell_name} -- frame intensity ({span_color} = flagged {mode} run)")
    ax.set_xlabel("frame")
    ax.set_ylabel("mean intensity")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    plot_path = cell_out_dir / f"{mode}_frame_check.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)

    json_path = cell_out_dir / f"{mode}_frames.json"
    with open(json_path, "w") as f:
        json.dump({
            "cell_name": cell_name,
            "n_frames": int(len(profile)),
            f"{mode}_threshold_factor": float(factor),
            "min_consecutive": min_consecutive,
            "intensity_threshold": float(threshold),
            f"{mode}_frame_ranges": [[int(s), int(e)] for s, e in ranges],
            f"{mode}_frame_indices": [int(i) for i in np.where(confirmed)[0]],
        }, f, indent=2)

    update_step(OUTPUTS_DIR, cell_name, step, "complete", {
        f"n_{mode}_runs": len(ranges),
        f"{mode}_frame_ranges": ranges,
    })
    print(f"         saved {plot_path.name} and {json_path.name}")


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def process_cell(cell_dir, summary_rows):
    cell_name = cell_dir.name
    print(f"[cell] {cell_name}")

    cell_out_dir = OUTPUTS_DIR / cell_name
    cell_out_dir.mkdir(parents=True, exist_ok=True)

    ensure_channels_split(cell_dir, cell_name)

    row = run_metadata_step(cell_dir, cell_out_dir)
    if row is not None:
        summary_rows.append(row)

    run_artifact_frame_step(cell_dir, cell_out_dir, mode="dark")
    run_artifact_frame_step(cell_dir, cell_out_dir, mode="bright")


def run(cell_names=None):
    """Runs this step for the given cell names, or every cell in inputs/
    if cell_names is None. Callable directly (e.g. from the main pipeline
    script) or via main() below."""
    if not INPUTS_DIR.is_dir():
        raise SystemExit(f"inputs dir not found: {INPUTS_DIR.resolve()}")
    OUTPUTS_DIR.mkdir(exist_ok=True)

    if cell_names:
        cell_dirs = [INPUTS_DIR / name for name in cell_names]
        missing = [d for d in cell_dirs if not d.is_dir()]
        if missing:
            raise SystemExit(f"Cell folder(s) not found: {[str(d) for d in missing]}")
    else:
        cell_dirs = sorted(d for d in INPUTS_DIR.iterdir() if d.is_dir())

    print(f"Processing {len(cell_dirs)} cell(s)...")
    summary_rows = []
    for cell_dir in cell_dirs:
        process_cell(cell_dir, summary_rows)

    if summary_rows:
        summary_path = OUTPUTS_DIR / "metadata_summary.csv"
        fieldnames = ["cell_name"] + [k for k in summary_rows[0].keys() if k != "cell_name"]
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n[saved] combined summary: {summary_path}")

    print("\nDone. Check the dark/bright_frame_check.png plots before trusting the json flags.")


def main():
    run(sys.argv[1:] or None)


if __name__ == "__main__":
    main()
