"""
Shared utilities for the 4 Fiji export scripts.
"""

import os
import json
import glob
import logging

import numpy as np
import pandas as pd
import tifffile


# ---------------- Generic ----------------

def get_dark_frame_indices(dff0_outputs_dir, cell_name):
    dark_path = os.path.join(dff0_outputs_dir, cell_name, "dark_frames.json")
    if not os.path.exists(dark_path):
        return None
    with open(dark_path) as f:
        data = json.load(f)
    indices = data.get("dark_frame_indices", [])
    return indices if indices else None


def save_tif(array, out_path, imagej=True, compression="zlib"):
    """Saves a single-channel array (not RGB) -- for real numeric data
    (16/32 bit), not colored images. Embeds a fixed min/max visualization
    range in the ImageJ metadata: without this, Fiji recalculates
    contrast per frame, making the same background value (0.0 in signed
    data like dF/F0 or log2) flicker between a different gray each frame.
    A fixed range keeps that gray consistent across the stack -- the
    background still won't be pure black in signed data, which is
    expected, not a bug."""
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    finite = array[np.isfinite(array)] if np.issubdtype(array.dtype, np.floating) else array
    if finite.size > 0:
        vmin, vmax = float(finite.min()), float(finite.max())
    else:
        vmin, vmax = 0.0, 1.0
    tifffile.imwrite(out_path, array, imagej=imagej, compression=compression,
                      metadata={"min": vmin, "max": vmax})
    print(f"    saved: {out_path}  (shape={array.shape}, dtype={array.dtype}, "
          f"fixed display range=[{vmin:.4g}, {vmax:.4g}])")


def save_rgb_tiff(rgb_array, out_path, compression="zlib"):
    """Saves an RGB array (uint8 x3), for colored state exports."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tifffile.imwrite(out_path, rgb_array, imagej=True, metadata={"axes": "TYXS"},
                      photometric="rgb", compression=compression)
    print(f"    saved: {out_path}  (shape={rgb_array.shape})")


# ---------------- Particle masks / tracks ----------------

def get_mask_files(masks_dir):
    return sorted(glob.glob(os.path.join(masks_dir, "particle_*_mask.tif")))


def particle_id_from_mask_path(mask_path):
    return int(os.path.basename(mask_path).split("_")[1])


def find_tracks_csv(celloutdir, cell_name):
    candidate = os.path.join(celloutdir, f"C2_RAW_{cell_name}_tracks_dff_filt.csv")
    if os.path.exists(candidate):
        return candidate
    matches = glob.glob(os.path.join(celloutdir, "*_dff_filt.csv"))
    return matches[0] if matches else None


def to_log2(df_f0_values):
    """log2(F/F0) = log2(1 + df_f0). F/F0 <= 0 (df_f0 <= -1) -> NaN."""
    ratio = 1 + df_f0_values
    out = np.full_like(ratio, np.nan, dtype=np.float64)
    valid = ratio > 0
    out[valid] = np.log2(ratio[valid])
    return out


class _TiffErrorCapture(logging.Handler):
    """Captures ERROR-level log records tifffile emits internally during
    a read (not exceptions). See safe_read_mask_stack for why."""
    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _read_with_error_capture(read_fn, mask_path):
    """Runs read_fn(mask_path) (tifffile.memmap or tifffile.imread) while
    capturing any ERROR tifffile logs during the read. Returns (array,
    records); a non-empty records list means something went wrong
    internally even without a raised exception."""
    tiff_logger = logging.getLogger("tifffile")
    handler = _TiffErrorCapture()
    tiff_logger.addHandler(handler)
    try:
        arr = read_fn(mask_path)
    finally:
        tiff_logger.removeHandler(handler)
    return arr, handler.records


def safe_read_mask_stack(mask_path, expected_shape=None):
    """Reads a particle mask via memmap first (fast, low memory); falls
    back to a full read if the tif is malformed.

    tifffile can silently recover from a truncated/corrupt file without
    raising an exception and without the resulting shape looking wrong --
    missing frames just come back as silent zeros, indistinguishable from
    "no particle here". Detection relies on capturing tifffile's internal
    error log messages directly, confirmed necessary against a real
    truncated file (shape and try/except alone both missed it).

    Returns the array, or None if nothing could be read at all."""
    try:
        arr, records = _read_with_error_capture(tifffile.memmap, mask_path)
    except Exception as e1:
        print(f"    [WARN] memmap failed for {mask_path} ({e1}); trying a full read...")
        try:
            arr, records = _read_with_error_capture(tifffile.imread, mask_path)
        except Exception as e2:
            print(f"    [WARN] {mask_path} looks corrupt, skipping this particle. ({e2})")
            return None

    if records:
        print(f"    [WARN] {mask_path}: tifffile logged {len(records)} internal error(s) "
              f"during the read -- the file is probably CORRUPT/TRUNCATED. The resulting "
              f"shape may still look normal ({arr.shape}), with missing frames read back "
              f"silently as zero instead of a visible error. Review/regenerate this file -- "
              f"this particle's data may be incomplete.")
    elif expected_shape is not None and arr.shape != expected_shape:
        print(f"    [WARN] {mask_path}: read shape {arr.shape} != expected shape "
              f"{expected_shape} -- possible corruption. Review this file.")

    return arr


def get_mask_shape(mask_files):
    """Tries each mask file until one reads successfully -- does NOT
    assume mask_files[0] is valid, since that could be the corrupt one."""
    for path in mask_files:
        arr = safe_read_mask_stack(path)
        if arr is not None:
            return arr.shape, path
    return None, None


def build_combined_value_canvas(F, H, W, masks_dir, particle_value_lookup, dtype=np.float32):
    """particle_value_lookup: {particle_id: {frame: value}}. Paints each
    particle's value (one flat number per particle-frame, not pixel-
    interpolated) onto its own mask, with all particles combined into one
    (F,H,W) single-channel canvas rather than one file per particle.
    Background (no mask present) = 0. A particle whose mask can't be read
    at all is skipped with a warning; a truncated mask is used partially,
    with an explicit warning -- see safe_read_mask_stack."""
    canvas = np.zeros((F, H, W), dtype=dtype)
    mask_files = get_mask_files(masks_dir)
    if not mask_files:
        return canvas, 0

    n_ok = 0
    for mask_path in mask_files:
        particle_id = particle_id_from_mask_path(mask_path)
        frame_values = particle_value_lookup.get(particle_id)
        if frame_values is None:
            continue

        mm = safe_read_mask_stack(mask_path, expected_shape=(F, H, W))
        if mm is None:
            continue

        n_frames_avail = min(F, mm.shape[0])
        for f in range(n_frames_avail):
            if f not in frame_values:
                continue
            val = frame_values[f]
            if isinstance(val, float) and np.isnan(val):
                continue
            frame_mask = mm[f] > 0
            if not frame_mask.any():
                continue
            canvas[f][frame_mask] = val
        n_ok += 1

    return canvas, n_ok


# ---------------- State (ER and lysosome) ----------------

def create_simple_state_matrix(state_probs_float):
    """(F,H,W,3) float32 -> (F,H,W) int8. -1 for invalid pixels, 0/1/2
    for the dominant state (argmax)."""
    F, H, W, _ = state_probs_float.shape
    simple = np.full((F, H, W), -1, dtype=np.int8)
    valid = ~np.isnan(state_probs_float).any(axis=-1)
    simple[valid] = state_probs_float[valid].argmax(axis=-1).astype(np.int8)
    return simple


def find_states_csv(celloutdir, pattern="*C2*tracks_dff_filt_with_states_ordered.csv"):
    matches = sorted(glob.glob(os.path.join(celloutdir, pattern)))
    return matches[0] if matches else None


def load_particle_states(celloutdir, pattern="*C2*tracks_dff_filt_with_states_ordered.csv"):
    """{particle_id: {frame: predicted_state}}, or None if the CSV is
    missing or missing expected columns."""
    states_csv = find_states_csv(celloutdir, pattern)
    if states_csv is None:
        return None
    df = pd.read_csv(states_csv)
    required = {"predicted_state", "particle", "frame"}
    if not required.issubset(df.columns):
        print(f"    [WARN] states CSV found but missing expected columns: {states_csv}")
        return None
    lookup = {}
    for pid, group in df.groupby("particle"):
        lookup[int(pid)] = dict(zip(group["frame"].astype(int), group["predicted_state"]))
    return lookup
