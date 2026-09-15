#!/usr/bin/env python
# deconvolution.py
"""
Step 1: deconvolution, optional per RUN_DECONVOLUTION below.

Expects C1_RAW_<cell>.tif and C2_RAW_<cell>.tif to already exist in each
cell folder.

If RUN_DECONVOLUTION is True:
    C1: despeckle, rolling-ball background subtraction, Richardson-Lucy
        deconvolution.
    C2: Richardson-Lucy deconvolution, then total-variation denoising.
    Saves deconv/C1_Deconv_<cell>.tif and deconv/C2_Deconv_<cell>.tif.

If RUN_DECONVOLUTION is False:
    Creates deconv/C1_Deconv_<cell>.tif and deconv/C2_Deconv_<cell>.tif
    as symlinks back to the raw split channels, so downstream steps that
    expect files there still find something.

Run from the folder containing inputs/:
    python deconvolution.py                 # all cells
    python deconvolution.py 010519-Cell2     # one cell
"""

import os
import sys
from pathlib import Path

import numpy as np
import tifffile
import skimage as sk
from tv_denoise import denoise_stack

from progress_tracker import update_step, is_step_complete

INPUTS_DIR = Path("./inputs")
OUTPUTS_DIR = Path("./outputs")

STEP_NAME = "deconvolution"
FORCE_RERUN = False

RUN_DECONVOLUTION = True
PSF_PATH = Path("./PSF.tif")
C1_ITERATIONS = 30
C2_ITERATIONS = 20
C2_DENOISE_WEIGHT = 0.01
ROLLING_BALL_RADIUS = 50


# ---------------------------------------------------------------------------
# C1 preprocessing
# ---------------------------------------------------------------------------

def despeckle_frame(frame):
    from skimage.filters import median
    from skimage.morphology import disk
    return median(frame, disk(1))


def subtract_background_rolling_ball(frame, radius=ROLLING_BALL_RADIUS):
    from skimage.restoration import rolling_ball
    background = rolling_ball(frame, radius=radius)
    result = frame.astype(np.float32) - background.astype(np.float32)
    result = np.clip(result, 0, None)
    return result.astype(frame.dtype)


def preprocess_c1_stack(stack, radius=ROLLING_BALL_RADIUS):
    n_frames = stack.shape[0]
    out = np.zeros_like(stack)
    for f in range(n_frames):
        frame = despeckle_frame(stack[f])
        frame = subtract_background_rolling_ball(frame, radius=radius)
        out[f] = frame
        print(f"      preprocessing frame {f + 1}/{n_frames}", end="\r")
    print()
    return out


# ---------------------------------------------------------------------------
# deconvolution
# ---------------------------------------------------------------------------

def deconvolve_frame(frame, psf, iterations):
    frame = frame.astype(np.float32, copy=False)
    psf = psf.astype(np.float32, copy=False)
    psf = psf / (psf.sum() + 1e-10)
    frame = frame / (np.max(frame) + 1e-10)
    return sk.restoration.richardson_lucy(frame, psf, num_iter=iterations)


def deconvolve_stack(stack, psf, iterations):
    n_frames = stack.shape[0]
    out = np.zeros_like(stack, dtype=np.float32)
    for f in range(n_frames):
        out[f] = deconvolve_frame(stack[f], psf, iterations)
        print(f"    deconvolving frame {f + 1}/{n_frames}", end="\r")
    print()
    return out


# ---------------------------------------------------------------------------
# per-cell processing
# ---------------------------------------------------------------------------

def run_deconvolution(cell_dir, cell_name, deconv_dir, psf):
    c1_raw = tifffile.imread(cell_dir / f"C1_RAW_{cell_name}.tif")
    c2_raw = tifffile.imread(cell_dir / f"C2_RAW_{cell_name}.tif")

    print("  preprocessing C1...")
    c1_preprocessed = preprocess_c1_stack(c1_raw)

    print("  deconvolving C1...")
    c1_deconv = deconvolve_stack(c1_preprocessed, psf, C1_ITERATIONS)

    print("  deconvolving C2...")
    c2_deconv = deconvolve_stack(c2_raw, psf, C2_ITERATIONS)

    print("  denoising C2...")
    c2_denoised = denoise_stack(c2_deconv, d_weight=C2_DENOISE_WEIGHT)

    c1_out = deconv_dir / f"C1_Deconv_{cell_name}.tif"
    c2_out = deconv_dir / f"C2_Deconv_{cell_name}.tif"
    tifffile.imwrite(c1_out, c1_deconv, imagej=True)
    tifffile.imwrite(c2_out, c2_denoised, imagej=True)

    return {
        "deconvolution_run": True,
        "c1_iterations": C1_ITERATIONS,
        "c2_iterations": C2_ITERATIONS,
        "c2_denoise_weight": C2_DENOISE_WEIGHT,
    }


def make_passthrough_symlinks(cell_dir, cell_name, deconv_dir):
    """No deconvolution -- link deconv/ names back to the raw split channels."""
    pairs = [
        (f"C1_RAW_{cell_name}.tif", f"C1_Deconv_{cell_name}.tif"),
        (f"C2_RAW_{cell_name}.tif", f"C2_Deconv_{cell_name}.tif"),
    ]
    for src_name, link_name in pairs:
        src = cell_dir / src_name
        link = deconv_dir / link_name

        if link.is_symlink() and link.exists():
            continue
        if link.is_symlink() and not link.exists():
            link.unlink()  # broken symlink, recreate below
        elif link.exists():
            print(f"    [warn] {link_name} exists as a real file, leaving it alone")
            continue

        rel_src = os.path.relpath(src, deconv_dir)
        link.symlink_to(rel_src)
        print(f"  [link] {link_name} -> {rel_src}")

    return {"deconvolution_run": False}


def process_cell(cell_dir, psf):
    cell_name = cell_dir.name

    if not FORCE_RERUN and is_step_complete(OUTPUTS_DIR, cell_name, STEP_NAME):
        print(f"[skip] {cell_name}: deconvolution step already done")
        return

    c1_raw_path = cell_dir / f"C1_RAW_{cell_name}.tif"
    c2_raw_path = cell_dir / f"C2_RAW_{cell_name}.tif"
    if not c1_raw_path.exists() or not c2_raw_path.exists():
        print(f"[skip] {cell_name}: missing C1_RAW/C2_RAW, split channels first")
        return

    print(f"[cell] {cell_name}")
    deconv_dir = cell_dir / "deconv"
    deconv_dir.mkdir(exist_ok=True)

    try:
        if RUN_DECONVOLUTION:
            notes = run_deconvolution(cell_dir, cell_name, deconv_dir, psf)
        else:
            notes = make_passthrough_symlinks(cell_dir, cell_name, deconv_dir)
    except Exception as e:
        print(f"  [error] {cell_name}: {e}")
        update_step(OUTPUTS_DIR, cell_name, STEP_NAME, "error", {"message": str(e)})
        return

    update_step(OUTPUTS_DIR, cell_name, STEP_NAME, "complete", notes)
    print(f"  [done] {cell_name}")


def run(cell_names=None):
    """Runs this step for the given cell names, or every cell in inputs/
    if cell_names is None."""
    if not INPUTS_DIR.is_dir():
        raise SystemExit(f"inputs dir not found: {INPUTS_DIR.resolve()}")
    OUTPUTS_DIR.mkdir(exist_ok=True)

    psf = None
    if RUN_DECONVOLUTION:
        if not PSF_PATH.exists():
            raise SystemExit(f"PSF file not found: {PSF_PATH.resolve()}")
        psf = tifffile.imread(PSF_PATH)

    if cell_names:
        cell_dirs = [INPUTS_DIR / name for name in cell_names]
        missing = [d for d in cell_dirs if not d.is_dir()]
        if missing:
            raise SystemExit(f"Cell folder(s) not found: {[str(d) for d in missing]}")
    else:
        cell_dirs = sorted(d for d in INPUTS_DIR.iterdir() if d.is_dir())

    mode = "deconvolving" if RUN_DECONVOLUTION else "symlinking (deconvolution disabled)"
    print(f"Processing {len(cell_dirs)} cell(s), {mode}...")
    for cell_dir in cell_dirs:
        process_cell(cell_dir, psf)

    print("\nDone.")


def main():
    run(sys.argv[1:] or None)


if __name__ == "__main__":
    main()
