# calc_diff_cube.py
"""
calcDiffCube in both its dF/F0 and log2(F/F0) forms, extracted from
pipeline_funcs_v2.py and pipeline_funcs_log2.py respectively, since
importing either pulled in a large unrelated import chain (dogmasks,
masktracking, dist_matrix, dist_an, video_funcs) needed only by other,
unused functions in those files -- older orchestration code
(c2_process_pipeline, c1_c2_overlay_pipeline, c1_state_probs,
c1_c2_overlay) superseded by this pipeline's own scripts.

Both variants share the same baseline-window dark-frame guard (the exact
same check used to be duplicated word-for-word in both original files):
raises before computing anything if frame_range overlaps a known dark
period, since an all-dark baseline window would otherwise silently
produce NaN F0 for the affected ROIs, which then propagates NaN into the
entire output for those ROIs, not just the dark frames -- see
tifffun.dff_pixel_optimized / tifffun_log2.dff_pixel_optimized_log2's
"if f0 <= 0: continue" check, which does not catch NaN.
"""

import time

import tifffile

from tifffun import process_frame_frame_optimized
from tifffun_log2 import process_frame_frame_optimized_log2


def _check_baseline_not_dark(frame_range, dark_frame_indices, variant_label):
    if not dark_frame_indices:
        return
    start_frame, end_frame = frame_range
    baseline_range = set(range(start_frame, end_frame + 1))
    overlap = baseline_range & set(dark_frame_indices)
    if overlap:
        raise ValueError(
            f"calcDiffCube ({variant_label}): baseline window frame_range={frame_range} "
            f"overlaps {len(overlap)} dark (blacked-out) frame(s) {sorted(overlap)}. "
            f"F0 for affected ROIs would be computed from all-dark data, which "
            f"propagates NaN into the entire output for those ROIs (not just the dark "
            f"frames). Pick a frame_range that avoids the dark period for this cell."
        )


def calcDiffCube(img_path, roi_size=(10, 10), frame_range=(1, 4), use_memmap=False,
                  dark_frame_indices=None):
    """dF/F0 = (F-F0)/F0."""
    time_s = time.time()
    _check_baseline_not_dark(frame_range, dark_frame_indices, "dF/F0")

    img = tifffile.imread(img_path)
    print(f"Loaded image shape: {img.shape}, dtype: {img.dtype}")

    dff_result = process_frame_frame_optimized(img, roi_size, frame_range, use_memmap)
    print(f"Finished in {time.time() - time_s:.2f} seconds")
    return dff_result


def calcDiffCube_log2(img_path, roi_size=(10, 10), frame_range=(1, 4), use_memmap=False,
                       dark_frame_indices=None):
    """log2(F/F0). Same baseline-window guard as calcDiffCube; only what's
    done with F0 once known differs."""
    time_s = time.time()
    _check_baseline_not_dark(frame_range, dark_frame_indices, "log2")

    img = tifffile.imread(img_path)
    print(f"Loaded image shape: {img.shape}, dtype: {img.dtype}")

    dff_result = process_frame_frame_optimized_log2(img, roi_size, frame_range, use_memmap)
    print(f"Finished in {time.time() - time_s:.2f} seconds")
    return dff_result
