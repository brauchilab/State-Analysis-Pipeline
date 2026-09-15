import numpy as np

# Reuse everything else from tifffun.py unchanged -- ROI generation,
# neighbor computation, F0 calculation (calculate_f0_per_roi_optimized),
# sample_pois, etc. all work the same regardless of the transform applied
# to F0 downstream.
#
# NOT reused as-is: the three remove_positive_outliers_* functions. Despite
# their "positive side only" docstrings, all three hardcode
# np.clip(data, -1, upper_threshold) -- a lower bound that's a harmless
# no-op for dF/F0 (which has a real floor at -1) but silently truncates
# real, informative data for log2(F/F0), which has no such floor (a pixel
# 4x dimmer than baseline is a legitimate log2=-2, not an outlier). Log2-
# safe versions that clip ONLY the upper side are defined below.
from tifffun import *  # noqa: F401,F403


def remove_positive_outliers_iqr_log2(data, iqr_multiplier=1.5):
    """Same as tifffun.remove_positive_outliers_iqr, but clips ONLY the
    upper (positive) side, matching what the docstring actually promises --
    no hardcoded -1 lower bound, since log2(F/F0) has no natural floor
    there the way dF/F0 does."""
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    upper_threshold = q3 + (iqr_multiplier * iqr)
    return np.minimum(data, upper_threshold)


def remove_positive_outliers_percentile_log2(data, upper_percentile=99):
    """Same as tifffun.remove_positive_outliers_percentile, but clips ONLY
    the upper (positive) side -- see remove_positive_outliers_iqr_log2."""
    upper_threshold = np.percentile(data, upper_percentile)
    return np.minimum(data, upper_threshold)


def remove_positive_outliers_zscore_log2(data, z_threshold=3):
    """Same as tifffun.remove_positive_outliers_zscore, but clips ONLY the
    upper (positive) side -- see remove_positive_outliers_iqr_log2."""
    mean = np.mean(data)
    std = np.std(data)
    upper_threshold = mean + (z_threshold * std)
    return np.minimum(data, upper_threshold)


def dff_pixel_optimized_log2(frame, all_masks, f0_per_roi):
    """Same as tifffun.dff_pixel_optimized, but returns log2(F/F0) instead
    of (F-F0)/F0.

    Why: dF/F0 = (F-F0)/F0 has a hard floor at -1 (when F=0) and is
    asymmetric -- a 2x brighter pixel gives dF/F0=+1, but a 2x dimmer pixel
    only gives dF/F0=-0.5, not -1. log2(F/F0) is symmetric in log-space:
    2x brighter -> +1, 2x dimmer -> -1, unchanged -> 0. That symmetry may
    suit HMM emission modeling (e.g. Gaussian assumptions) better than the
    skewed, floor-clipped dF/F0 range.

    F0 itself is computed identically either way -- this function only
    changes what's done with F0 once it's already known; F0's own
    dark-frame safety (calculate_f0_per_roi_optimized's nanmean handling,
    guarded against baseline-window overlap in calcDiffCube) is unaffected.

    Pixels with roi_pixels <= 0 (masked-out / no-signal, including fully
    dark acquisition frames where the whole frame is 0) are left at 0 in
    the output, same "0 = no data" sentinel convention as the original --
    downstream code (predict_states_probabilities_vectorized's
    `frame != 0` check) already relies on this and needs no changes.
    """
    result_frame = np.zeros_like(frame, dtype=np.float32)

    for roi_index, (y_slice, x_slice) in enumerate(all_masks):
        f0 = f0_per_roi[roi_index]
        if f0 <= 0:
            continue

        roi_pixels = frame[y_slice, x_slice]
        non_zero_mask = roi_pixels > 0

        if np.any(non_zero_mask):
            temp_result = np.zeros_like(roi_pixels, dtype=np.float32)
            temp_result[non_zero_mask] = np.log2(roi_pixels[non_zero_mask] / f0)
            result_frame[y_slice, x_slice] = temp_result

    return result_frame


def process_frame_frame_optimized_log2(image_array, roi_size, frame_range, use_memmap=False):
    """Same as tifffun.process_frame_frame_optimized, but calls
    dff_pixel_optimized_log2 for the final per-frame transform instead of
    dff_pixel_optimized. Everything before that (ROI generation, neighbor
    computation, F0 calculation) is identical -- copied here rather than
    partially reusing the original, since that function inlines the
    dff_pixel_optimized call rather than taking it as a parameter."""
    frames = image_array.shape[0]
    height, width = image_array.shape[1:]

    print("Starting optimized processing (log2 variant)")

    print("Generating ROIs...")
    rois, cols, rows, rois_shape = generate_rois_vectorized((height, width), roi_size)

    print("Computing neighbors...")
    neighbors_list = define_neighbors_for_all_rois(cols, rows)

    print("Creating ROI masks...")
    all_masks = create_all_roi_masks(rois, (height, width))

    all_roi_intensities = np.zeros((frames, len(rois)), dtype=np.float32)

    print("Processing frame intensities...")
    for frame_idx in range(frames):
        frame = image_array[frame_idx]
        all_roi_intensities[frame_idx] = process_frame_range_optimized(frame, rois)

        if frame_idx % 10 == 0 or frame_idx == frames - 1:
            print(f"Processed roi_intensities of frame {frame_idx+1}/{frames}")
    print(all_roi_intensities.shape)

    print("Calculating F0 values...")
    f0_per_roi = calculate_f0_per_roi_optimized(all_roi_intensities, frame_range, neighbors_list)

    if use_memmap:
        dff_result, memmap_filename = create_memmap_array((frames, height, width))
    else:
        dff_result = np.zeros((frames, height, width), dtype=np.float32)

    print("Calculating log2(F/F0) for all frames...")
    for frame_idx in range(frames):
        dff_result[frame_idx] = dff_pixel_optimized_log2(image_array[frame_idx], all_masks, f0_per_roi)

        if frame_idx % 10 == 0 or frame_idx == frames - 1:
            print(f"Processed pixels of frame {frame_idx+1}/{frames}")

    return dff_result
