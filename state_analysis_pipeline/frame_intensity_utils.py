# frame_intensity_utils.py
"""
Shared logic for the dark/bright frame detection scripts. Both flag
frames whose mean intensity strays too far from the stack median -- they
only differ in direction (below for dark, above for bright) and defaults.
"""

import numpy as np
import tifffile


def frame_intensity_profile(tif_path):
    """Mean intensity per frame for a (t, y, x) raw stack."""
    stack = tifffile.imread(tif_path)
    if stack.ndim != 3:
        raise ValueError(f"Expected a (t, y, x) stack, got shape {stack.shape}")
    return stack.mean(axis=(1, 2))


def detect_intensity_runs(profile, factor, min_consecutive, mode):
    """
    mode: 'below' for dark frames, 'above' for bright frames.
    Returns (confirmed bool array, threshold, list of (start, end) inclusive ranges).
    Runs shorter than min_consecutive are ignored.
    """
    threshold = factor * np.median(profile)
    flagged = profile < threshold if mode == "below" else profile > threshold

    ranges = []
    start = None
    for i, flag in enumerate(flagged):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= min_consecutive:
                ranges.append((start, i - 1))
            start = None
    if start is not None and len(flagged) - start >= min_consecutive:
        ranges.append((start, len(flagged) - 1))

    confirmed = np.zeros_like(flagged)
    for s, e in ranges:
        confirmed[s:e + 1] = True

    return confirmed, threshold, ranges
