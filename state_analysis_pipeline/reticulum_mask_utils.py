# reticulum_mask_utils.py
"""
c1_dec_mask_raw and _greedy_maximal_spacing, split out of
old1_pipe_funcs_anex.py so importing them doesn't pull in that file's
other top-level imports (pipeline_funcs.py, with its own fragile import
chain -- dogmasks, masktracking, dist_matrix, dist_an, video_funcs --
none of which either function actually uses). The rest of that file is
ring/lysosome correlation analysis and an older, superseded POI sampling
function, not used by this pipeline.

The DoG/mask/contour helpers below are inlined from utils.py, trimmed to
the single-channel case c1_dec_mask_raw actually uses (utils.py's
versions support multi-channel 3D/4D stacks and loading from ND2 files,
none of which apply here). This also drops utils.py's module-level
joblib and nd2 dependencies, which are only needed by functions in that
file this pipeline doesn't use.

Known issue carried over unchanged: see future_considerations.md for the
DoG step's temporal blur.
"""

import cv2
import numpy as np
import tifffile as tiff
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_triangle


def _difference_of_gaussians(stack, sigma1, sigma2):
    """NOTE: gaussian_filter's sigma here applies to all three axes of
    stack (frames, height, width), including frames -- this blurs across
    time as well as space, not spatial-only as intended. Left unchanged;
    see future_considerations.md."""
    stack = stack.astype(np.float32)
    blur1 = gaussian_filter(stack, sigma=sigma1) if sigma1 > 0 else stack
    blur2 = gaussian_filter(stack, sigma=sigma2)
    return blur1 - blur2


def _binary_mask(dog_stack):
    mask_stack = np.zeros_like(dog_stack, dtype=np.uint16)
    for frame in range(dog_stack.shape[0]):
        image = dog_stack[frame]
        thresh_val = threshold_triangle(image)
        _, bin_mask = cv2.threshold(image, thresh_val, 65535, cv2.THRESH_BINARY)
        mask_stack[frame] = bin_mask
    return mask_stack


def _extract_contours(mask_stack):
    contours_per_frame = []
    for frame in range(mask_stack.shape[0]):
        bin_img = mask_stack[frame].astype(np.uint8)
        bin_img[bin_img > 0] = 255
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_per_frame.append(contours)
    return contours_per_frame


def _draw_contours(mask_stack, contours_per_frame):
    drawn_stack = []
    for i, contours in enumerate(contours_per_frame):
        canvas = np.zeros_like(mask_stack[i], dtype=np.uint8)
        cv2.drawContours(canvas, contours, -1, color=255, thickness=1)
        drawn_stack.append(canvas)
    return np.stack(drawn_stack)


def c1_dec_mask_raw(c1_dec_img_path, c1_raw_img_path, celloutdir):
    """Builds the ER binary mask from the deconvolved C1 stack (DoG +
    threshold), saves the mask and its contours, then applies the mask to
    the raw C1 video and saves c1_raw_masked.tif -- what downstream state
    classification actually reads."""
    stack = tiff.imread(c1_dec_img_path)
    print(f"Input shape: {stack.shape}")

    dog_stack = _difference_of_gaussians(stack, sigma1=1, sigma2=3)
    binary_stack = _binary_mask(dog_stack)

    contours = _extract_contours(binary_stack)
    contour_drawing = _draw_contours(binary_stack, contours)

    tiff.imwrite(f"{celloutdir}/c1_binary_mask.tif", binary_stack, imagej=True)
    tiff.imwrite(f"{celloutdir}/contours.tif", contour_drawing, imagej=True)

    raw_video = tiff.imread(c1_raw_img_path)
    mask_normalized = (binary_stack > 0).astype(np.float32)
    masked_raw = raw_video * mask_normalized

    tiff.imwrite(f"{celloutdir}/c1_raw_masked.tif", masked_raw.astype(raw_video.dtype), imagej=True)


def _greedy_maximal_spacing(candidates, n_select):
    """Greedily selects n_select points from candidates, maximizing the
    minimum distance between selected points: start near the centroid,
    then repeatedly add whichever remaining candidate is farthest from
    everything already selected."""
    n_candidates = len(candidates)
    if n_candidates <= n_select:
        return candidates.tolist()

    center_y, center_x = np.mean(candidates, axis=0)
    distances_from_center = np.sqrt(
        (candidates[:, 0] - center_y) ** 2 + (candidates[:, 1] - center_x) ** 2
    )
    start_idx = np.argmin(distances_from_center)

    selected_indices = [start_idx]
    selected_coords = [candidates[start_idx]]

    min_distances = np.full(n_candidates, np.inf)
    for i in range(n_candidates):
        if i != start_idx:
            min_distances[i] = np.sqrt(
                (candidates[i, 0] - candidates[start_idx, 0]) ** 2 +
                (candidates[i, 1] - candidates[start_idx, 1]) ** 2
            )

    for _ in range(n_select - 1):
        min_distances[selected_indices] = -1  # exclude already selected
        next_idx = np.argmax(min_distances)

        selected_indices.append(next_idx)
        selected_coords.append(candidates[next_idx])

        new_point = candidates[next_idx]
        for i in range(n_candidates):
            if i not in selected_indices:
                dist = np.sqrt(
                    (candidates[i, 0] - new_point[0]) ** 2 +
                    (candidates[i, 1] - new_point[1]) ** 2
                )
                min_distances[i] = min(min_distances[i], dist)

    return [(int(y), int(x)) for y, x in selected_coords]
