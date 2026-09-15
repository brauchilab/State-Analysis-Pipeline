
from skimage.restoration import denoise_tv_chambolle
from scipy.ndimage import gaussian_filter
import numpy as np


def denoise_stack(tif_stack,
                  d_weight = 0.01):
    
    frames, h, w = tif_stack.shape

    denoised_stack = np.zeros_like(tif_stack, dtype=np.uint16)

    for f in range(frames):
        current_frame = tif_stack[f].astype(np.float32)
        background = gaussian_filter(current_frame, sigma=10)
        bg_removed_frame = current_frame-background
        bg_removed_frame = np.maximum(bg_removed_frame,0)

        normalized_frame = bg_removed_frame/np.max(bg_removed_frame) # 0 - 1 normalization

        denoised_frame = denoise_tv_chambolle(normalized_frame, weight=d_weight)

        denoised_16b = np.round(denoised_frame*65535).astype(np.uint16)
        denoised_stack[f] = denoised_16b

    return denoised_stack



