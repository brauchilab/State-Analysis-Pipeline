import atrouse_algorithm
import tifffile
import numpy as np
from glob import glob
import os
from scipy.ndimage import gaussian_filter

# Same calibration reference as the trackpy scaling in the liso script:
# these sigma values were tuned at ~0.1 um/px on the original 19 cells.
# A lysosome (or any real feature) covers proportionally more pixels at a
# finer pixel size, so the pre-smoothing and DoG band-pass need to widen
# by the same scale factor to keep picking out the same PHYSICAL feature
# size, not a fixed pixel size.
REFERENCE_PIXEL_SIZE_UM = 0.1

BASE_PRESMOOTH_SIGMA = 2   # gaussian_filter sigma (px), spatial-only: (0, 2, 2)
BASE_DOG_SIGMA1 = 1        # DoG g1 sigma (px), spatial-only: (0, 1, 1)
BASE_DOG_SIGMA2 = 3        # DoG g2 sigma (px), spatial-only: (0, 3, 3)

# scale_number/sigma_th/detection_level (passed to atrouse_algorithm.recursive_atrous)
# stay fixed across pixel sizes. sigma_th and detection_level are
# statistical thresholds (MAD multiplier, correlation cutoff), not
# spatial measurements, so pixel size doesn't apply to them directly.
# scale_number sets how many wavelet scales get examined, which loosely
# tracks feature size but doesn't scale linearly the way a Gaussian sigma
# does. Tune by hand if mask quality on finer-pixel cells looks off.


class MaskProcessor:
    def __init__(self):
        self.in_folder = None
        self.out_folder = None
        self.image_basename = None
        self.presmooth_sigma = BASE_PRESMOOTH_SIGMA
        self.dog_sigma1 = BASE_DOG_SIGMA1
        self.dog_sigma2 = BASE_DOG_SIGMA2

    def _set_scaled_sigmas(self, pixel_size_um):
        """Scale the spatial sigmas for this cell's pixel size, or fall back
        to the original (unscaled) values when pixel_size_um is None -- e.g.
        the original 19 cells, already at the reference pixel size."""
        if pixel_size_um is None:
            self.presmooth_sigma = BASE_PRESMOOTH_SIGMA
            self.dog_sigma1 = BASE_DOG_SIGMA1
            self.dog_sigma2 = BASE_DOG_SIGMA2
            return

        scale = REFERENCE_PIXEL_SIZE_UM / pixel_size_um  # >1 if this cell is finer
        self.presmooth_sigma = BASE_PRESMOOTH_SIGMA * scale
        self.dog_sigma1 = BASE_DOG_SIGMA1 * scale
        self.dog_sigma2 = BASE_DOG_SIGMA2 * scale
        print(f"  DoG mask sigmas scaled for pixel size {pixel_size_um} um/px -> "
              f"scale factor {scale:.3f}: presmooth={self.presmooth_sigma:.2f}, "
              f"dog_sigma1={self.dog_sigma1:.2f}, dog_sigma2={self.dog_sigma2:.2f}")

    def process_batch(self, in_folder, out_folder, pixel_size_um=None):
        self._set_scaled_sigmas(pixel_size_um)
        self.in_folder = in_folder
        self.out_folder = out_folder
        images_in_folder = glob(os.path.join(self.in_folder, "*.tif"))
        for image_path in images_in_folder:
            self.image_basename = os.path.splitext(os.path.basename(image_path))[0]
            self.mask(image_path)

    def process_single(self, img_path, out_folder, pixel_size_um=None):
        self._set_scaled_sigmas(pixel_size_um)
        self.out_folder = out_folder
        self.image_basename = os.path.splitext(os.path.basename(img_path))[0]
        self.mask(img_path)

    @staticmethod
    def _safe_normalize(arr, denom):
        """Per-frame arr/denom without producing NaN/inf when denom is 0 for
        a frame -- e.g. a fully dark/blacked-out acquisition frame, whose max
        (and therefore stack_max/d_max/dog_f_max) is legitimately 0. Those
        frames come out as all-zero instead of NaN, every other frame is
        normalized exactly as before."""
        safe_denom = np.where(denom == 0, 1, denom)
        result = arr / safe_denom
        return np.where(denom == 0, 0, result)

    def mask(self, image_path):
        print(f"  processing {self.image_basename}...")

        image = tifffile.imread(image_path).astype(np.float32)
        image = gaussian_filter(image, sigma=(0, self.presmooth_sigma, self.presmooth_sigma))
        stack_max = image.max(axis=(1, 2), keepdims=True)
        image = self._safe_normalize(image, stack_max)
        image = self.DoG(image)

        frames, width, height = image.shape
        mask_stack = np.zeros_like(image, dtype=np.uint8)
        for f in range(frames):
            img_f = image[f]
            mask_f = atrouse_algorithm.recursive_atrous(img_f,
                                                        scale_number=3,
                                                        sigma_th=5,
                                                        detection_level=1)
            mask_stack[f] = mask_f
            if f % 10 == 0:
                print(f"    frame {f}/{frames}", end="\r")
        self.save_tif(mask_stack)

    def DoG(self, stack):
        g1 = gaussian_filter(stack, sigma=(0, self.dog_sigma1, self.dog_sigma1))
        g2 = gaussian_filter(stack, sigma=(0, self.dog_sigma2, self.dog_sigma2))
        dog = g1 - g2

        d_max = dog.max(axis=(1, 2), keepdims=True)
        dog = np.maximum(dog, 0)
        dog = self._safe_normalize(dog, d_max)

        dog_f = dog + stack
        dog_f_max = dog_f.max(axis=(1, 2), keepdims=True)
        dog_f = self._safe_normalize(dog_f, dog_f_max)
        return dog_f

    def save_tif(self, stack):
        save_path = os.path.join(self.out_folder, f"{self.image_basename}_mask.tif")
        tifffile.imwrite(save_path, stack)
