
import numpy as np
from scipy.stats import median_abs_deviation


def conv_1d(signal, kernel):
    signal = np.array(signal)
    kernel_size = len(kernel)
    pad_size = (kernel_size -1 ) // 2
    padded_signal = np.pad(signal, pad_size, mode="reflect")

    return np.convolve(padded_signal, kernel, mode = "valid")

def conv_2d(array_2d, kernel_1d):
    temp = np.apply_along_axis(lambda x: conv_1d(x, kernel_1d), 1, array_2d)
    return np.apply_along_axis(lambda x: conv_1d(x, kernel_1d), 0, temp)

def dilate_kernel(kernel, scale):
    num_zeros = 2**(scale-1) - 1

    if scale == 0:
        return kernel
    new_length = len(kernel) + (len(kernel) - 1) * num_zeros
    dilated = np.zeros(new_length)
    spacing = num_zeros + 1
    dilated[::spacing] = kernel
    return dilated

def wj(a1, a2):
    return a1 - a2

def recursive_atrous(array_2d, scale_number = 3, sigma_th = 5, detection_level = 5):

    kernel = np.array([1/16, 1/4, 3/8, 1/4, 1/16])

    frame_h, frame_w = array_2d.shape

    details_array = np.zeros((scale_number+1, frame_h, frame_w), dtype=np.float32)
    coefficients_array = np.zeros((scale_number, frame_h, frame_w), dtype=np.float32)
    details_array[0] = array_2d

    for scale in range(1, scale_number+1):
        conv_kernel = dilate_kernel(kernel, scale)
        details_frame = conv_2d(details_array[scale-1], conv_kernel) # calculate Aj from Aj-1
        coefficients_array[scale-1] = details_array[scale-1] - details_frame
        details_array[scale] =details_frame

    mad = [median_abs_deviation(coeff_plane.ravel(), scale = "normal") for coeff_plane in coefficients_array]
    thresholded_img = np.zeros_like(coefficients_array)

    for i in range(coefficients_array.shape[0]):
        thresholded_img[i] = np.where(coefficients_array[i] >= (sigma_th*mad[i]), coefficients_array[i], 0)
    correlation_img = np.prod(np.abs(thresholded_img), axis = 0)
    correlation_img = correlation_img / np.max(correlation_img)*255
    mask = np.where(correlation_img > detection_level, 255, 0).astype(np.uint8)

    return mask





