import numpy as np
import time
import os
import tifffile

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt

import pandas as pd
import random

# Memory-map the arrays for better memory management
def create_memmap_array(shape, filename=None, dtype=np.float32):
    """Create a memory-mapped array for large datasets"""
    if filename is None:
        filename = f"memmap_temp_{os.getpid()}.dat"
    
    memmap_array = np.memmap(filename, dtype=dtype, mode='w+', shape=shape)
    return memmap_array, filename
# Use memory-mapped arrays for large data

def generate_rois_vectorized(image_shape, roi_size):
    """Generate ROIs using vectorized operations"""
    height, width = image_shape
    roi_w, roi_h = roi_size
    
    # Calculate grid dimensions
    cols = int(np.ceil(width / roi_w))
    rows = int(np.ceil(height / roi_h))

    
    # Create 2D arrays of coordinates for better vectorization
    col_indices, row_indices = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Convert to 1D arrays
    x_coords = (col_indices.flatten() * roi_w).astype(np.int32)
    y_coords = (row_indices.flatten() * roi_h).astype(np.int32)
    
    # Calculate dimensions with boundary handling (vectorized)
    widths = np.minimum(roi_w, width - x_coords).astype(np.int32)
    heights = np.minimum(roi_h, height - y_coords).astype(np.int32)
    
    # Create ROIs as a structured numpy array for better memory access patterns
    dtype = np.dtype([('x', np.int32), ('y', np.int32), ('w', np.int32), ('h', np.int32)])
    rois = np.zeros(len(x_coords), dtype=dtype)
    rois['x'] = x_coords
    rois['y'] = y_coords
    rois['w'] = widths
    rois['h'] = heights
    
    return rois, cols, rows, (cols,rows)

def define_neighbors_for_all_rois(rois_per_row, rois_per_col):
    """Pre-compute all neighbors for all ROIs at once"""
    total_rois = rois_per_row * rois_per_col
    neighbors_list = []
    
    # Create a 2D grid representation of ROIs
    roi_grid = np.arange(total_rois).reshape(rois_per_col, rois_per_row)
    
    # Define neighbor offsets (excluding self)
    offsets = [(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1] if not (i == 0 and j == 0)]
    
    for roi_idx in range(total_rois):
        row = roi_idx // rois_per_row
        col = roi_idx % rois_per_row
        
        # Find all valid neighbors
        neighbors = []
        for dy, dx in offsets:
            new_row, new_col = row + dy, col + dx
            if 0 <= new_row < rois_per_col and 0 <= new_col < rois_per_row:
                neighbors.append(roi_grid[new_row, new_col])
        
        neighbors_list.append(neighbors)
    
    return neighbors_list

def process_frame_range_optimized(frame, rois):
    """Calculate intensities for all ROIs in a frame with vectorized operations where possible"""
    roi_intensities = np.full(len(rois), fill_value = np.nan, dtype=np.float32)
    
    # Extract ROI data with a more efficient access pattern
    for i in range(len(rois)):
        roi = rois[i]
        x, y, w, h = roi['x'], roi['y'], roi['w'], roi['h']
        
        # Get ROI pixels directly
        roi_pixels = frame[y:y+h, x:x+w]
        non_zero_mask = roi_pixels > 0
        
        if np.any(non_zero_mask):
            roi_intensities[i] = np.mean(roi_pixels[non_zero_mask])
    
    return roi_intensities

def calculate_f0_per_roi_optimized(avg_roi_array, frame_range, neighbors_list):
    """Calculate baseline fluorescence for each ROI with vectorized operations"""
    start_frame, end_frame = frame_range
    frame_indices = np.arange(start_frame, end_frame + 1)
    num_rois = len(neighbors_list)
    
    # Pre-allocate arrays for better memory usage
    f0_per_roi = np.zeros(num_rois, dtype=np.float32)
    
    for roi_idx in range(num_rois):
        neighbors = neighbors_list[roi_idx]
        roi_and_neighbors = neighbors + [roi_idx]

        subarray = avg_roi_array[start_frame:end_frame+1, roi_and_neighbors]
        subarray = np.where(subarray == 0, np.nan, subarray)

        frame_means = np.nanmean(subarray, axis = 1)
        f0_per_roi[roi_idx] = np.nanmean(frame_means)
        
    
    return f0_per_roi

def create_all_roi_masks(rois, img_shape):
    """Pre-compute masks for all ROIs"""
    height, width = img_shape
    all_masks = []
    
    for i in range(len(rois)):
        roi = rois[i]
        x, y, w, h = roi['x'], roi['y'], roi['w'], roi['h']
        
        # Create sparse representation of mask: just store coordinates
        y_indices = slice(y, min(y+h, height))
        x_indices = slice(x, min(x+w, width))
        all_masks.append((y_indices, x_indices))
    
    return all_masks

def dff_pixel_optimized(frame, all_masks, f0_per_roi):
    """Apply dF/F transformation with pre-computed masks and vectorized operations"""
    result_frame = np.zeros_like(frame, dtype=np.float32)
    
    for roi_index, (y_slice, x_slice) in enumerate(all_masks):
        f0 = f0_per_roi[roi_index]
        if f0 <= 0:
            continue
        
        # Get ROI pixels directly using the pre-computed slices
        roi_pixels = frame[y_slice, x_slice]
        
        # Create mask for non-zero pixels
        non_zero_mask = roi_pixels > 0
        
        if np.any(non_zero_mask):
            # Create temporary result array
            temp_result = np.zeros_like(roi_pixels, dtype=np.float32)
            
            # Calculate dF/F only for non-zero pixels
            temp_result[non_zero_mask] = (roi_pixels[non_zero_mask] - f0) / f0
            
            # Assign results back to the frame
            result_frame[y_slice, x_slice] = temp_result
    
    return result_frame

def process_frame_frame_optimized(image_array, roi_size, frame_range, use_memmap=False):
    """Main processing function with optimized operations"""
    frames = image_array.shape[0]
    height, width = image_array.shape[1:]
    
    print("Starting optimized processing")
    
    # Generate ROIs only once at the beginning 
    print("Generating ROIs...")
    rois, cols, rows, rois_shape = generate_rois_vectorized((height, width), roi_size)
    
    # Pre-compute neighbors for all ROIs
    print("Computing neighbors...")
    neighbors_list = define_neighbors_for_all_rois(cols, rows)
    #print(neighbors_list)
    
    # Pre-compute masks for all ROIs
    print("Creating ROI masks...")
    all_masks = create_all_roi_masks(rois, (height, width))
    #print(all_masks)
    
    # Pre-allocate ROI intensities array instead of using a list
    all_roi_intensities = np.zeros((frames, len(rois)), dtype=np.float32)

    # Process all frames to get ROI intensities
    print("Processing frame intensities...")
    for frame_idx in range(frames):
        frame = image_array[frame_idx]
        all_roi_intensities[frame_idx] = process_frame_range_optimized(frame, rois)
        
        if frame_idx % 10 == 0 or frame_idx == frames-1:
            print(f"Processed roi_intensities of frame {frame_idx+1}/{frames}")
    print(all_roi_intensities.shape)
    
    #np.savetxt('array_contents.txt', all_roi_intensities, fmt='%f')
    # Calculate F0 for each ROI
    print("Calculating F0 values...")
    f0_per_roi = calculate_f0_per_roi_optimized(all_roi_intensities, frame_range, neighbors_list)
    
    # Create output array, optionally memory-mapped
    if use_memmap:
        dff_result, memmap_filename = create_memmap_array((frames, height, width))
    else:
        dff_result = np.zeros((frames, height, width), dtype=np.float32)
    
    # Process all frames for dF/F calculation
    print("Calculating dF/F for all frames...")
    for frame_idx in range(frames):
        dff_result[frame_idx] = dff_pixel_optimized(image_array[frame_idx], all_masks, f0_per_roi)
        
        if frame_idx % 10 == 0 or frame_idx == frames-1:
            print(f"Processed pixels of frame {frame_idx+1}/{frames}")
    
    return dff_result

################################

def compare_frames_animation(img_display,dff_display):

    # Example: Using your image stacks
    # img_display and dff_display shape: (num_frames, height, width)
    num_frames = img_display.shape[0]
    
    # Create subplot figure with 2 columns
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=("Original Image", "ΔF/F Heatmap"),
                        horizontal_spacing=0.02)  # Reduced spacing between subplots
    
    # Initial images
    fig.add_trace(
        go.Heatmap(z=img_display[0], colorscale='Gray', 
                   colorbar=dict(title="Original", x=0.48)),  # Adjusted position
        row=1, col=1
    )
    fig.add_trace(
        go.Heatmap(z=dff_display[0], colorscale='RdBu', zmin=-1, zmax=1, 
                   colorbar=dict(title="ΔF/F", x=1.02)),
        row=1, col=2
    )
    
    frames = []
    for i in range(num_frames):
        frames.append(go.Frame(
            data=[
                go.Heatmap(z=img_display[i], colorscale='Gray',
                          colorbar=dict(title="Original", x=0.48)),   # original image
                go.Heatmap(z=dff_display[i], colorscale='RdBu', zmin=-1, zmax=1,
                          colorbar=dict(title="ΔF/F", x=1.02)),  # dff heatmap
            ],
            name=f'frame{i}'
        ))
    
    # Slider steps
    steps = [
        dict(
            method="animate",
            args=[[f"frame{k}"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
            label=str(k)
        ) for k in range(num_frames)
    ]
    
    fig.update_layout(
        width=1200,  # Increased width
        height=600,  # Increased height
        sliders=[dict(
            active=0,
            currentvalue={"prefix": "Frame: "},
            pad={"t": 50},
            steps=steps
        )],
        # Fix aspect ratio for each subplot axis
        xaxis=dict(scaleanchor="y", scaleratio=1, constrain='domain'),
        yaxis=dict(scaleanchor="x", scaleratio=1, autorange='reversed'),
        xaxis2=dict(scaleanchor="y2", scaleratio=1, constrain='domain'),
        yaxis2=dict(scaleanchor="x2", scaleratio=1, autorange='reversed'),
        # Reduce margins to maximize plot area
        margin=dict(l=50, r=50, t=80, b=80)
    )
    
    fig.frames = frames
    fig.show()

#######

def show_all_frames(img_roi,dff_result):
# Assuming you already have:
# img_roi (shape: frames, height, width)
# dff_result (same shape)

    for i in range(img_roi.shape[0]):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        ax1.imshow(img_roi[i], cmap='gray', vmin=0, vmax=255)
        ax1.set_title(f"Original - Frame {i+1}")
        ax1.axis('off')
        
        ax2.imshow(dff_result[i], cmap='bwr', vmin=-1, vmax=1)
        ax2.set_title(f"ΔF/F - Frame {i+1}")
        ax2.axis('off')
        
        plt.show()



def numeric_frame_check(img_roi,dff_result):
    # Compare original stack
    print("=== Original Image Stack ===")
    for i in range(img_roi.shape[0]):
        diff = np.abs(img_roi[i] - img_roi[0]).sum()
        if diff == 0:
            print(f"Frame {i} is IDENTICAL to Frame 0")
        else:
            print(f"Frame {i} differs from Frame 0 (sum abs diff = {diff:.2f})")
    
    # Compare ΔF/F results
    print("\n=== ΔF/F Result Stack ===")
    for i in range(dff_result.shape[0]):
        diff = np.abs(dff_result[i] - dff_result[0]).sum()
        if diff == 0:
            print(f"Frame {i} is IDENTICAL to Frame 0")
        else:
            print(f"Frame {i} differs from Frame 0 (sum abs diff = {diff:.6f})")

################################################

def sample_pois(
    dff_results,
    skip_step=10,
    max_pois=1000,
    empty_threshold=0.2,
    output_csv="poi_timeseries.csv",
    mode="diagonal",     # "diagonal", "horizontal", "vertical"
    slope=1.0,           # slope for diagonal mode (y = slope * x)
    randomize=False,
    random_seed=None
):
    """
    Sample points of interest (POIs) from a 3D (frames, height, width) array.

    Parameters
    ----------
    dff_results : np.ndarray
        Array of shape (frames, height, width).
    skip_step : int
        Number of steps to skip between POIs.
    max_pois : int
        Maximum number of POIs to collect.
    empty_threshold : float
        Fraction of frames allowed to be empty (0 or NaN) before a pixel is invalid.
    output_csv : str
        File path to save CSV.
    mode : str
        "diagonal" (default), "horizontal", or "vertical".
    slope : float
        Slope of diagonal (only used if mode="diagonal").
    randomize : bool
        If True, randomly sample candidates (still reproducible with random_seed).
    random_seed : int or None
        Seed for RNG if randomize=True.

    Returns
    -------
    pois : list of tuple
        List of (y, x) coordinates of selected POIs.
    """
    frames, height, width = dff_results.shape

    # Candidate coordinates
    if mode == "diagonal":
        max_x = width
        coords = []
        for x in range(width):
            y = int(round(slope * x))
            if 0 <= y < height:
                coords.append((y, x))
    elif mode == "horizontal":
        y = height // 2  # middle row
        coords = [(y, x) for x in range(width)]
    elif mode == "vertical":
        x = width // 2  # middle column
        coords = [(y, x) for y in range(height)]
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'diagonal', 'horizontal', or 'vertical'.")

    # Optionally randomize order
    if randomize:
        rng = random.Random(random_seed)
        rng.shuffle(coords)

    pois = []
    i = 0
    while i < len(coords) and len(pois) < max_pois:
        y, x = coords[i]
        ts = dff_results[:, y, x]
        empty_frac = np.mean((ts == 0) | np.isnan(ts))

        if empty_frac <= empty_threshold:
            pois.append((y, x))
            i += skip_step
        else:
            # look ahead for the next valid pixel
            j = i + 1
            found = False
            while j < len(coords):
                y2, x2 = coords[j]
                ts2 = dff_results[:, y2, x2]
                empty_frac2 = np.mean((ts2 == 0) | np.isnan(ts2))
                if empty_frac2 <= empty_threshold:
                    pois.append((y2, x2))
                    i = j + skip_step
                    found = True
                    break
                j += 1
            if not found:
                break  # no more valid pixels

    # Build DataFrame
    data = {"frame": np.arange(frames)}
    for (y, x) in pois:
        data[f"({y},{x})"] = dff_results[:, y, x]

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)

    print(f"Saved {len(pois)} POIs to {output_csv}")
    return pois
    
################################################
def remove_positive_outliers_iqr(data, iqr_multiplier=1.5):
    """
    Remove outliers only on the positive side using IQR method.
    
    Parameters:
    - data: numpy array of any shape
    - iqr_multiplier: multiplier for IQR (1.5 is standard, 3.0 is more conservative)
    
    Returns:
    - data with positive outliers clipped to upper threshold
    """
    # Calculate quartiles
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    # Calculate upper threshold only
    upper_threshold = q3 + (iqr_multiplier * iqr)
    
    # Clip only positive outliers (keep -1 as lower bound)
    dff_result_cleaned = np.clip(data, -1, upper_threshold)
    
    return dff_result_cleaned


def remove_positive_outliers_percentile(data, upper_percentile=99):
    """
    Remove outliers only on the positive side using percentile method.
    
    Parameters:
    - data: numpy array of any shape
    - upper_percentile: percentile to use as upper threshold (e.g., 95, 99, 99.5)
    
    Returns:
    - data with positive outliers clipped to upper percentile
    """
    upper_threshold = np.percentile(data, upper_percentile)
    
    # Clip only positive outliers (keep -1 as lower bound)
    dff_result_cleaned = np.clip(data, -1, upper_threshold)
    
    return dff_result_cleaned
    
    
def remove_positive_outliers_zscore(data, z_threshold=3):
    """
    Remove outliers only on the positive side using z-score method.
    
    Parameters:
    - data: numpy array of any shape
    - z_threshold: z-score threshold (typically 2.5-3)
    
    Returns:
    - data with positive outliers clipped
    """
    mean = np.mean(data)
    std = np.std(data)
    
    # Calculate upper threshold only
    upper_threshold = mean + (z_threshold * std)
    
    # Clip only positive outliers (keep -1 as lower bound)
    dff_result_cleaned = np.clip(data, -1, upper_threshold)
    
    return dff_result_cleaned
