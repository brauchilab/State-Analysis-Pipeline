"""
Save/Load utilities for state probability results

This module provides efficient save/load functions for state probability arrays,
with automatic conversion between tuple format (runtime) and numpy array format (storage).
"""

import numpy as np
import os


def save_state_probs(state_probs, filepath, compress=True):
    """
    Save state probabilities to disk in efficient numpy format.
    
    Converts from object array with tuples to float32 array for storage.
    
    Parameters:
    -----------
    state_probs : ndarray (frames, height, width) with dtype=object
        State probabilities as tuples (p0, p1, p2) or (-1, -1, -1)
    filepath : str
        Path to save file (will add .npz extension if not present)
    compress : bool
        If True, use compressed format (smaller file, slightly slower)
        
    Returns:
    --------
    filepath : str
        Full path where file was saved
    """
    print(f"Saving state probabilities to {filepath}...")
    
    # Ensure filepath has correct extension
    if compress and not filepath.endswith('.npz'):
        filepath = filepath.replace('.npy', '') + '.npz'
    elif not compress and not filepath.endswith('.npy'):
        filepath = filepath.replace('.npz', '') + '.npy'
    
    # Get dimensions
    frames, height, width = state_probs.shape
    
    # Convert to efficient format: (frames, height, width, 3) float32
    state_array = np.full((frames, height, width, 3), np.nan, dtype=np.float32)
    
    print(f"Converting {frames}x{height}x{width} object array to float32 array...")
    
    # Convert tuples to array
    for f in range(frames):
        for y in range(height):
            for x in range(width):
                value = state_probs[f, y, x]
                if value != (-1, -1, -1) and isinstance(value, tuple):
                    state_array[f, y, x] = value
    
    # Save with metadata
    if compress:
        np.savez_compressed(
            filepath,
            state_probs=state_array,
            shape=np.array([frames, height, width]),
            dtype_original='object_tuples',
            n_states=3
        )
    else:
        np.save(filepath, state_array)
    
    # Get file size
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"✓ Saved to {filepath}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Format: {'compressed npz' if compress else 'uncompressed npy'}")
    
    return filepath


def load_state_probs(filepath, return_tuples=True):
    """
    Load state probabilities from disk.
    
    Automatically converts back to tuple format if needed.
    
    Parameters:
    -----------
    filepath : str
        Path to saved file (.npz or .npy)
    return_tuples : bool
        If True, return object array with tuples (backward compatible)
        If False, return float32 array (more efficient)
        
    Returns:
    --------
    state_probs : ndarray
        If return_tuples=True: (frames, height, width) object array with tuples
        If return_tuples=False: (frames, height, width, 3) float32 array
    """
    print(f"Loading state probabilities from {filepath}...")
    
    # Load the data
    if filepath.endswith('.npz'):
        data = np.load(filepath)
        state_array = data['state_probs']
        print(f"  Loaded from compressed npz format")
    else:
        state_array = np.load(filepath)
        print(f"  Loaded from uncompressed npy format")
    
    frames, height, width, n_states = state_array.shape
    print(f"  Shape: {frames}x{height}x{width}, {n_states} states")
    
    # Get file size
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB")
    
    if not return_tuples:
        # Return as-is (efficient format)
        print(f"✓ Returning float32 array")
        return state_array
    
    # Convert back to tuple format for backward compatibility
    print(f"Converting to tuple format for backward compatibility...")
    state_probs = np.empty((frames, height, width), dtype=object)
    state_probs.fill((-1, -1, -1))
    
    for f in range(frames):
        for y in range(height):
            for x in range(width):
                if not np.isnan(state_array[f, y, x, 0]):
                    state_probs[f, y, x] = tuple(state_array[f, y, x])
    
    print(f"✓ Returning object array with tuples")
    return state_probs


def test_save_load_integrity(state_probs, test_filepath):
    """
    Test that save/load preserves data integrity.
    
    Parameters:
    -----------
    state_probs : ndarray
        State probabilities to test
    test_filepath : str
        Temporary filepath for testing
        
    Returns:
    --------
    success : bool
        True if test passed, False otherwise
    """
    print("\n" + "="*60)
    print("TESTING SAVE/LOAD INTEGRITY")
    print("="*60)
    
    # Save
    print("\n1. Saving original data...")
    saved_path = save_state_probs(state_probs, test_filepath, compress=True)
    
    # Load
    print("\n2. Loading saved data...")
    loaded_probs = load_state_probs(saved_path, return_tuples=True)
    
    # Compare
    print("\n3. Comparing original vs loaded...")
    frames, height, width = state_probs.shape
    
    mismatches = 0
    total_compared = 0
    max_diff = 0.0
    
    for f in range(frames):
        for y in range(height):
            for x in range(width):
                orig = state_probs[f, y, x]
                load = loaded_probs[f, y, x]
                
                # Both should be tuples or both should be (-1,-1,-1)
                if orig == (-1, -1, -1) and load == (-1, -1, -1):
                    continue
                
                if orig == (-1, -1, -1) or load == (-1, -1, -1):
                    print(f"  ✗ Mismatch at ({f},{y},{x}): orig={orig}, loaded={load}")
                    mismatches += 1
                    continue
                
                # Compare values
                orig_arr = np.array(orig)
                load_arr = np.array(load)
                diff = np.abs(orig_arr - load_arr)
                max_diff = max(max_diff, np.max(diff))
                
                if np.max(diff) > 1e-6:
                    print(f"  ✗ Value mismatch at ({f},{y},{x}): max_diff={np.max(diff):.2e}")
                    mismatches += 1
                
                total_compared += 1
    
    # Results
    print("\n" + "-"*60)
    print("RESULTS:")
    print(f"  Total pixels compared: {total_compared}")
    print(f"  Mismatches: {mismatches}")
    print(f"  Max numerical difference: {max_diff:.2e}")
    
    # Check file size reduction
    if hasattr(state_probs, 'nbytes'):
        original_size_mb = state_probs.nbytes / (1024 * 1024)
        saved_size_mb = os.path.getsize(saved_path) / (1024 * 1024)
        reduction_pct = (1 - saved_size_mb / original_size_mb) * 100
        print(f"\n  Original size (in memory): {original_size_mb:.2f} MB")
        print(f"  Saved size (on disk): {saved_size_mb:.2f} MB")
        print(f"  Size reduction: {reduction_pct:.1f}%")
    
    # Clean up test file
    if os.path.exists(saved_path):
        os.remove(saved_path)
        print(f"\n  Cleaned up test file: {saved_path}")
    
    print("="*60)
    
    if mismatches == 0:
        print("✓✓✓ TEST PASSED ✓✓✓")
        print("Save/load preserves data perfectly!")
        return True
    else:
        print("✗✗✗ TEST FAILED ✗✗✗")
        print(f"Found {mismatches} mismatches")
        return False


def quick_save(state_probs, cell_name, output_dir, method="global"):
    """
    Convenience function to save with automatic naming.
    
    Parameters:
    -----------
    state_probs : ndarray
        State probabilities
    cell_name : str
        Name of the cell
    output_dir : str
        Output directory
    method : str
        Training method used ("global" or "roi")
        
    Returns:
    --------
    filepath : str
        Path where file was saved
    """
    filename = f"{cell_name}_state_probs_{method}.npz"
    filepath = os.path.join(output_dir, filename)
    return save_state_probs(state_probs, filepath, compress=True)


def quick_load(cell_name, output_dir, method="global", return_tuples=True):
    """
    Convenience function to load with automatic naming.
    
    Parameters:
    -----------
    cell_name : str
        Name of the cell
    output_dir : str
        Output directory
    method : str
        Training method used ("global" or "roi")
    return_tuples : bool
        Return format
        
    Returns:
    --------
    state_probs : ndarray
        Loaded state probabilities
    """
    filename = f"{cell_name}_state_probs_{method}.npz"
    filepath = os.path.join(output_dir, filename)
    return load_state_probs(filepath, return_tuples=return_tuples)


# Example usage
if __name__ == "__main__":
    print("State Probabilities Save/Load Utilities")
    print("="*60)
    print("\nExample usage:")
    print("""
# After computing state probabilities:
state_probs = c1_state_probs_masked_raw(currcell, celloutdir, use_roi=True)

# Save with automatic naming:
saved_path = quick_save(state_probs, currcell['cell_name'], celloutdir, method="roi")

# Or save with custom name:
save_state_probs(state_probs, "./my_results/cell_001_probs.npz")

# Load later:
state_probs = quick_load(currcell['cell_name'], celloutdir, method="roi")

# Or load from custom name:
state_probs = load_state_probs("./my_results/cell_001_probs.npz")

# Test integrity:
test_save_load_integrity(state_probs, "./test_temp.npz")
    """)
