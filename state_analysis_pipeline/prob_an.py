import numpy as np
from tqdm import tqdm
from scipy.stats import norm

def predict_states_probabilities_vectorized(dff_data, emission_means, emission_stds):
    """Fast vectorized prediction returning probabilities for each state using trained HMM parameters"""
    print("Starting vectorized state probability prediction...")
    
    frames, height, width = dff_data.shape
    print(f"Processing video data: {frames} frames, {height}x{width} pixels")
    
    # Initialize empty object array and fill with (-1,-1,-1) tuples for empty pixels
    state_results = np.empty((frames, height, width), dtype=object)
    state_results.fill((-1, -1, -1))
    
    for frame_idx in tqdm(range(frames), desc="Processing frames"):
        frame = dff_data[frame_idx]
        non_zero_mask = frame != 0
        
        if np.any(non_zero_mask):
            non_zero_values = frame[non_zero_mask]
            
            # Vectorized probability calculation for all 3 states
            probs = np.zeros((len(non_zero_values), 3))
            
            for state in range(3):
                probs[:, state] = norm.pdf(non_zero_values, 
                                         emission_means[state], 
                                         emission_stds[state])
            
            # Normalize probabilities so they sum to 1 for each pixel
            prob_sums = np.sum(probs, axis=1, keepdims=True)
            # Avoid division by zero
            prob_sums[prob_sums == 0] = 1
            normalized_probs = probs / prob_sums
            
            # Convert to tuples and assign back to result array
            for i, (row, col) in enumerate(np.argwhere(non_zero_mask)):
                state_results[frame_idx, row, col] = tuple(normalized_probs[i])
    
    return state_results
