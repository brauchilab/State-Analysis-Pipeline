import pandas as pd
import numpy as np
from scipy import stats
import os
import re

def calculate_dff_for_tracks(csv_file, intensity_column='mean_intensity', baseline_method='robust_mean'):
    """
    Calculate dF/F for tracked particles from a CSV file.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file containing tracking data
    intensity_column : str
        Name of the column containing intensity values
    baseline_method : str
        Method for calculating F0: 'mean', 'median', 'robust_mean', 'percentile_10'
    
    Returns:
    --------
    pandas.DataFrame
        Original data with added 'df_f0' column
    """
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Ensure the dataframe is sorted by particle and frame
    df = df.sort_values(['particle', 'frame']).reset_index(drop=True)
    
    # Initialize the dF/F column
    df['df_f0'] = np.nan
    
    # Get unique particles
    unique_particles = df['particle'].unique()
    
    print(f"Processing {len(unique_particles)} particles...")
    
    for particle_id in unique_particles:
        # Get data for this particle
        particle_data = df[df['particle'] == particle_id].copy()
        
        if len(particle_data) < 2:  # Need at least 2 points
            print(f"Warning: Particle {particle_id} has less than 2 data points, skipping...")
            continue
            
        # Get intensity values (excluding any NaN or zero values)
        intensities = particle_data[intensity_column].values
        valid_mask = ~np.isnan(intensities) & (intensities > 0)
        
        if np.sum(valid_mask) < 2:
            print(f"Warning: Particle {particle_id} has insufficient valid intensity values, skipping...")
            continue
            
        valid_intensities = intensities[valid_mask]
        
        # Calculate F0 using different methods
        if baseline_method == 'mean':
            f0 = np.mean(valid_intensities)
        elif baseline_method == 'median':
            f0 = np.median(valid_intensities)
        elif baseline_method == 'robust_mean':
            # Use trimmed mean (exclude extreme values)
            f0 = stats.trim_mean(valid_intensities, 0.2)  # Trim 20% from each end
        elif baseline_method == 'percentile_10':
            # Use 10th percentile as baseline (similar to your image approach)
            f0 = np.percentile(valid_intensities, 10)
        else:
            raise ValueError(f"Unknown baseline method: {baseline_method}")
        
        if f0 <= 0:
            print(f"Warning: Particle {particle_id} has F0 <= 0 ({f0:.3f}), skipping...")
            continue
        
        # Calculate dF/F for all frames of this particle
        particle_indices = df[df['particle'] == particle_id].index
        
        for idx in particle_indices:
            intensity = df.loc[idx, intensity_column]
            if pd.notna(intensity) and intensity > 0:
                df.loc[idx, 'df_f0'] = (intensity - f0) / f0
    
    return df

def calculate_dff_with_temporal_baseline(csv_file, intensity_column='mean_intensity', 
                                       baseline_frames=None, baseline_method='robust_mean'):
    """
    Calculate dF/F using specific baseline frames (similar to your image approach).
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file containing tracking data
    intensity_column : str
        Name of the column containing intensity values
    baseline_frames : tuple or None
        (start_frame, end_frame) for baseline calculation. If None, use all frames.
    baseline_method : str
        Method for calculating F0 within baseline frames
    
    Returns:
    --------
    pandas.DataFrame
        Original data with added 'df_f0' column
    """
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Ensure the dataframe is sorted by particle and frame
    df = df.sort_values(['particle', 'frame']).reset_index(drop=True)
    
    # Initialize the dF/F column
    df['df_f0'] = np.nan
    
    # Get unique particles
    unique_particles = df['particle'].unique()
    
    print(f"Processing {len(unique_particles)} particles...")
    
    for particle_id in unique_particles:
        # Get data for this particle
        particle_data = df[df['particle'] == particle_id].copy()
        
        if len(particle_data) < 2:
            print(f"Warning: Particle {particle_id} has less than 2 data points, skipping...")
            continue
        
        # Filter baseline frames if specified
        if baseline_frames is not None:
            start_frame, end_frame = baseline_frames
            baseline_data = particle_data[
                (particle_data['frame'] >= start_frame) & 
                (particle_data['frame'] <= end_frame)
            ]
        else:
            baseline_data = particle_data
        
        if len(baseline_data) == 0:
            print(f"Warning: Particle {particle_id} has no data in baseline frames, using all frames...")
            baseline_data = particle_data
        
        # Get baseline intensity values
        baseline_intensities = baseline_data[intensity_column].values
        valid_mask = ~np.isnan(baseline_intensities) & (baseline_intensities > 0)
        
        if np.sum(valid_mask) < 1:
            print(f"Warning: Particle {particle_id} has no valid baseline intensities, skipping...")
            continue
            
        valid_baseline_intensities = baseline_intensities[valid_mask]
        
        # Calculate F0
        if baseline_method == 'mean':
            f0 = np.mean(valid_baseline_intensities)
        elif baseline_method == 'median':
            f0 = np.median(valid_baseline_intensities)
        elif baseline_method == 'robust_mean':
            f0 = stats.trim_mean(valid_baseline_intensities, 0.2)
        elif baseline_method == 'percentile_10':
            f0 = np.percentile(valid_baseline_intensities, 10)
        else:
            raise ValueError(f"Unknown baseline method: {baseline_method}")
        
        if f0 <= 0:
            print(f"Warning: Particle {particle_id} has F0 <= 0 ({f0:.3f}), skipping...")
            continue
        
        # Calculate dF/F for all frames of this particle
        particle_indices = df[df['particle'] == particle_id].index
        
        for idx in particle_indices:
            intensity = df.loc[idx, intensity_column]
            if pd.notna(intensity) and intensity > 0:
                df.loc[idx, 'df_f0'] = (intensity - f0) / f0
    
    return df

def analyze_tracking_stats(df):
    """
    Analyze the tracking data to understand frame coverage per particle.
    """
    stats_summary = []
    
    unique_particles = df['particle'].unique()
    
    for particle_id in unique_particles:
        particle_data = df[df['particle'] == particle_id]
        
        frames = particle_data['frame'].values
        min_frame = np.min(frames)
        max_frame = np.max(frames)
        total_frames = max_frame - min_frame + 1
        observed_frames = len(frames)
        missing_frames = total_frames - observed_frames
        
        stats_summary.append({
            'particle': particle_id,
            'first_frame': min_frame,
            'last_frame': max_frame,
            'total_possible_frames': total_frames,
            'observed_frames': observed_frames,
            'missing_frames': missing_frames,
            'coverage_ratio': observed_frames / total_frames
        })
    
    return pd.DataFrame(stats_summary)


################################


def filter_particles_by_length(csv_file, min_frames, output_file=None):
   """
   Filter particles that have at least min_frames data points.
   
   Parameters:
   -----------
   csv_file : str
       Path to the input CSV file
   min_frames : int
       Minimum number of frames required per particle
   output_file : str, optional
       Output filename. If None, auto-generates based on input name and threshold
   
   Returns:
   --------
   pandas.DataFrame
       Filtered dataframe
   """
   
   # Read CSV
   df = pd.read_csv(csv_file)
   
   # Count frames per particle
   particle_counts = df['particle'].value_counts()
   
   # Get particles with enough frames
   valid_particles = particle_counts[particle_counts >= min_frames].index
   
   # Filter dataframe
   filtered_df = df[df['particle'].isin(valid_particles)].copy()
   
   # Generate output filename if not provided
   if output_file is None:
       base_name = csv_file.rsplit('.', 1)[0]
       extension = csv_file.rsplit('.', 1)[1] if '.' in csv_file else 'csv'
       output_file = f"{base_name}_min{min_frames}frames.{extension}"
   
   # Save
   filtered_df.to_csv(output_file, index=False)
   
   print(f"Filtered {len(df)} → {len(filtered_df)} measurements")
   print(f"Kept {len(valid_particles)}/{df['particle'].nunique()} particles")
   print(f"Saved to: {output_file}")
   
   return filtered_df


#################


def clean_particle_tifs(directory, valid_ids, dry_run=False):
    """
    Deletes .tif files in 'directory' that are not in the valid_ids list/set.
    
    Args:
        directory (str): Path to the folder with .tif files.
        valid_ids (list or set of int): Particle IDs to keep.
        dry_run (bool): If True, only print what would be deleted (default: False).
    """
    # Make sure valid_ids is a set for fast lookup
    valid_ids = set(valid_ids)

    # Regex for filenames like "particle_0395_mask.tif"
    pattern = re.compile(r"particle_(\d{4})_mask\.tif$")

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            particle_id = int(match.group(1))  # "0395" -> 395
            if particle_id not in valid_ids:
                filepath = os.path.join(directory, filename)
                if dry_run:
                    print(f"[DRY RUN] Would delete: {filepath}")
                else:
                    print(f"Deleting: {filepath}")
                    os.remove(filepath)


