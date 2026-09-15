import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import tifffile
from matplotlib.patches import Circle

def plot_particle_trajectories_line(df_out, output_folder, max_x=255, max_y=255):
    """
    Creates and saves movement trajectory plots (x vs y) for each particle,
    using only a continuous line (no markers).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for particle_id in df_out['particle'].unique():
        df_particle = df_out[df_out['particle'] == particle_id].sort_values('frame')
        x = df_particle['x']
        y = df_particle['y']

        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Plot trajectory as line only
        ax.plot(x, y, linewidth=1.5, color='blue', alpha=0.8)
        
        # Mark start and end
        ax.scatter(x.iloc[0], y.iloc[0], color='green', s=50, label="Start")
        ax.scatter(x.iloc[-1], y.iloc[-1], color='red', s=50, label="End")
        
        ax.set_title(f"Particle {particle_id} - Trajectory (Line Only)")
        ax.set_xlabel("X position (pixels)")
        ax.set_ylabel("Y position (pixels)")
        ax.set_xlim(0, max_x)
        ax.set_ylim(0, max_y)
        ax.invert_yaxis()
        ax.legend()
        ax.grid(True)
        
        filename = os.path.join(output_folder, f"particle_{particle_id}_trajectory_line.png")
        plt.savefig(filename, dpi=150)
        plt.close(fig)
        print(f"Saved line-only trajectory plot for particle {particle_id} to {filename}")


def plot_particle_timeseries_accumvar(df_out, output_folder):
    """
    Creates and saves timeseries plots for each particle.
    Plots state averages and counts, and overlays accumulated variance of df_f0.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    colors = ['blue', 'orange', 'green']

    def plot_with_var(ax, frames, states_df, acc_var, title, ylabel):
        for i, col in enumerate(states_df.columns):
            ax.plot(frames, states_df[col], label=col, color=colors[i])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)

        axb = ax.twinx()
        axb.plot(frames, acc_var, color='red', linewidth=1, alpha=0.7, label='Accum. Var(df_f0)')
        axb.set_ylabel("Accumulated Variance of df_f0")
        return ax, axb

    for particle_id in df_out['particle'].unique():
        df_particle = df_out[df_out['particle'] == particle_id].sort_values('frame')
        frames = df_particle['frame']
        df_f0 = df_particle['df_f0']

        # Compute accumulated variance over time
        acc_var = [df_f0.iloc[:i+1].var() for i in range(len(df_f0))]

        states_avg = df_particle[['avg_state0', 'avg_state1', 'avg_state2']]
        states_count = df_particle[['count_state0', 'count_state1', 'count_state2']]

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Average probs
        plot_with_var(axes[0], frames, states_avg, acc_var,
                      f"Particle {particle_id} - Avg State Probabilities", "Average Probability")
        # Counts
        plot_with_var(axes[1], frames, states_count, acc_var,
                      f"Particle {particle_id} - Count of Pixels per State", "Count")

        for ax in axes:
            ax.legend(loc='upper left')
        plt.xlabel("Frame")
        plt.tight_layout()

        filename = os.path.join(output_folder, f"particle_{particle_id}_accumvar.png")
        plt.savefig(filename, dpi=150)
        plt.close(fig)
        print(f"Saved accumulated-variance timeseries plot for particle {particle_id} to {filename}")


def save_particle_earliest_frame_images(df_out, tif_path, output_folder, circle_radius=5):
    """
    For each particle, loads its earliest frame from a TIFF stack,
    draws a red circle around its (x,y) position, and saves the image.
    
    Parameters:
        df_out (pd.DataFrame): DataFrame with 'particle','frame','x','y'.
        tif_path (str): Path to TIFF stack.
        output_folder (str): Where to save output images.
        circle_radius (int): Radius of red circle in pixels.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frames = tifffile.imread(tif_path)  # shape: (n_frames, height, width)

    for particle_id in df_out['particle'].unique():
        df_particle = df_out[df_out['particle'] == particle_id]
        earliest = df_particle.loc[df_particle['frame'].idxmin()]

        frame_idx = int(earliest['frame'])
        y, x = earliest['y'], earliest['x']

        img = frames[frame_idx]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img, cmap='gray')
        circ = Circle((x, y), circle_radius, color='red', fill=False, linewidth=2)
        ax.add_patch(circ)

        ax.set_title(f"Particle {particle_id} - Frame {frame_idx}")
        ax.axis('off')

        filename = os.path.join(output_folder, f"particle_{particle_id}_earliest.png")
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved earliest-frame image for particle {particle_id} to {filename}")


def plot_particle_timeseries_accumvar_no_outliers(df_out, output_folder, method="IQR", z_thresh=3):
    """
    Creates timeseries plots for each particle like plot_particle_timeseries_accumvar,
    but excludes outliers in df_f0 before computing accumulated variance.

    Parameters:
        df_out (pd.DataFrame): Input data with columns ['particle','frame','df_f0',...].
        output_folder (str): Folder where plots will be saved.
        method (str): Outlier detection method: "IQR" (default) or "Z".
        z_thresh (float): Z-score threshold if method="Z".
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    colors = ['blue', 'orange', 'green']

    def remove_outliers(series, method="IQR", z_thresh=3):
        """Return mask of non-outlier values."""
        if method == "IQR":
            Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            return (series >= lower) & (series <= upper)
        elif method == "Z":
            z = (series - series.mean()) / series.std()
            return z.abs() <= z_thresh
        else:
            raise ValueError("method must be 'IQR' or 'Z'")

    def plot_with_var(ax, frames, states_df, acc_var, title, ylabel):
        for i, col in enumerate(states_df.columns):
            ax.plot(frames, states_df[col], label=col, color=colors[i])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)

        axb = ax.twinx()
        axb.plot(frames, acc_var, color='red', linewidth=1, alpha=0.7, label='Accum. Var(df_f0, no outliers)')
        axb.set_ylabel("Accumulated Variance of df_f0 (no outliers)")
        return ax, axb

    for particle_id in df_out['particle'].unique():
        df_particle = df_out[df_out['particle'] == particle_id].sort_values('frame')
        frames = df_particle['frame']
        df_f0 = df_particle['df_f0']

        # Exclude outliers
        mask = remove_outliers(df_f0, method=method, z_thresh=z_thresh)
        df_f0_clean = df_f0[mask]

        # Compute accumulated variance using only non-outlier values up to each frame
        acc_var = []
        for i in range(len(df_f0)):
            subset = df_f0.iloc[:i+1]
            subset_clean = subset[remove_outliers(subset, method=method, z_thresh=z_thresh)]
            acc_var.append(subset_clean.var() if len(subset_clean) > 1 else 0)

        states_avg = df_particle[['avg_state0', 'avg_state1', 'avg_state2']]
        states_count = df_particle[['count_state0', 'count_state1', 'count_state2']]

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Average probabilities
        plot_with_var(axes[0], frames, states_avg, acc_var,
                      f"Particle {particle_id} - Avg State Probabilities", "Average Probability")

        # Count
        plot_with_var(axes[1], frames, states_count, acc_var,
                      f"Particle {particle_id} - Count of Pixels per State", "Count")

        for ax in axes:
            ax.legend(loc='upper left')
        plt.xlabel("Frame")
        plt.tight_layout()

        filename = os.path.join(output_folder, f"particle_{particle_id}_accumvar_no_outliers.png")
        plt.savefig(filename, dpi=150)
        plt.close(fig)
        print(f"Saved no-outlier accumulated variance plot for particle {particle_id} to {filename}")




##########################
def plot_particle_timeseries_save(df_out, output_folder, alpha_df_f0=0.3):
    """
    Creates and saves timeseries plots for each particle in df_out.
    Only plots average and count states (sum plot is commented out).
    
    Parameters:
        df_out (pd.DataFrame): Dataframe containing particle data.
        output_folder (str): Path to folder where plots will be saved.
        alpha_df_f0 (float): Transparency for df_f0 line (default 0.3).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    colors = ['blue', 'orange', 'green']
    
    # Helper function to plot states + df_f0
    def plot_with_df(ax, frames, states_df, df_f0, title, ylabel):
        for i, col in enumerate(states_df.columns):
            ax.plot(frames, states_df[col], label=col, color=colors[i])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)
        
        axb = ax.twinx()
        axb.plot(frames, df_f0, color='red', linewidth=1, alpha=alpha_df_f0, label='df_f0')
        axb.set_ylabel("df_f0")
        axb.set_ylim(-0.4, 0.4)
        return ax, axb
    
    # Loop over all particles
    for particle_id in df_out['particle'].unique():
        df_particle = df_out[df_out['particle'] == particle_id].sort_values('frame')
        frames = df_particle['frame']
        df_f0 = df_particle['df_f0']
        
        states_avg = df_particle[['avg_state0', 'avg_state1', 'avg_state2']]
        # states_sum = df_particle[['sum_state0', 'sum_state1', 'sum_state2']]  # optional
        states_count = df_particle[['count_state0', 'count_state1', 'count_state2']]
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Average probabilities
        plot_with_df(axes[0], frames, states_avg, df_f0,
                     f"Particle {particle_id} - Average State Probabilities", "Average Probability")
        
        # Sum probabilities (optional)
        # plot_with_df(axes[1], frames, states_sum, df_f0,
        #              f"Particle {particle_id} - Sum of State Probabilities", "Sum Probability")
        
        # Count
        plot_with_df(axes[1], frames, states_count, df_f0,
                     f"Particle {particle_id} - Count of Pixels per State", "Count")
        
        for ax in axes:
            ax.legend(loc='upper left')
        plt.xlabel("Frame")
        plt.tight_layout()
        
        # Save figure
        filename = os.path.join(output_folder, f"particle_{particle_id}.png")
        plt.savefig(filename, dpi=150)
        plt.close(fig)
        print(f"Saved plot for particle {particle_id} to {filename}")
