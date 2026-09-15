#!/usr/bin/env python3
"""
HMM Parameter Estimation for Video Pixel Classification

This module trains an HMM on Points of Interest (POIs) data to estimate emission parameters
(means and standard deviations) for three states: Low (0), Medium (1), and High (2).

Author: Adapted for POI-based HMM parameter estimation
Date: September 2025
"""

import numpy as np
import pandas as pd
import logging
import warnings
from datetime import datetime
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Suppress common warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Defaults
N_STATES = 3
MAX_ITERATIONS = 100
CONVERGENCE_TOLERANCE = 1e-6


class HMMParameterEstimator:
    def __init__(self, n_states=N_STATES, max_iter=MAX_ITERATIONS, tol=CONVERGENCE_TOLERANCE):
        self.n_states = n_states
        self.max_iter = max_iter
        self.tol = tol
        self.model = None
        self.emission_means = None
        self.emission_stds = None
        self.transition_matrix = None

    def load_poi_data(self, file_path):
        """Load POI data from CSV file."""
        logger.info(f"Loading POI data from {file_path}")
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded data with shape: {df.shape}")

            if "frame" not in df.columns:
                logger.error("No 'frame' column found in data")
                return None

            coord_columns = [col for col in df.columns if col != "frame"]
            logger.info(
                f"Found {len(coord_columns)} POI coordinates: "
                f"{coord_columns[:5]}..." if len(coord_columns) > 5 else coord_columns
            )

            timeseries_data = df[coord_columns].values
            frames = df["frame"].values

            logger.info(f"Timeseries data shape: {timeseries_data.shape}")
            logger.info(f"Frame range: {frames.min()} to {frames.max()}")

            return timeseries_data, coord_columns, frames

        except FileNotFoundError:
            logger.error(f"File {file_path} not found")
            return None
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None

    def create_sequences(self, timeseries_data):
        """Convert timeseries data to sequences for HMM training."""
        n_frames, n_pois = timeseries_data.shape
        sequences = []
        for poi_idx in range(n_pois):
            sequence = timeseries_data[:, poi_idx]
            sequence = sequence[~np.isnan(sequence)]
            if len(sequence) > 0:
                sequences.append(sequence)

        logger.info(f"Created {len(sequences)} sequences for HMM training")
        total_points = sum(len(seq) for seq in sequences)
        logger.info(f"Total data points: {total_points}")
        return sequences

    def estimate_initial_parameters(self, sequences):
        """Estimate initial parameters using Gaussian Mixture Model with better separation."""
        logger.info("Estimating initial parameters using GMM...")
        all_values = np.concatenate(sequences)
        logger.info(f"Value range: {all_values.min():.3f} to {all_values.max():.3f}")

        best_gmm = None
        best_score = -np.inf
        for random_seed in range(5):
            try:
                gmm = GaussianMixture(
                    n_components=self.n_states,
                    random_state=random_seed,
                    max_iter=200,
                    n_init=10,
                )
                gmm.fit(all_values.reshape(-1, 1))
                score = gmm.score(all_values.reshape(-1, 1))
                if score > best_score:
                    best_score = score
                    best_gmm = gmm
            except Exception as e:
                logger.warning(f"GMM initialization {random_seed} failed: {e}")
                continue

        if best_gmm is None:
            logger.error("All GMM initializations failed, using manual initialization")
            percentiles = [10, 50, 90]
            initial_means = np.array([np.percentile(all_values, p) for p in percentiles])
            initial_stds = np.array([np.std(all_values)] * 3) * 0.3
        else:
            initial_means = best_gmm.means_.flatten()
            initial_stds = np.sqrt(best_gmm.covariances_.flatten())

        sorted_indices = np.argsort(initial_means)
        initial_means = initial_means[sorted_indices]
        initial_stds = initial_stds[sorted_indices]

        min_separation = (initial_means[-1] - initial_means[0]) * 0.2
        for i in range(1, len(initial_means)):
            if initial_means[i] - initial_means[i - 1] < min_separation:
                initial_means[i] = initial_means[i - 1] + min_separation

        data_range = all_values.max() - all_values.min()
        min_std = data_range * 0.05
        max_std = data_range * 0.4
        initial_stds = np.clip(initial_stds, min_std, max_std)

        logger.info(f"Initial emission means: {initial_means}")
        logger.info(f"Initial emission stds: {initial_stds}")
        return initial_means, initial_stds

    def prepare_training_data(self, sequences):
        X = np.concatenate(sequences).reshape(-1, 1)
        lengths = [len(seq) for seq in sequences]
        return X, lengths

    def train_hmm(self, sequences):
        logger.info(f"Training HMM on {len(sequences)} sequences...")
        initial_means, initial_stds = self.estimate_initial_parameters(sequences)
        X, lengths = self.prepare_training_data(sequences)

        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",
            n_iter=self.max_iter,
            tol=self.tol,
            random_state=42,
            init_params="",
            params="stmc",
            verbose=True,
        )

        self.model.startprob_ = np.ones(self.n_states) / self.n_states
        stay_prob = 0.7
        switch_prob = (1.0 - stay_prob) / (self.n_states - 1)
        transition_matrix = np.full((self.n_states, self.n_states), switch_prob)
        np.fill_diagonal(transition_matrix, stay_prob)
        self.model.transmat_ = transition_matrix

        self.model.means_ = initial_means.reshape(-1, 1)
        self.model.covars_ = (initial_stds**2).reshape(-1, 1)

        try:
            self.model.fit(X, lengths)
            final_logprob = self.model.score(X, lengths)
            logger.info(f"Final log-likelihood: {final_logprob:.4f}")
        except Exception as e:
            logger.error(f"Error during HMM training: {e}")
            return False

        self.extract_parameters()
        return True

    def extract_parameters(self):
        means = self.model.means_.flatten()
        stds = np.sqrt(self.model.covars_.flatten())
        sorted_indices = np.argsort(means)
        self.emission_means = means[sorted_indices]
        self.emission_stds = stds[sorted_indices]
        self.transition_matrix = self.model.transmat_[np.ix_(sorted_indices, sorted_indices)]

    def print_results(self):
        print("\n" + "=" * 50)
        print("FINAL TRAINED HMM PARAMETERS:")
        print("=" * 50)
        print(f"Emission means: {self.emission_means}")
        print(f"Emission stds: {self.emission_stds}")
        print("\nTransition matrix:")
        for i, row in enumerate(self.transition_matrix):
            print(f"  State {i}: {row}")
        print("=" * 50)

    def save_parameters(self, output_file="hmm_parameters.txt"):
        with open(output_file, "w") as f:
            f.write(f"# HMM Parameters - Generated {datetime.now()}\n")
            f.write("import numpy as np\n\n")
            f.write("emission_means = np.array([")
            f.write(", ".join([f"{mean:.8f}" for mean in self.emission_means]))
            f.write("])\n\n")
            f.write("emission_stds = np.array([")
            f.write(", ".join([f"{std:.8f}" for std in self.emission_stds]))
            f.write("])\n\n")
            f.write("transition_matrix = np.array([\n")
            for row in self.transition_matrix:
                f.write("    [" + ", ".join([f"{val:.8f}" for val in row]) + "],\n")
            f.write("])\n")
        logger.info(f"Parameters saved to {output_file}")

    def get_parameters_dict(self, version="V6"):
        return {
            f"trained_params_{version}": {
                "emission_means": self.emission_means,
                "emission_stds": self.emission_stds,
                "transition_matrix": self.transition_matrix,
            }
        }


def run_estimation(
    csv_file,
    output_file="hmm_parameters.txt",
    version="V6",
    n_states=N_STATES,
    max_iter=MAX_ITERATIONS,
    tol=CONVERGENCE_TOLERANCE,
):
    estimator = HMMParameterEstimator(n_states=n_states, max_iter=max_iter, tol=tol)
    result = estimator.load_poi_data(csv_file)
    if result is None:
        raise FileNotFoundError(f"Could not load {csv_file}")
    timeseries_data, coord_columns, frames = result
    sequences = estimator.create_sequences(timeseries_data)
    if len(sequences) == 0:
        raise ValueError("No valid sequences found in the data")
    success = estimator.train_hmm(sequences)
    if not success:
        raise RuntimeError("HMM training failed")
    estimator.print_results()
    estimator.save_parameters(output_file=output_file)
    return estimator.get_parameters_dict(version=version)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HMM parameter estimation on POI CSV data.")
    parser.add_argument("csv_file", nargs="?", default="clipped_POIs.csv", help="Path to POI CSV file")
    parser.add_argument("--output", default="hmm_parameters.txt", help="Output file for parameters")
    parser.add_argument("--version", default="V6", help="Version tag for output dictionary key")
    args = parser.parse_args()

    params = run_estimation(args.csv_file, output_file=args.output, version=args.version)
    print("\nTraining completed successfully!")
    print(f"Parameters have been saved to '{args.output}'")
    print("Returned dictionary:")
    print(params)

