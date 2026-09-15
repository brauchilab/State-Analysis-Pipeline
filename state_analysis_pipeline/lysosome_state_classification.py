#!/usr/bin/env python
# lysosome_state_classification.py
"""
Step 4: lysosome state classification (2-state HMM) on df_f0 (or
log2(F/F0), configurable).

For each cell:
  - loads the filtered lysosome tracks CSV (from lysosome tracking)
  - marks artifact frames (bright-flash treatment artifact) and dark
    frames as excluded from training:
      - artifact frame: manual entry in treatment_frames.json if present
        for this cell, else auto-detected as the midpoint of the first
        range in bright_frames.json, else no artifact exclusion
      - dark frames: all frames in dark_frames.json, no buffer, since a
        camera blackout doesn't bleed into neighboring frames the way an
        optical flash transient does
  - fits a GaussianHMM on non-excluded, clipped training data
  - scores every frame independently (no Viterbi, no temporal locking)
  - reorders states by mean signal (state 0 = lowest)
  - saves CSVs and plots

Run from the folder containing outputs/:
    python lysosome_state_classification.py                 # all cells
    python lysosome_state_classification.py 010519-Cell2     # one cell
"""

import json
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from scipy.stats import norm

from cell_data_io import find_cell_data
from progress_tracker import update_step, is_step_complete

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

OUTPUTS_DIR = Path("./outputs")
TREATMENT_FRAMES_PATH = Path("./treatment_frames.json")

STEP_NAME = "lysosome_state_classification"
FORCE_RERUN = False

N_STATES = 2
OUTPUT_PREFIX = "param3-"
GENERATE_PLOTS = True

# "dff" -> classify on df_f0 directly. "log2" -> classify on
# log2(1 + df_f0), which equals log2(F/F0) since df_f0 = (F-F0)/F0 means
# F/F0 = 1 + df_f0. Reuses the same baseline as df_f0, no separate
# intensity recomputation needed.
NORMALIZATION_METHOD = "dff"

# symmetric clip on training data only, |norm_value| > threshold excluded.
# scoring always uses the unclipped values.
CLIP_THRESHOLD = 0.3

# frames excluded on each side of the treatment (artifact) frame
ARTIFACT_BUFFER = 10

# True: artifact frames get classified (states assigned, is_artifact=True)
# False: left as NaN state (safer for downstream analysis)
CLASSIFY_ARTIFACT_FRAMES = False


# ---------------------------------------------------------------------------
# treatment frame / artifact window
# ---------------------------------------------------------------------------

def load_treatment_frames():
    if not TREATMENT_FRAMES_PATH.exists():
        return {}
    with open(TREATMENT_FRAMES_PATH) as f:
        data = json.load(f)
    data.pop("_comment", None)
    return data


def get_bright_frame_ranges(cell_dir):
    bright_path = Path(cell_dir) / "bright_frames.json"
    if not bright_path.exists():
        return []
    with open(bright_path) as f:
        data = json.load(f)
    return [tuple(r) for r in data.get("bright_frame_ranges", [])]


def get_artifact_frames(cell_name, cell_dir, treatment_frames):
    """Manual entry first. If absent, auto-detect from the midpoint of the
    first bright frame run. Returns (frame_list, source) where source is
    'manual', 'auto_bright', or 'none'."""
    manual = treatment_frames.get(cell_name)
    if manual is not None:
        frames = [int(manual)] if isinstance(manual, (int, np.integer)) else list(manual)
        return frames, "manual"

    bright_ranges = get_bright_frame_ranges(cell_dir)
    if bright_ranges:
        start, end = bright_ranges[0]
        midpoint = (start + end) // 2
        return [midpoint], "auto_bright"

    return [], "none"


def make_artifact_mask(frames_array, artifact_frame_list, buffer):
    mask = np.zeros(len(frames_array), dtype=bool)
    for af in artifact_frame_list:
        mask |= (frames_array >= af - buffer) & (frames_array <= af + buffer)
    return mask


# ---------------------------------------------------------------------------
# dark frame exclusion, no buffer
# ---------------------------------------------------------------------------

def get_dark_frame_info(cell_dir):
    """Returns (flat_index_list, ranges_list, file_found). ranges_list is
    used for plotting real contiguous dark blocks; file_found tells apart
    'no dark_frames.json' from 'file exists, zero dark frames'."""
    dark_path = Path(cell_dir) / "dark_frames.json"
    if not dark_path.exists():
        return [], [], False
    with open(dark_path) as f:
        data = json.load(f)
    indices = sorted(int(i) for i in data.get("dark_frame_indices", []))
    ranges = [tuple(r) for r in data.get("dark_frame_ranges", [])]
    return indices, ranges, True


def make_dark_frame_mask(frames_array, dark_frame_indices):
    if not dark_frame_indices:
        return np.zeros(len(frames_array), dtype=bool)
    return np.isin(frames_array, dark_frame_indices)


# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------

def compute_norm_value(df):
    if NORMALIZATION_METHOD == "log2":
        ratio = np.clip(1 + df["df_f0"].values, 1e-6, None)  # guard against F/F0 <= 0
        return np.log2(ratio)
    return df["df_f0"].values


# ---------------------------------------------------------------------------
# HMM fitting and scoring
# ---------------------------------------------------------------------------

def clip_for_training(df, threshold):
    mask = np.abs(df["norm_value"].values) <= threshold
    n_removed = (~mask).sum()
    print(f"      symmetric clip |value| > {threshold}: {n_removed} row(s) removed "
          f"({100 * n_removed / len(df):.2f}%)")
    return df[mask].copy()


def fit_hmm(train_values, n_states, random_state=42, n_iter=200):
    model = GaussianHMM(n_components=n_states, covariance_type="full",
                         n_iter=n_iter, random_state=random_state, verbose=False)
    model.fit(train_values.reshape(-1, 1), lengths=[len(train_values)])
    means = model.means_.flatten()
    stds = np.sqrt(model.covars_.flatten())
    order = np.argsort(means)  # state 0 = lowest
    return means[order], stds[order]


def gaussian_score_frames(values, emission_means, emission_stds):
    """Classify each frame independently by Gaussian likelihood. No
    transition matrix, no temporal locking."""
    likelihoods = np.column_stack([
        norm.pdf(values, loc=emission_means[s], scale=emission_stds[s])
        for s in range(len(emission_means))
    ])
    row_sums = likelihoods.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    state_probs = likelihoods / row_sums
    states = state_probs.argmax(axis=1)
    return states, state_probs


def reorder_states(df_predicted, n_states):
    """Re-map state labels so state 0 has the lowest mean signal."""
    state_means = {
        s: df_predicted.loc[df_predicted["predicted_state"] == s, "norm_value"].mean()
        for s in range(n_states)
    }
    sorted_states = sorted(state_means.items(), key=lambda x: x[1])
    state_mapping = {old: new for new, (old, _) in enumerate(sorted_states)}

    df_ordered = df_predicted.copy()
    df_ordered["predicted_state"] = df_ordered["predicted_state"].map(state_mapping)
    for new_state in range(n_states):
        old_state = [k for k, v in state_mapping.items() if v == new_state][0]
        df_ordered[f"state{new_state}_prob"] = df_predicted[f"state{old_state}_prob"]

    return df_ordered, state_mapping


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

_STATE_COLORS = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]


def plot_state_predictions(df_ordered, n_states, output_path, value_label,
                            y_lim=None, dark_frame_ranges=None):
    particles = sorted(df_ordered["particle"].unique())
    n_cols = 3
    n_rows = int(np.ceil(len(particles) / n_cols))
    colors = _STATE_COLORS[:n_states]
    state_labels = ["Low", "Mid", "High", "Ultra"][:n_states]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = np.array([axes]).flatten() if len(particles) == 1 else axes.flatten()

    for idx, pid in enumerate(particles):
        ax = axes[idx]
        pdt = df_ordered[df_ordered["particle"] == pid]

        if "is_artifact" in pdt.columns and pdt["is_artifact"].any():
            art_frames = pdt.loc[pdt["is_artifact"], "frame"].values
            if len(art_frames):
                ax.axvspan(art_frames.min(), art_frames.max(),
                           color="salmon", alpha=0.25, label="artifact window")

        if dark_frame_ranges:
            for i, (start, end) in enumerate(dark_frame_ranges):
                ax.axvspan(start, end, color="gray", alpha=0.2,
                           label="dark frame(s)" if i == 0 else None)

        ax.plot(pdt["frame"], pdt["norm_value"], "o-",
                 alpha=0.4, color="gray", linewidth=1, markersize=3, label=value_label)

        for s in range(n_states):
            mask = pdt["predicted_state"] == s
            ax.scatter(pdt[mask]["frame"], pdt[mask]["norm_value"],
                       color=colors[s], s=30, alpha=0.75,
                       label=f"State {s} ({state_labels[s]})",
                       edgecolor="black", linewidth=0.5)

        if y_lim is not None:
            ax.set_ylim(y_lim)

        ax.set_xlabel("Frame", fontsize=10)
        ax.set_ylabel(value_label, fontsize=10)
        ax.set_title(f"Particle {pid}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    for idx in range(len(particles), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def compute_scaled_ylim(df_ordered, percentile_cutoff=95):
    particles = sorted(df_ordered["particle"].unique())
    lo = (100 - percentile_cutoff) / 2
    hi = 100 - lo
    all_ranges = [
        (np.percentile(df_ordered.loc[df_ordered["particle"] == pid, "norm_value"], lo),
         np.percentile(df_ordered.loc[df_ordered["particle"] == pid, "norm_value"], hi))
        for pid in particles
    ]
    g_min = min(r[0] for r in all_ranges)
    g_max = max(r[1] for r in all_ranges)
    pad = (g_max - g_min) * 0.05
    return g_min - pad, g_max + pad


# ---------------------------------------------------------------------------
# per-cell pipeline
# ---------------------------------------------------------------------------

def process_cell(cell_dir, cell_name, treatment_frames):
    print(f"[cell] {cell_name}")

    matches = sorted(Path(cell_dir).glob("C2*tracks_dff_filt.csv"))
    if not matches:
        raise FileNotFoundError(f"no C2*tracks_dff_filt.csv found in {cell_dir}")

    csv_path = str(matches[0])
    c2_basename = Path(csv_path).name.replace("_tracks_dff_filt.csv", "")

    output_raw = f"{cell_dir}/{OUTPUT_PREFIX}{c2_basename}_tracks_dff_filt_with_states.csv"
    output_ordered = f"{cell_dir}/{OUTPUT_PREFIX}{c2_basename}_tracks_dff_filt_with_states_ordered.csv"
    plot_path = f"{cell_dir}/{OUTPUT_PREFIX}HMM_state_predictions_ordered.png"
    plot_scaled = f"{cell_dir}/{OUTPUT_PREFIX}HMM_state_predictions_ordered_scaled.png"

    df = pd.read_csv(csv_path)
    df["norm_value"] = compute_norm_value(df)
    value_label = "log2(F/F0)" if NORMALIZATION_METHOD == "log2" else "df_f0"

    frames_array = df["frame"].values.astype(int)

    art_frame_list, art_source = get_artifact_frames(cell_name, cell_dir, treatment_frames)
    if art_frame_list:
        art_mask = make_artifact_mask(frames_array, art_frame_list, ARTIFACT_BUFFER)
        print(f"  treatment frame(s): {art_frame_list} (source: {art_source}, "
              f"buffer +-{ARTIFACT_BUFFER})")
    else:
        art_mask = np.zeros(len(df), dtype=bool)
        print("  no treatment frame for this cell, no artifact exclusion")

    dark_frame_indices, dark_frame_ranges, dark_file_found = get_dark_frame_info(cell_dir)
    dark_mask = make_dark_frame_mask(frames_array, dark_frame_indices)
    if dark_frame_indices:
        print(f"  dark frames: {len(dark_frame_indices)}, ranges={dark_frame_ranges}, no buffer")
    elif dark_file_found:
        print("  dark_frames.json found but empty")

    df["is_artifact"] = art_mask
    df["is_dark_frame"] = dark_mask
    df["is_excluded"] = art_mask | dark_mask
    print(f"  total excluded: {df['is_excluded'].sum()}/{len(df)} "
          f"({100 * df['is_excluded'].sum() / len(df):.1f}%)")

    df_train_pool = df[~df["is_excluded"]].copy()
    df_train_pool = clip_for_training(df_train_pool, CLIP_THRESHOLD)
    if len(df_train_pool) < 10:
        raise ValueError(f"only {len(df_train_pool)} training rows after exclusion + clip")

    emission_means, emission_stds = fit_hmm(df_train_pool["norm_value"].values, N_STATES)
    std_ratio = emission_stds.max() / (emission_stds.min() + 1e-9)
    print(f"  emission_means={emission_means}, emission_stds={emission_stds}, "
          f"std_ratio={std_ratio:.2f}")

    particles = sorted(df["particle"].unique())
    all_predictions = []
    for pid in particles:
        pdata = df[df["particle"] == pid].copy()
        is_excluded = pdata["is_excluded"].values

        if CLASSIFY_ARTIFACT_FRAMES:
            score_mask = ~pdata["norm_value"].isna().values
        else:
            score_mask = ~pdata["norm_value"].isna().values & ~is_excluded

        vals = pdata["norm_value"].values
        pdata["predicted_state"] = np.nan
        for s in range(N_STATES):
            pdata[f"state{s}_prob"] = np.nan

        if score_mask.sum() >= 2:
            states, state_probs = gaussian_score_frames(vals[score_mask], emission_means, emission_stds)
            pdata.loc[pdata.index[score_mask], "predicted_state"] = states.astype(float)
            for s in range(N_STATES):
                pdata.loc[pdata.index[score_mask], f"state{s}_prob"] = state_probs[:, s]

        all_predictions.append(pdata)

    df_predicted = pd.concat(all_predictions, ignore_index=True)
    df_predicted.to_csv(output_raw, index=False)

    df_ordered, state_mapping = reorder_states(df_predicted, N_STATES)
    df_ordered.to_csv(output_ordered, index=False)
    print(f"  saved {output_ordered}")

    if GENERATE_PLOTS:
        plot_state_predictions(df_ordered, N_STATES, plot_path, value_label,
                                dark_frame_ranges=dark_frame_ranges)
        y_lim = compute_scaled_ylim(df_ordered, percentile_cutoff=95)
        plot_state_predictions(df_ordered, N_STATES, plot_scaled, value_label,
                                y_lim=y_lim, dark_frame_ranges=dark_frame_ranges)
        print(f"  saved plots")

    return {
        "normalization_method": NORMALIZATION_METHOD,
        "artifact_frame_source": art_source,
        "artifact_frames": art_frame_list,
        "n_dark_frames_excluded": len(dark_frame_indices),
        "n_particles": len(particles),
        "std_ratio": float(std_ratio),
    }


def run(cell_names=None):
    """Runs this step for the given cell names, or every cell with an
    outputs/ folder if cell_names is None."""
    if not OUTPUTS_DIR.is_dir():
        raise SystemExit(f"outputs dir not found: {OUTPUTS_DIR.resolve()}")

    treatment_frames = load_treatment_frames()

    cells2process = find_cell_data()
    valid_cells = {d.name for d in OUTPUTS_DIR.iterdir() if d.is_dir()}
    cells2process = [c for c in cells2process if c["cell_name"] in valid_cells]

    if cell_names:
        requested = set(cell_names)
        run_queue = [c for c in cells2process if c["cell_name"] in requested]
        missing = requested - {c["cell_name"] for c in cells2process}
        if missing:
            print(f"WARNING: cell name(s) not found: {sorted(missing)}")
    else:
        run_queue = cells2process

    print(f"Processing {len(run_queue)} cell(s), normalization={NORMALIZATION_METHOD}.")

    results_log = {"success": [], "failed": [], "skipped": []}
    for cell in run_queue:
        cell_name = cell["cell_name"]
        cell_dir = OUTPUTS_DIR / cell_name

        if not FORCE_RERUN and is_step_complete(OUTPUTS_DIR, cell_name, STEP_NAME):
            print(f"[skip] {cell_name}: state classification already done")
            results_log["skipped"].append(cell_name)
            continue

        if not list(cell_dir.glob("C2*tracks_dff_filt.csv")):
            print(f"[skip] {cell_name}: no C2*tracks_dff_filt.csv")
            results_log["skipped"].append(cell_name)
            continue

        try:
            notes = process_cell(str(cell_dir), cell_name, treatment_frames)
            update_step(OUTPUTS_DIR, cell_name, STEP_NAME, "complete", notes)
            results_log["success"].append(cell_name)
        except Exception as e:
            traceback.print_exc()
            update_step(OUTPUTS_DIR, cell_name, STEP_NAME, "error", {"message": str(e)})
            results_log["failed"].append(cell_name)

    print(f"\nDone. success={len(results_log['success'])}, "
          f"failed={len(results_log['failed'])}, skipped={len(results_log['skipped'])}")
    if results_log["failed"]:
        print(f"Failed: {results_log['failed']}")


def main():
    run(sys.argv[1:] or None)


if __name__ == "__main__":
    main()
