# How to run the pipeline

## 1. Files needed in the same folder

All of these must be together, in the folder from which the pipeline
will be run (the one containing `inputs/`):

```
run_pipeline.py
get_data_and_artifacts.py
deconvolution.py
lysosome_tracking.py
reticulum_state_classification.py
lysosome_state_classification.py
export_state_videos.py
progress_tracker.py
frame_intensity_utils.py
cell_data_io.py
reticulum_mask_utils.py
calc_diff_cube.py
tifffun.py                 (dependency of calc_diff_cube.py)
tifffun_log2.py             (dependency of calc_diff_cube.py, and used directly by reticulum_state_classification.py)
treatment_frames.json
pipeline_config.json
PSF.tif                    (only if you're going to run deconvolution)
```

## 2. Expected input structure

```
inputs/
  <cell_name>/
    RAW_<cell_name>.tif           (required)
    C1_RAW_<cell_name>.tif        (generated if missing)
    C2_RAW_<cell_name>.tif        (generated if missing)
    cell_coords.csv               (optional, hand-drawn cell border)
```

`cell_name` follows the pattern `<date>-Cell<number>`, e.g. `010519-Cell2`.

The pipeline creates everything else inside `outputs/<cell_name>/` as
each step runs.

## 3. Running the full pipeline

Edit `pipeline_config.json`, and run `run_pipeline.py`. No need to change the code to change the configuration.

```json
{
  "cell_names": null,
  "force_rerun_all": false,
  "run_deconvolution": true,
  "deconv_c1_iterations": 30,
  "deconv_c2_iterations": 20,
  "deconv_c2_denoise_weight": 0.01,
  "reticulum_normalization_method": "dff",
  "reticulum_outlier_handle": "remove_percentile",
  "lysosome_normalization_method": "dff",
  "run_export_videos": false
}
```

- `cell_names`: `null` = all cells in `inputs/`, or a list of names,
  e.g. `["010519-Cell2", "010519-Cell3"]`.
- `force_rerun_all`: `true` ignores `progress.json` and redoes everything.
- `run_deconvolution`: `false` creates symlinks to the raw data instead
  of deconvolving.
- `reticulum_normalization_method` / `lysosome_normalization_method`:
  `"dff"` or `"log2"`, independent of each other.
- `reticulum_outlier_handle`: `clip`, `remove_iqr`, `remove_percentile`,
  `remove_zscore`, or `none`.
- `run_export_videos`: step 5, optional.

If `pipeline_config.json` doesn't exist, or is missing a key, default
values (the same ones shown above) are used for whatever is missing.

Then:

```
python run_pipeline.py
```

## 4. Progress and resuming

Each cell keeps its own `outputs/<cell_name>/progress.json`, with the
status of each step (complete, error, or relevant notes such as which
normalization method was used). If the pipeline is interrupted,
running it again picks up from the last completed step for each cell.

## 5. Output per cell

```
outputs/<cell_name>/
  progress.json
  cell_metadata.json
  dark_frames.json, dark_frame_check.png
  bright_frames.json, bright_frame_check.png
  deconv/                              (deconvolved, or symlinks to raw)
  individual_masks/                    (per-particle lysosome masks)
  <basename>_tracks.csv, ..._dff.csv, ..._dff_filt.csv
  param3-..._tracks_dff_filt_with_states_ordered.csv  (lysosome states)
  <cell_name>_state_probs_global.npz   (reticulum state probabilities)
  POIs.csv, trained_params.txt
fiji_export/<cell_name>/               (if RUN_EXPORT_VIDEOS = True)
  C1_state.tif
  C2_state.tif
```

## 6. Example

Run only the `010519-Cell2` cell, without deconvolution, normalizing
with log2, generating the state videos at the end.

`pipeline_config.json`:

```json
{
  "cell_names": ["010519-Cell2"],
  "force_rerun_all": false,
  "run_deconvolution": false,
  "deconv_c1_iterations": 30,
  "deconv_c2_iterations": 20,
  "deconv_c2_denoise_weight": 0.01,
  "reticulum_normalization_method": "log2",
  "reticulum_outlier_handle": "remove_percentile",
  "lysosome_normalization_method": "log2",
  "run_export_videos": true
}
```

And then:

```bash
python run_pipeline.py
```

## 7. Environment

The required libraries are in `requirements.txt`:

```
pip install -r requirements.txt
```
