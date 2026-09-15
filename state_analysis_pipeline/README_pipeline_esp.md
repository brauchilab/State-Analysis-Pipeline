# Como correr el pipeline

## 1. Archivos necesarios en la misma carpeta

Todos estos deben estar juntos, en la carpeta desde donde se va a correr el
pipeline (la que contiene `inputs/`):

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
PSF.tif                    (solo si vas a correr deconvolucion)
```

## 2. Estructura de entrada esperada

```
inputs/
  <cell_name>/
    RAW_<cell_name>.tif           (obligatorio)
    C1_RAW_<cell_name>.tif        (se genera solo si falta)
    C2_RAW_<cell_name>.tif        (se genera solo si falta)
    cell_coords.csv               (opcional, borde de celula dibujado a mano)
```

`cell_name` sigue el patron `<fecha>-Cell<numero>`, ej. `010519-Cell2`.

El pipeline crea todo lo demas dentro de `outputs/<cell_name>/` a medida
que corre cada paso.

## 3. Correr el pipeline completo

Editar `pipeline_config.json`, y correr `run_pipeline.py` (como script o
como celda). No hace falta cambiar el codigo para cambiar la configuracion.

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

- `cell_names`: `null` = todas las celulas en `inputs/`, o una lista de
  nombres, ej. `["010519-Cell2", "010519-Cell3"]`.
- `force_rerun_all`: `true` ignora `progress.json` y rehace todo.
- `run_deconvolution`: `false` crea symlinks al raw en vez de
  deconvolucionar.
- `reticulum_normalization_method` / `lysosome_normalization_method`:
  `"dff"` o `"log2"`, independientes entre si.
- `reticulum_outlier_handle`: `clip`, `remove_iqr`, `remove_percentile`,
  `remove_zscore`, o `none`.
- `run_export_videos`: paso 5, opcional.

Si `pipeline_config.json` no existe, o le falta alguna clave, se usan
valores por defecto (los mismos que aparecen arriba) para lo que falte.

Luego:

```
python run_pipeline.py
```

## 4. Progreso y resume

Cada celula lleva su propio `outputs/<cell_name>/progress.json`, con el
estado de cada paso (completo, error, o notas relevantes como que metodo
de normalizacion se uso). Si el pipeline se interrumpe, correrlo de nuevo
retoma desde el ultimo paso completo para cada celula.

## 5. Salida por celula

```
outputs/<cell_name>/
  progress.json
  cell_metadata.json
  dark_frames.json, dark_frame_check.png
  bright_frames.json, bright_frame_check.png
  deconv/                              (deconvolucionado, o symlinks al raw)
  individual_masks/                    (mascaras por particula de lisosoma)
  <basename>_tracks.csv, ..._dff.csv, ..._dff_filt.csv
  param3-..._tracks_dff_filt_with_states_ordered.csv  (estados de lisosoma)
  <cell_name>_state_probs_global.npz   (probabilidades de estado del reticulo)
  POIs.csv, trained_params.txt
fiji_export/<cell_name>/               (si RUN_EXPORT_VIDEOS = True)
  C1_state.tif
  C2_state.tif
```

## 6. Ejemplo

Correr solo la celula `010519-Cell2`, sin deconvolucion, normalizando
con log2, generando los videos de estado al final.

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

Y luego:

```bash
python run_pipeline.py
```

## 7. Entorno

Las librerias necesarias estan en `requirements.txt`:

```
pip install -r requirements.txt
```


