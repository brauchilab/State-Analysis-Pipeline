# cell_data_io.py
"""
find_cell_data and load_tracks_data, split out of pipeline_funcs.py so
importing them doesn't pull in that file's other imports (dogmasks,
masktracking, dist_matrix, dist_an, video_funcs) -- none of which this
pipeline uses, but which would break the import entirely if any of them
were missing or broken.
"""

import os
from typing import List, Dict, Union

import numpy as np
import pandas as pd


def find_cell_data(base_input_folder="./inputs") -> List[Dict[str, Union[str, os.PathLike]]]:
    """Scans base_input_folder for cell subfolders containing a C1 and C2
    tif (matching a 'C1-'/'C1_' or 'C2-'/'C2_' filename prefix). Returns
    one dict per cell with cell_name, cell_folder_path, c1_file_path,
    c2_file_path.

    Also used, deliberately, with a cell's deconv/ folder as
    base_input_folder (by passing cell_folder_path instead of the inputs
    root) -- deconv/C1_Deconv_<cell>.tif and C2_Deconv_<cell>.tif match
    the same 'C1_'/'C2_' prefix check, so the same function finds the
    deconvolved pair. The returned cell_name in that case is the
    subfolder name ("deconv"), which callers relying on this trick
    ignore -- only c1_file_path/c2_file_path are used.
    """
    cell_data = []

    for item_name in os.listdir(base_input_folder):
        cell_folder_path = os.path.join(base_input_folder, item_name)
        if not os.path.isdir(cell_folder_path):
            continue

        cell_name = item_name
        c1_file_path = None
        c2_file_path = None

        for file_name in os.listdir(cell_folder_path):
            if not file_name.endswith(".tif"):
                continue
            if file_name.startswith("C1-") or file_name.startswith("C1_"):
                c1_file_path = os.path.join(cell_folder_path, file_name)
            elif file_name.startswith("C2-") or file_name.startswith("C2_"):
                c2_file_path = os.path.join(cell_folder_path, file_name)

        if c1_file_path and c2_file_path:
            cell_data.append({
                "cell_name": cell_name,
                "cell_folder_path": cell_folder_path,
                "c1_file_path": c1_file_path,
                "c2_file_path": c2_file_path,
            })
        else:
            print(f"Warning: cell folder '{cell_name}' is missing a C1 or C2 tif")

    return cell_data


def load_tracks_data(csv_file, round_coordinates=False):
    """Loads a tracks CSV and returns just [particle, frame, y, x, df_f0],
    sorted by particle and frame."""
    df = pd.read_csv(csv_file, sep=",")
    df_filtered = df[["particle", "frame", "y", "x", "df_f0"]].copy()

    if round_coordinates:
        df_filtered["y"] = np.round(df_filtered["y"]).astype(int)
        df_filtered["x"] = np.round(df_filtered["x"]).astype(int)

    return df_filtered.sort_values(["particle", "frame"]).reset_index(drop=True)
