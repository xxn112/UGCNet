import os
import re
import torch
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from config_paper import *


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def calculate_dominant_layer(labels):
    return np.bincount(labels).argmax() if len(labels) > 0 else -1


def clean_curve(values, global_min_val, global_max_val):
    """Missing-value handling + linear interpolation + min-max normalization."""
    vals = values.astype(np.float64).copy()
    for mv in MISSING_VALUES:
        vals[np.isclose(vals, mv, atol=0.5)] = np.nan
    mask = (vals < global_min_val) | (vals > global_max_val)
    vals[mask] = np.nan
    n_nan_original = np.isnan(vals).sum()
    nans = np.isnan(vals)
    if nans.sum() == len(vals):
        return np.full_like(vals, 0.5), n_nan_original
    not_nans = np.where(~nans)[0]
    if len(not_nans) == 0:
        return np.full_like(vals, 0.5), n_nan_original
    vals_interp = vals.copy()
    for i in range(len(vals)):
        if np.isnan(vals_interp[i]):
            left = not_nans[not_nans < i]
            right = not_nans[not_nans > i]
            if len(left) > 0 and len(right) > 0:
                l_idx = left[-1]
                r_idx = right[0]
                ratio = (i - l_idx) / (r_idx - l_idx)
                vals_interp[i] = vals_interp[l_idx] + ratio * (vals_interp[r_idx] - vals_interp[l_idx])
            elif len(left) > 0:
                vals_interp[i] = vals_interp[left[-1]]
            elif len(right) > 0:
                vals_interp[i] = vals_interp[right[0]]
    vals_interp = np.clip(vals_interp, global_min_val, global_max_val)
    vals_normalized = (vals_interp - global_min_val) / (max(global_max_val - global_min_val, 1e-8))
    vals_normalized = np.clip(vals_normalized, 0.0, 1.0)
    return vals_normalized, n_nan_original


def load_and_preprocess_data():
    """Load single-resolution 4-channel logs (GR, AC, DEN, LLD) and build the well graph."""
    print("\n" + "=" * 60)
    print("UGCNet (paper-consistent) - 4ch single-resolution data loading")
    print("=" * 60)

    curve_names = ALL_CURVE_COLS
    global_min = {}
    global_max = {}
    for log_name in curve_names:
        phys_min, phys_max = PHYSICAL_RANGES[log_name]
        global_min[log_name] = phys_min
        global_max[log_name] = phys_max

    print("Normalization: physical ranges")
    print("Pipeline: missing-value interp -> Med(3) -> SG(7) -> 4ch (GR,AC,DEN,LLD)")

    try:
        coord_df = pd.read_excel(COORD_FILE)
        coord_df['井名_clean'] = coord_df['井名'].astype(str).str.replace(r'\s+', '', regex=True)
        coord_dict = coord_df.set_index('井名_clean')[['X', 'Y', '补心海拔']].to_dict('index')
    except Exception as e:
        raise FileNotFoundError(f"Coordinate file read failed: {e}")

    well_data_list = []
    coords_list = []
    altitude_list = []

    data_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.xlsx') and not f.startswith('~$')]
    data_files.sort(key=natural_sort_key)

    total_raw_points = 0
    total_cleaned_nan = 0
    n_curves = len(curve_names)

    print(f"\nFound {len(data_files)} single-well files...")

    for filename in data_files:
        well_name_raw = os.path.splitext(filename)[0]
        well_name_clean = well_name_raw.replace(' ', '')

        if well_name_clean not in coord_dict:
            print(f"  [SKIP] {well_name_raw}: missing coordinates")
            continue

        try:
            df = pd.read_excel(os.path.join(DATA_DIR, filename))
            if '深度' not in df.columns or '分层' not in df.columns:
                continue

            depth = df['深度'].values
            labels_raw = df['分层'].values

            labels = []
            valid_mask = []
            for l in labels_raw:
                l_str = str(l).strip()
                if l_str not in LABEL_MAP:
                    labels.append(-1)
                    valid_mask.append(False)
                else:
                    labels.append(LABEL_MAP[l_str])
                    valid_mask.append(True)

            labels = np.array(labels)
            valid_mask = np.array(valid_mask)

            if valid_mask.sum() == 0 or (FILTER_STRATEGY == 'keep_at_least_two'
                                         and len(np.unique(labels[valid_mask])) < 2):
                continue

            curves = np.zeros((len(depth), n_curves))
            well_nan = 0
            for i, cn in enumerate(curve_names):
                if cn in df.columns:
                    cleaned, n_nan = clean_curve(df[cn].values, global_min[cn], global_max[cn])
                    curves[:, i] = cleaned
                    well_nan += n_nan
                else:
                    curves[:, i] = 0.5

            for i in range(n_curves):
                if curves.shape[0] > 3:
                    curves[:, i] = median_filter(curves[:, i], size=3, mode='nearest')

            w = min(7, curves.shape[0] - 1)
            if w % 2 == 0:
                w -= 1
            if w >= 5:
                for i in range(n_curves):
                    curves[:, i] = savgol_filter(curves[:, i], w, 2, mode='nearest')

            total_raw_points += len(depth)
            total_cleaned_nan += well_nan

            kb = coord_dict[well_name_clean]['补心海拔']
            tvdss = kb - depth

            well_data_list.append({
                'curves': curves, 'labels': labels, 'valid_mask': valid_mask,
                'depth': depth, 'tvdss': tvdss, 'name': well_name_raw,
                'kb': kb, 'X': coord_dict[well_name_clean]['X'], 'Y': coord_dict[well_name_clean]['Y'],
            })
            coords_list.append([coord_dict[well_name_clean]['X'], coord_dict[well_name_clean]['Y']])
            altitude_list.append(kb)

            dominant = INV_LABEL_MAP[calculate_dominant_layer(labels[valid_mask])]
            print(f"  [OK] {well_name_raw} ({len(depth)}pts, {dominant})")

        except Exception as e:
            print(f"  [ERROR] {well_name_raw}: {e}")

    if len(well_data_list) == 0:
        raise ValueError("No well data loaded successfully.")

    print(f"\nStats: {total_raw_points} pts, cleaned {total_cleaned_nan} anomalies")

    coords = np.array(coords_list)
    altitudes = np.array(altitude_list).reshape(-1, 1)
    edge_index, edge_attr = [], []
    plane_dist_matrix = cdist(coords, coords, metric='euclidean')
    altitude_diff_matrix = cdist(altitudes, altitudes, metric='chebyshev')
    n_wells = len(well_data_list)

    for i in range(n_wells):
        for j in range(n_wells):
            if i == j:
                continue
            if altitude_diff_matrix[i, j] > ALTITUDE_DIFF_THRESH:
                continue
            if plane_dist_matrix[i, j] > PLANE_DIST_THRESH:
                continue
            edge_index.append([i, j])
            edge_attr.append(1.0 / (plane_dist_matrix[i, j] + 1e-6))

    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0,), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    print(f"Done: {n_wells} wells, {N_FEATURES}ch (GR,AC,DEN,LLD), {edge_index.size(1)} edges")
    return well_data_list, coords, edge_index, edge_attr, global_min, global_max
