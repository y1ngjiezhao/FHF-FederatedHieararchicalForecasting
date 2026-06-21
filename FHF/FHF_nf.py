from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


class LagDataset(Dataset):
    """Lag feature dataset with multi-step labels.

    X: (N, L)
    y: (N, H) or (N,)
    """

    def __init__(self, X, y):
        self.X = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        self.y = y_arr  # (N, H)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, i):
        x = self.X[i].reshape(-1, 1)  # (L, 1)
        y = self.y[i]                 # (H,)
        return {"x": torch.from_numpy(x), "y": torch.from_numpy(y)}

'''
def Datacollector(
    dff: pd.DataFrame,
    partition_col: str,
    target_col: str,
    lags: int = 12,
    ts: Iterable | None = None,
    fh: int = 0,
) -> Dict[int, pd.DataFrame]:
    """Create lagged and future-target columns for each partition.

    Notes
    -----
    - Uses the *actual* partition ids instead of assuming 0..n-1.
    - Uses the provided `target_col` rather than hard-coding `y`.
    """
    if ts is None:
        ts = dff[partition_col].drop_duplicates().tolist()

    arr: Dict[int, pd.DataFrame] = {}
    for pid in ts:
        df_ = dff[dff[partition_col] == pid].copy()
        for j in range(lags):
            df_[f"lags_{j + 1}"] = df_[target_col].shift(j + 1)
        if fh != 0:
            for j in range(fh):
                df_[f"post_{j + 1}"] = df_[target_col].shift(-(j + 1))
        arr[pid] = df_
    return arr'''

def Datacollector(
    dff: pd.DataFrame,
    partition_col: str,
    target_col: str,
    lags: int = 12,
    ts: Iterable | None = None,
    fh: int = 0,
) -> Dict[int, pd.DataFrame]:
    """Create lagged and future-target columns for each partition.

    Notes
    -----
    - Uses the *actual* partition ids instead of assuming 0..n-1.
    - Uses the provided `target_col` rather than hard-coding `y`.
    - Optimized to avoid DataFrame fragmentation.
    """
    if ts is None:
        ts = dff[partition_col].drop_duplicates().tolist()

    arr: Dict[int, pd.DataFrame] = {}
    
    for pid in ts:
        df_partition = dff[dff[partition_col] == pid].copy()
        
        # 批量创建滞后特征（避免循环插入列）
        lag_data = {}
        for j in range(lags):
            lag_data[f"lags_{j + 1}"] = df_partition[target_col].shift(j + 1)
        
        # 批量创建未来目标特征
        if fh != 0:
            for j in range(fh):
                lag_data[f"post_{j + 1}"] = df_partition[target_col].shift(-(j + 1))
        
        # 一次性合并所有新列
        if lag_data:
            new_cols_df = pd.DataFrame(lag_data, index=df_partition.index)
            df_partition = pd.concat([df_partition, new_cols_df], axis=1)
        
        arr[pid] = df_partition
    
    return arr


def load_clients_from_csv(
    csv_path: str,
    partition_col: str,
    series_col: str,
    time_col: str,
    target_col: str,
    lags: int,
    fh: int,
    truncated=None,
):
    """Load a long-format CSV and return (client_ids, cid2series, cid2df).

    Returns
    -------
    client_ids : list[int]
        Sorted client ids based on the *actual* partition ids.
    cid2series : dict[int, str]
        cid -> hierarchy path (series_col).
    cid2df : dict[int, pd.DataFrame]
        Lagged dataframe for each client.
    """
    raw = pd.read_csv(csv_path)

    missing = [
        c for c in [partition_col, series_col, time_col, target_col]
        if c not in raw.columns
    ]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    raw[time_col] = pd.to_datetime(raw[time_col], format="mixed", errors="coerce")
    raw = raw.dropna(subset=[time_col]).copy()

    if truncated is not None:
        cutoff = pd.to_datetime(truncated, format="mixed", errors="raise")
        raw = raw[raw[time_col] >= cutoff].copy()

    raw = raw.sort_values([partition_col, time_col]).reset_index(drop=True)
    client_ids = sorted(raw[partition_col].drop_duplicates().tolist())

    dict_ = Datacollector(
        raw,
        partition_col=partition_col,
        target_col=target_col,
        lags=lags,
        ts=client_ids,
        fh=fh,
    )
    df = pd.concat([dict_[key] for key in client_ids], axis=0).dropna().copy()

    lag_cols_reversed = list(
        reversed(
            sorted(
                [c for c in df.columns if c.startswith("lags_")],
                key=lambda x: int(x.split("_")[1]),
            )
        )
    )
    forecast_horizon = [target_col] + sorted(
        [c for c in df.columns if c.startswith("post_")],
        key=lambda x: int(x.split("_")[1]),
    )

    keep_cols = [series_col, partition_col, time_col] + lag_cols_reversed + forecast_horizon
    df = df[keep_cols].copy()

    # Standardise the main target column name expected by the rest of the pipeline.
    if target_col != 'y':
        df = df.rename(columns={target_col: 'y'})

    cid2series = {
        int(pid): str(s)
        for pid, s in df[[partition_col, series_col]].drop_duplicates().values
    }

    cid2df = {
        int(cid): df[df[partition_col] == cid].copy()
        for cid in client_ids
    }

    return [int(cid) for cid in client_ids], cid2series, cid2df
