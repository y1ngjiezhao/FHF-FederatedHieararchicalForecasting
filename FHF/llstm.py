import copy
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler


def Datacollector(dff, lags: int = 12, ts=range(15), fh: int = 0):
    from tqdm import tqdm

    arr = {}
    for i in tqdm(ts):
        df_ = dff[dff["partition_id"] == i].copy()
        for j in range(lags):
            df_[f"lags_{j + 1}"] = df_["y"].shift(j + 1)

        if fh != 0:
            for j in range(fh):
                df_[f"post_{j + 1}"] = df_["y"].shift(-(j + 1))
        arr[i] = df_
    return arr


class LagDataset(Dataset):
    def __init__(self, X_scaled, y_scaled):
        self.X = np.asarray(X_scaled, dtype=np.float32)
        self.y = np.asarray(y_scaled, dtype=np.float32)
        if self.y.ndim == 1:
            self.y = self.y.reshape(-1, 1)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, i):
        x = self.X[i].reshape(-1, 1)
        y1 = self.y[i]
        return {"x": torch.from_numpy(x), "y": torch.from_numpy(y1)}


class LSTM_reg(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size: int = 1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)


@dataclass
class PartitionArtifacts:
    model_state_dict: Dict[str, torch.Tensor]
    scaler_x: StandardScaler
    scaler_y: StandardScaler
    train_loss_history: List[float]
    test_metrics_scaled: Dict[str, float]
    n_train: int
    n_test: int
    n_gap: int


def train_fn(model, trainloader, optimizer, device="cpu", clip_grad=1.0, take_last=True):
    criterion = nn.MSELoss()
    model.to(device)
    model.train()

    total_loss = 0.0
    total_n = 0

    for batch in trainloader:
        x = batch["x"].to(device).float()
        y = batch["y"].to(device).float()

        optimizer.zero_grad(set_to_none=True)
        y_hat = model(x)

        if take_last and y_hat.dim() == 3 and y.dim() == 2:
            y_hat = y_hat[:, -1, :]

        loss = criterion(y_hat, y)
        loss.backward()

        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_n += bs

    return total_loss / max(total_n, 1)


@torch.no_grad()
def test_fn(model, testloader, device="cpu", take_last=True):
    criterion = nn.MSELoss(reduction="sum")
    model.to(device)
    model.eval()

    sum_sq_loss = 0.0
    sum_abs_err = 0.0
    total_n = 0

    for batch in testloader:
        x = batch["x"].to(device).float()
        y = batch["y"].to(device).float()
        y_hat = model(x)

        if take_last and y_hat.dim() == 3 and y.dim() == 2:
            y_hat = y_hat[:, -1, :]

        sum_sq_loss += criterion(y_hat, y).item()
        sum_abs_err += torch.abs(y_hat - y).sum().item()
        total_n += y.numel()

    mse = sum_sq_loss / max(total_n, 1)
    rmse = mse ** 0.5
    mae = sum_abs_err / max(total_n, 1)
    return mse, rmse, mae


@torch.no_grad()
def predict_loader(model, loader, device="cpu", take_last=True):
    model.to(device)
    model.eval()

    preds, trues = [], []
    for batch in loader:
        xb = batch["x"].to(device).float()
        yb = batch["y"].to(device).float()

        out = model(xb)
        if take_last and out.dim() == 3 and yb.dim() == 2:
            out = out[:, -1, :]

        if out.dim() == 1:
            out = out.unsqueeze(-1)
        if yb.dim() == 1:
            yb = yb.unsqueeze(-1)

        preds.append(out.detach().cpu())
        trues.append(yb.detach().cpu())

    preds_all = torch.cat(preds, dim=0).numpy() if preds else np.empty((0, 1), dtype=np.float32)
    trues_all = torch.cat(trues, dim=0).numpy() if trues else np.empty((0, 1), dtype=np.float32)
    return preds_all, trues_all


def _strict_window_split_positions(
    n_windows: int,
    lags: int,
    fh: int,
    train_ratio: float,
    context: str,
    drop_boundary_gap: bool = True,
):
    if n_windows <= 1:
        raise ValueError(f"{context} has only {n_windows} usable window(s) after lag/post generation")

    n_raw = n_windows + int(lags) + int(fh)
    n_train_raw = int(float(train_ratio) * n_raw)
    n_train_windows = n_train_raw - int(lags) - int(fh)
    gap = int(fh) if drop_boundary_gap else 0
    test_start = n_train_windows + gap

    if n_train_windows <= 0:
        raise ValueError(
            f"{context} cannot form a strict leak-free train split with lags={lags}, fh={fh}, "
            f"train_ratio={train_ratio}. Usable windows={n_windows}, inferred raw length={n_raw}, "
            f"train windows={n_train_windows}."
        )
    if test_start >= n_windows:
        raise ValueError(
            f"{context} cannot form a non-empty strict test split with lags={lags}, fh={fh}, "
            f"train_ratio={train_ratio}, drop_boundary_gap={drop_boundary_gap}. "
            f"Usable windows={n_windows}, train windows={n_train_windows}, test_start={test_start}."
        )
    return n_train_windows, test_start


def strict_time_ordered_partition_split(
    df: pd.DataFrame,
    partition_col: str = "partition_id",
    train_ratio: float = 0.8,
    lags: int = 48,
    fh: int = 0,
    time_col: str = "ds",
    drop_boundary_gap: bool = True,
):
    train_idx, test_idx = [], []
    parts = sorted(df[partition_col].unique())
    for pid in parts:
        partition = df[df[partition_col] == pid].copy()
        if time_col is not None and time_col in partition.columns:
            partition = partition.sort_values(time_col)
        sub_idx = partition.index.to_numpy()
        n_train, test_start = _strict_window_split_positions(
            n_windows=len(sub_idx),
            lags=lags,
            fh=fh,
            train_ratio=train_ratio,
            context=f"partition {pid}",
            drop_boundary_gap=drop_boundary_gap,
        )
        train_idx.extend(sub_idx[:n_train])
        test_idx.extend(sub_idx[test_start:])
    return np.asarray(train_idx, dtype=np.int64), np.asarray(test_idx, dtype=np.int64)


def build_partition_loaders(
    partition_df: pd.DataFrame,
    lag_cols_reversed: List[str],
    forecast_horizon: List[str],
    batch_size: int,
    train_ratio: float = 0.8,
    lags: int = 48,
    fh: int = 0,
    time_col: str = "ds",
    drop_boundary_gap: bool = True,
):
    partition = partition_df.copy()
    if time_col is not None and time_col in partition.columns:
        partition = partition.sort_values(time_col).reset_index(drop=True)

    X = partition[lag_cols_reversed].to_numpy(dtype=np.float32)
    y = partition[forecast_horizon].to_numpy(dtype=np.float32)
    n_train, test_start = _strict_window_split_positions(
        n_windows=len(partition),
        lags=lags,
        fh=fh,
        train_ratio=train_ratio,
        context="partition",
        drop_boundary_gap=drop_boundary_gap,
    )

    X_tr, y_tr = X[:n_train], y[:n_train]
    X_ts, y_ts = X[test_start:], y[test_start:]

    sx = StandardScaler()
    sy = StandardScaler()
    X_tr_s = sx.fit_transform(X_tr)
    y_tr_s = sy.fit_transform(y_tr)
    X_ts_s = sx.transform(X_ts)
    y_ts_s = sy.transform(y_ts)

    train_ds = LagDataset(X_tr_s, y_tr_s)
    test_ds = LagDataset(X_ts_s, y_ts_s)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader_eval = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return {
        "sx": sx,
        "sy": sy,
        "train_loader": train_loader,
        "train_loader_eval": train_loader_eval,
        "test_loader": test_loader,
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "n_gap": int(test_start - n_train),
    }



# -----------------------------------------------------------------------------
# Modified evaluation / saving helpers: MAE, MASE, RMSSE + Naive baseline
# -----------------------------------------------------------------------------

def _as_2d(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def _safe_scale_denominators(train_true, eps: float = 1e-12):
    """Return MASE and RMSSE denominators using training true values only."""
    train_true = _as_2d(train_true)
    if train_true.shape[0] <= 1:
        return np.nan, np.nan
    diff = np.diff(train_true, axis=0)
    mase_denom = np.nanmean(np.abs(diff))
    rmsse_denom = np.nanmean(diff ** 2)
    if (not np.isfinite(mase_denom)) or mase_denom <= eps:
        mase_denom = np.nan
    if (not np.isfinite(rmsse_denom)) or rmsse_denom <= eps:
        rmsse_denom = np.nan
    return float(mase_denom), float(rmsse_denom)


def _metric_values(y_true, y_pred, train_true=None, eps: float = 1e-12):
    y_true = _as_2d(y_true)
    y_pred = _as_2d(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Metric shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
    err = y_true - y_pred
    mae = float(np.nanmean(np.abs(err)))
    mse = float(np.nanmean(err ** 2))
    out = {"MAE": mae, "MSE": mse}
    if train_true is None:
        out["MASE"] = np.nan
        out["RMSSE"] = np.nan
    else:
        mase_denom, rmsse_denom = _safe_scale_denominators(train_true, eps=eps)
        out["MASE"] = float(mae / mase_denom) if np.isfinite(mase_denom) else np.nan
        out["RMSSE"] = float(np.sqrt(mse / rmsse_denom)) if np.isfinite(rmsse_denom) else np.nan
    return out


@torch.no_grad()
def evaluate_loader_losses(model, loader, scaler_y=None, device="cpu", take_last=True):
    """Return MSE on scaled data and, if scaler_y is supplied, original-scale MSE."""
    pred_s, true_s = predict_loader(model, loader, device=device, take_last=take_last)
    pred_s = _as_2d(pred_s)
    true_s = _as_2d(true_s)
    avg_normalised_loss = float(np.nanmean((pred_s - true_s) ** 2)) if true_s.size else np.nan
    if scaler_y is None:
        avg_actual_loss = np.nan
    else:
        pred_a = scaler_y.inverse_transform(pred_s)
        true_a = scaler_y.inverse_transform(true_s)
        avg_actual_loss = float(np.nanmean((pred_a - true_a) ** 2)) if true_a.size else np.nan
    return avg_normalised_loss, avg_actual_loss


def make_naive_forecast_from_test_frame(test_frame: pd.DataFrame, output_dim: int, lag_col: str = "lags_1"):
    """Simple persistence baseline: use the most recent available lag as forecast."""
    if lag_col not in test_frame.columns:
        raise KeyError(f"Cannot compute naive forecast because {lag_col!r} is missing.")
    naive_1d = test_frame[lag_col].to_numpy(dtype=np.float64)
    return np.repeat(naive_1d[:, None], int(output_dim), axis=1)


def standardize_method_name(name: str, base_model_name: str):
    """Map HierarchicalForecast column names into compact method labels."""
    s = str(name)
    if s == base_model_name:
        return "base"
    if s == "naive":
        return "naive"
    lower = s.lower()
    if "bottomup" in lower:
        return "bu"
    if "mint_shrink" in lower:
        return "mint_shrinkage"
    if "wls_var" in lower:
        return "mint_var"
    if "wls_struct" in lower:
        return "wls_structure"
    if "ols" in lower:
        return "mint_ols"
    return s.replace(f"{base_model_name}/", "")


def make_reconciliation_frames(
    df: pd.DataFrame,
    train_idx,
    test_idx,
    parts,
    dict_tr_pred: Dict[int, np.ndarray],
    dict_pred: Dict[int, np.ndarray],
    forecast_horizon: List[str],
    base_model_name: str,
    horizon_idx: int,
):
    """Build train/test frames for HierarchicalForecast reconciliation."""
    hcol = forecast_horizon[horizon_idx]
    train_frame = df.loc[train_idx, ["unique_id", "partition_id", "ds", hcol]].copy()
    test_frame = df.loc[test_idx, ["unique_id", "partition_id", "ds", "lags_1", hcol]].copy()

    train_frame[base_model_name] = np.concatenate([
        _as_2d(dict_tr_pred[int(pid)])[:, horizon_idx] for pid in parts
    ]).reshape(-1)
    test_frame[base_model_name] = np.concatenate([
        _as_2d(dict_pred[int(pid)])[:, horizon_idx] for pid in parts
    ]).reshape(-1)

    train_frame = train_frame.rename(columns={hcol: "y"})
    test_frame = test_frame.rename(columns={hcol: "y"})
    test_frame["naive"] = test_frame["lags_1"].astype(float)

    y_hat_df = test_frame[["unique_id", "ds", "y", base_model_name]].copy()
    y_df = train_frame[["unique_id", "ds", "y", base_model_name]].copy()
    return y_df, y_hat_df, train_frame, test_frame


def evaluate_reconciliation_results(
    recon_results: Dict[int, pd.DataFrame],
    train_frames: Dict[int, pd.DataFrame],
    test_frames: Dict[int, pd.DataFrame],
    forecast_horizon: List[str],
    base_model_name: str,
    output_prefix: str = None,
    approach: str = None,
    round_logs=None,
    timings: Dict[str, float] = None,
):
    """Evaluate and optionally save base/reconciled/naive forecasts with MAE, MASE, RMSSE."""
    forecast_rows = []
    metric_rows = []

    for h_idx, rr0 in recon_results.items():
        rr = rr0.copy()
        rr["ds"] = pd.to_datetime(rr["ds"], format="mixed")
        tf = test_frames[h_idx].copy()
        trf = train_frames[h_idx].copy()
        tf["ds"] = pd.to_datetime(tf["ds"], format="mixed")
        trf["ds"] = pd.to_datetime(trf["ds"], format="mixed")

        if "naive" not in rr.columns:
            rr = rr.merge(tf[["unique_id", "ds", "naive"]], on=["unique_id", "ds"], how="left")

        base_cols = ["unique_id", "ds", "y"]
        method_cols = [c for c in rr.columns if c not in base_cols]
        rr["level"] = rr["unique_id"].astype(str).str.count("/") + 1
        levels = sorted(rr["level"].unique())
        min_level, max_level = min(levels), max(levels)

        denom = {}
        for uid, g in trf.sort_values("ds").groupby("unique_id"):
            denom[uid] = _safe_scale_denominators(g["y"].to_numpy(dtype=float))

        for _, row in rr.iterrows():
            out = {
                "unique_id": row["unique_id"],
                "ds": row["ds"],
                "horizon_index": int(h_idx),
                "target_col": forecast_horizon[h_idx],
                "level": int(row["level"]),
                "y_true": float(row["y"]),
            }
            for m in method_cols:
                out[standardize_method_name(m, base_model_name)] = float(row[m]) if pd.notna(row[m]) else np.nan
            forecast_rows.append(out)

        for (level, uid), g in rr.groupby(["level", "unique_id"]):
            y_true = g["y"].to_numpy(dtype=float)
            mase_denom, rmsse_denom = denom.get(uid, (np.nan, np.nan))
            role = "top" if level == min_level else ("bottom" if level == max_level else "middle")
            for m in method_cols:
                method_name = standardize_method_name(m, base_model_name)
                y_pred = g[m].to_numpy(dtype=float)
                err = y_true - y_pred
                mae = float(np.nanmean(np.abs(err)))
                mse = float(np.nanmean(err ** 2))
                mase = mae / mase_denom if np.isfinite(mase_denom) else np.nan
                rmsse = np.sqrt(mse / rmsse_denom) if np.isfinite(rmsse_denom) else np.nan
                metric_rows.append({
                    "unique_id": uid,
                    "level": int(level),
                    "role": role,
                    "horizon_index": int(h_idx),
                    "target_col": forecast_horizon[h_idx],
                    "method": method_name,
                    "MAE": mae,
                    "MASE": float(mase) if np.isfinite(mase) else np.nan,
                    "RMSSE": float(rmsse) if np.isfinite(rmsse) else np.nan,
                })

    forecast_table = pd.DataFrame(forecast_rows)
    per_series_metrics = pd.DataFrame(metric_rows)

    metrics_by_level = (
        per_series_metrics
        .groupby(["level", "role", "method"], as_index=False)
        .agg(MAE=("MAE", "mean"), MASE=("MASE", "mean"), RMSSE=("RMSSE", "mean"), n_series=("unique_id", "nunique"))
        .sort_values(["level", "method"])
        .reset_index(drop=True)
    )
    overall_metrics = (
        per_series_metrics
        .groupby(["method"], as_index=False)
        .agg(MAE=("MAE", "mean"), MASE=("MASE", "mean"), RMSSE=("RMSSE", "mean"), n_series=("unique_id", "nunique"))
        .sort_values(["method"])
        .reset_index(drop=True)
    )
    if approach is not None:
        per_series_metrics["approach"] = approach
        metrics_by_level["approach"] = approach
        overall_metrics["approach"] = approach

    timing_df = pd.DataFrame([{"module": k, "seconds": v} for k, v in (timings or {}).items()])
    output_paths = {}
    if output_prefix is not None:
        output_paths = {
            "forecasts": f"{output_prefix}_forecasts.csv",
            "per_series_metrics": f"{output_prefix}_per_series_metrics.csv",
            "metrics_by_level": f"{output_prefix}_metrics_by_level.csv",
            "overall_metrics": f"{output_prefix}_overall_metrics.csv",
            "round_logs": f"{output_prefix}_round_logs.csv",
            "timing": f"{output_prefix}_timing.csv",
        }
        forecast_table.to_csv(output_paths["forecasts"], index=False)
        per_series_metrics.to_csv(output_paths["per_series_metrics"], index=False)
        metrics_by_level.to_csv(output_paths["metrics_by_level"], index=False)
        overall_metrics.to_csv(output_paths["overall_metrics"], index=False)
        pd.DataFrame(round_logs if round_logs is not None else []).to_csv(output_paths["round_logs"], index=False)
        timing_df.to_csv(output_paths["timing"], index=False)

    return {
        "forecast_table": forecast_table,
        "per_series_metrics": per_series_metrics,
        "metrics_by_level": metrics_by_level,
        "overall_metrics": overall_metrics,
        "timing_df": timing_df,
        "output_paths": output_paths,
    }


def compute_metrics_from_dicts(dict_true, dict_pred, dict_train_true=None, h_idx=0):
    """Compute MAE, MASE and RMSSE for dictionaries keyed by partition_id."""
    parts = sorted(dict_true.keys())
    rows = []
    for pid in parts:
        yt = _as_2d(dict_true[pid])[:, h_idx]
        yp = _as_2d(dict_pred[pid])[:, h_idx]
        tr = _as_2d(dict_train_true[pid])[:, h_idx] if dict_train_true is not None and pid in dict_train_true else None
        met = _metric_values(yt, yp, train_true=tr)
        rows.append({
            "partition_id": pid,
            "n_test": len(yt),
            "MAE": met["MAE"],
            "MASE": met["MASE"],
            "RMSSE": met["RMSSE"],
        })

    dfm = pd.DataFrame(rows).sort_values("partition_id").reset_index(drop=True)
    overall = pd.DataFrame([{
        "partition_id": "Overall",
        "n_test": int(dfm["n_test"].sum()),
        "MAE": float(dfm["MAE"].mean()),
        "MASE": float(dfm["MASE"].mean()),
        "RMSSE": float(dfm["RMSSE"].mean()),
    }])
    return pd.concat([dfm, overall], ignore_index=True)


def run_localised(
    df: pd.DataFrame,
    lag_cols_reversed: List[str],
    forecast_horizon: List[str],
    input_size: int = 1,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    batch_size: int = 256,
    epochs: int = 100,
    lr: float = 1e-3,
    train_ratio: float = 0.8,
    partition_col: str = "partition_id",
    device: Optional[str] = None,
    clip_grad: float = 1.0,
    verbose: bool = True,
    disable_mkldnn_on_cpu: bool = True,
    lags: int = 48,
    fh: int = 0,
    time_col: str = "ds",
    drop_boundary_gap: bool = True,
    early_stop_enabled: bool = True,
    early_stop_tol: float = 1e-5,
    min_epochs: int = 20,
):
    """Train one independent LSTM per partition with normalised/actual loss reporting.

    Early stopping is applied inside each partition/local-model training loop.
    That is, each local model independently stops after ``min_epochs`` when its
    own ``abs(current_avg_normalised_loss - previous_avg_normalised_loss)`` is
    below the shared threshold ``early_stop_tol``. The aggregate epoch-level
    loss reported by this function is for monitoring only and is never used to
    stop all local models together.
    """
    run_t0 = time.perf_counter()
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    if disable_mkldnn_on_cpu and device.type == "cpu":
        torch.backends.mkldnn.enabled = False

    parts = sorted(df[partition_col].unique())
    train_idx, test_idx = strict_time_ordered_partition_split(
        df,
        partition_col=partition_col,
        train_ratio=train_ratio,
        lags=lags,
        fh=fh,
        time_col=time_col,
        drop_boundary_gap=drop_boundary_gap,
    )

    dict_pred: Dict[int, np.ndarray] = {}
    dict_true: Dict[int, np.ndarray] = {}
    dict_tr_pred: Dict[int, np.ndarray] = {}
    dict_tr_true: Dict[int, np.ndarray] = {}
    dict_naive: Dict[int, np.ndarray] = {}
    dict_train_meta: Dict[int, pd.DataFrame] = {}
    dict_test_meta: Dict[int, pd.DataFrame] = {}
    artifacts: Dict[int, PartitionArtifacts] = {}
    trained_models: Dict[int, nn.Module] = {}
    round_logs_by_partition = []
    partition_timings = []

    train_t0 = time.perf_counter()
    for pid in parts:
        part_t0 = time.perf_counter()
        partition = df[df[partition_col] == pid].copy()
        if time_col is not None and time_col in partition.columns:
            partition = partition.sort_values(time_col).reset_index(drop=True)
        else:
            partition = partition.reset_index(drop=True)

        loaders = build_partition_loaders(
            partition_df=partition,
            lag_cols_reversed=lag_cols_reversed,
            forecast_horizon=forecast_horizon,
            batch_size=batch_size,
            train_ratio=train_ratio,
            lags=lags,
            fh=fh,
            time_col=time_col,
            drop_boundary_gap=drop_boundary_gap,
        )

        model = LSTM_reg(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            output_size=len(forecast_horizon),
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        history_norm, history_actual = [], []
        prev_norm = None
        stopped_early = False
        stop_epoch = epochs
        stop_reason = "max_epochs"

        for e in range(epochs):
            _ = train_fn(model, loaders["train_loader"], optimizer, device=device, clip_grad=clip_grad)
            avg_norm, avg_actual = evaluate_loader_losses(
                model,
                loaders["train_loader_eval"],
                scaler_y=loaders["sy"],
                device=device,
            )
            delta = abs(avg_norm - prev_norm) if prev_norm is not None and np.isfinite(prev_norm) else np.nan
            should_stop = bool(
                early_stop_enabled
                and (e + 1) >= int(min_epochs)
                and np.isfinite(delta)
                and delta < float(early_stop_tol)
            )

            history_norm.append(float(avg_norm))
            history_actual.append(float(avg_actual))
            round_logs_by_partition.append({
                "partition_id": int(pid),
                "epoch": int(e + 1),
                "avg_normalised_loss": float(avg_norm),
                "avg_actual_loss": float(avg_actual),
                "normalised_loss_delta": float(delta) if np.isfinite(delta) else np.nan,
                "stopped": should_stop,
            })

            if verbose:
                print(
                    f"[Localised] pid={pid} epoch {e + 1:03d}/{epochs:03d} | "
                    f"avg_normalised_loss={avg_norm:.6f} | avg_actual_loss={avg_actual:.6f} | "
                    f"delta={delta if np.isfinite(delta) else np.nan:.6g} | stopped={should_stop}"
                )

            prev_norm = avg_norm
            if should_stop:
                stopped_early = True
                stop_epoch = e + 1
                stop_reason = f"normalised_loss_delta<{early_stop_tol} after min_epochs={min_epochs}"
                break

        mse, rmse, mae = test_fn(model, loaders["test_loader"], device=device)
        pred_test_s, true_test_s = predict_loader(model, loaders["test_loader"], device=device)
        pred_train_s, true_train_s = predict_loader(model, loaders["train_loader_eval"], device=device)

        sy = loaders["sy"]
        dict_pred[int(pid)] = sy.inverse_transform(pred_test_s)
        dict_true[int(pid)] = sy.inverse_transform(true_test_s)
        dict_tr_pred[int(pid)] = sy.inverse_transform(pred_train_s)
        dict_tr_true[int(pid)] = sy.inverse_transform(true_train_s)

        n_train = loaders["n_train"]
        test_start = loaders["n_train"] + loaders["n_gap"]
        dict_train_meta[int(pid)] = partition.iloc[:n_train][["unique_id", partition_col, time_col] + forecast_horizon].copy()
        dict_test_meta[int(pid)] = partition.iloc[test_start:][["unique_id", partition_col, time_col, "lags_1"] + forecast_horizon].copy()
        dict_naive[int(pid)] = make_naive_forecast_from_test_frame(dict_test_meta[int(pid)], output_dim=len(forecast_horizon))

        artifacts[int(pid)] = PartitionArtifacts(
            model_state_dict=copy.deepcopy(model.state_dict()),
            scaler_x=loaders["sx"],
            scaler_y=loaders["sy"],
            train_loss_history=history_norm,
            test_metrics_scaled={"mse": float(mse), "rmse": float(rmse), "mae": float(mae)},
            n_train=loaders["n_train"],
            n_test=loaders["n_test"],
            n_gap=loaders["n_gap"],
        )
        # Attach extra fields without changing the dataclass constructor/API.
        artifacts[int(pid)].actual_loss_history = history_actual
        artifacts[int(pid)].stopped_early = stopped_early
        artifacts[int(pid)].stop_epoch = stop_epoch
        artifacts[int(pid)].stop_reason = stop_reason
        trained_models[int(pid)] = model.cpu()
        partition_timings.append({"partition_id": int(pid), "training_sec": time.perf_counter() - part_t0})

    training_sec = time.perf_counter() - train_t0
    round_log_df = pd.DataFrame(round_logs_by_partition)

    # Per-local-model stopping summary. This is the authoritative early-stop
    # record for LLSTM because each partition trains and stops independently.
    local_stop_summary = []
    for pid in sorted(artifacts.keys()):
        art = artifacts[pid]
        norm_hist = list(getattr(art, "train_loss_history", []))
        actual_hist = list(getattr(art, "actual_loss_history", []))
        local_stop_summary.append({
            "partition_id": int(pid),
            "stopped_early": bool(getattr(art, "stopped_early", False)),
            "stop_epoch": int(getattr(art, "stop_epoch", len(norm_hist))),
            "stop_reason": str(getattr(art, "stop_reason", "max_epochs")),
            "early_stop_tol": float(early_stop_tol),
            "min_epochs": int(min_epochs),
            "final_avg_normalised_loss": float(norm_hist[-1]) if norm_hist else np.nan,
            "final_avg_actual_loss": float(actual_hist[-1]) if actual_hist else np.nan,
            "n_train": int(getattr(art, "n_train", 0)),
            "n_test": int(getattr(art, "n_test", 0)),
        })
    local_stop_summary_df = pd.DataFrame(local_stop_summary)

    if len(round_log_df) > 0:
        # This epoch-level table is only a monitoring/reporting summary over
        # the local models that were still active and emitted a log in that
        # epoch. It is not used for early stopping.
        round_log_df["stopped_this_epoch"] = round_log_df["stopped"].astype(bool)
        round_logs = (
            round_log_df
            .groupby("epoch", as_index=False)
            .agg(
                avg_normalised_loss=("avg_normalised_loss", "mean"),
                avg_actual_loss=("avg_actual_loss", "mean"),
                normalised_loss_delta=("normalised_loss_delta", "mean"),
                n_active_partitions_reported=("partition_id", "nunique"),
                n_partitions_stopped_this_epoch=("stopped_this_epoch", "sum"),
            )
            .to_dict("records")
        )
    else:
        round_logs = []

    metrics_df = compute_metrics_from_dicts(dict_true, dict_pred, dict_train_true=dict_tr_true, h_idx=0)
    naive_metrics_df = compute_metrics_from_dicts(dict_true, dict_naive, dict_train_true=dict_tr_true, h_idx=0)

    timings = {
        "training_sec": float(training_sec),
        "total_sec": float(time.perf_counter() - run_t0),
    }

    final_avg_normalised_loss = (
        float(local_stop_summary_df["final_avg_normalised_loss"].mean())
        if len(local_stop_summary_df) else np.nan
    )
    final_avg_actual_loss = (
        float(local_stop_summary_df["final_avg_actual_loss"].mean())
        if len(local_stop_summary_df) else np.nan
    )

    return {
        "parts": [int(p) for p in parts],
        "train_idx": train_idx,
        "test_idx": test_idx,
        "dict_pred": dict_pred,
        "dict_true": dict_true,
        "dict_tr_pred": dict_tr_pred,
        "dict_tr_true": dict_tr_true,
        "dict_naive": dict_naive,
        "dict_train_meta": dict_train_meta,
        "dict_test_meta": dict_test_meta,
        "artifacts": artifacts,
        "models": trained_models,
        "metrics_df": metrics_df,
        "naive_metrics_df": naive_metrics_df,
        "round_logs": round_logs,
        "round_logs_by_partition": round_logs_by_partition,
        "partition_timings": partition_timings,
        "local_stop_summary": local_stop_summary,
        "local_stop_summary_df": local_stop_summary_df,
        "timings": timings,
        "device": str(device),
        "stopped_early": any(getattr(a, "stopped_early", False) for a in artifacts.values()),
        "final_avg_normalised_loss": final_avg_normalised_loss,
        "final_avg_actual_loss": final_avg_actual_loss,
    }
