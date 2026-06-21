from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler
from FHF_models import build_model
from FHF_nf import LagDataset


class Client:
    def __init__(self, cid: int, series_name: str, df_one: pd.DataFrame, args: Args):
        self.cid = int(cid)
        self.series_name = str(series_name)
        self.df = df_one.copy()
        self.args = args
        self.device = args.device

        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        self.train_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        self.train_loader_unshuffle: Optional[DataLoader] = None

        self.model: nn.Module = build_model(args).to(self.device)
        if args.optimizer.lower() == "sgd":
            self.opt = torch.optim.SGD(
                self.model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
                momentum=0.0,
            )
        else:
            self.opt = torch.optim.Adam(
                self.model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        self.criterion = nn.MSELoss()

        # reconciliation artifacts
        self.P_col: Optional[np.ndarray] = None           # (n_series, H)
        self.base_forecast: Optional[np.ndarray] = None   # (T_test, H)
        self.reconciled_forecast: Optional[np.ndarray] = None
        self.reconciled_forecasts: Dict[str, np.ndarray] = {}
        self.residual_samples: Optional[np.ndarray] = None  # (T_fit, H)
        self.test_true: Optional[np.ndarray] = None
        self.y1: Optional[np.ndarray] = None
        self.y2: Optional[np.ndarray] = None

    def prepare_data(self):
        args = self.args
        df = self.df.copy()

        # Keep each client's windows in chronological order before splitting.
        time_col = getattr(args, "time_col", None)
        if time_col is not None and time_col in df.columns:
            df = df.sort_values(time_col).reset_index(drop=True)

        lag_cols_reversed = list(
            reversed(
                sorted(
                    [c for c in df.columns if c.startswith("lags_")],
                    key=lambda x: int(x.split("_")[1]),
                )
            )
        )
        forecast_horizon = ["y"] + sorted(
            [c for c in df.columns if c.startswith("post_")],
            key=lambda x: int(x.split("_")[1]),
        )

        X = df[lag_cols_reversed].to_numpy()
        Y = df[forecast_horizon].to_numpy()

        n_windows = X.shape[0]
        if n_windows <= 1:
            raise ValueError(
                f"Client {self.cid} ({self.series_name}) has only {n_windows} usable window(s) "
                "after lag/post generation; cannot create strict train/test split."
            )

        # Strict leak-free split while still following:
        #   1) generate lag/post windows first,
        #   2) split afterwards,
        #   3) drop train windows whose targets cross into the raw test period.
        #
        # After lag/post generation we keep:
        #   n_windows = T_raw - lags - fh
        # because each usable row predicts [y, post_1, ..., post_fh].
        #
        # We first define the raw-time split:
        #   n_train_raw = floor((1 - test_ratio) * T_raw)
        #
        # Window j in the windowed dataframe maps to raw target time index i = lags + j,
        # and its final target is at i + fh. Leak-free training requires:
        #   i + fh <= n_train_raw - 1
        # which gives the strict count of train windows:
        #   n_tr = n_train_raw - lags - fh
        n_raw = n_windows + int(args.lags) + int(args.fh)
        n_train_raw = int((1.0 - float(args.test_ratio)) * n_raw)
        n_tr = n_train_raw - int(args.lags) - int(args.fh)

        if n_tr <= 0 or n_tr >= n_windows:
            raise ValueError(
                f"Client {self.cid} ({self.series_name}) cannot form a strict leak-free split "
                f"with lags={args.lags}, fh={args.fh}, test_ratio={args.test_ratio}. "
                f"Usable windows={n_windows}, inferred raw length={n_raw}, train windows={n_tr}."
            )

        X_tr, Y_tr = X[:n_tr], Y[:n_tr]
        X_ts, Y_ts = X[n_tr:], Y[n_tr:]

        X_tr_s = self.x_scaler.fit_transform(X_tr)
        Y_tr_s = self.y_scaler.fit_transform(Y_tr)
        X_ts_s = self.x_scaler.transform(X_ts)
        Y_ts_s = self.y_scaler.transform(Y_ts)

        self.train_loader = DataLoader(LagDataset(X_tr_s, Y_tr_s), batch_size=args.batch_size, shuffle=True)
        self.test_loader = DataLoader(LagDataset(X_ts_s, Y_ts_s), batch_size=args.batch_size, shuffle=False)
        self.train_loader_unshuffle = DataLoader(LagDataset(X_tr_s, Y_tr_s), batch_size=args.batch_size, shuffle=False)

    def set_state_dict(self, sd: Dict[str, torch.Tensor]):
        self.model.load_state_dict(sd, strict=True)

    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def num_train_samples(self) -> int:
        if self.train_loader is None:
            return 0
        return len(self.train_loader.dataset)

    def local_train(self) -> float:
        """Run local training and return the average training loss.

        The returned value is the sample-weighted average MSE over all local
        epochs and mini-batches. It is used only for training-loop logging and
        early stopping; callers may safely ignore it.
        """
        if self.train_loader is None:
            return float("nan")

        self.model.to(self.device)
        self.model.train()

        total_loss = 0.0
        total_obs = 0
        for _ in range(int(self.args.local_epochs)):
            for batch in self.train_loader:
                x = batch["x"].to(self.device).float()
                y = batch["y"].to(self.device).float()
                self.opt.zero_grad()
                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss.backward()
                self.opt.step()

                batch_size = int(y.shape[0])
                total_loss += float(loss.detach().cpu().item()) * batch_size
                total_obs += batch_size

        if total_obs <= 0:
            return float("nan")
        return total_loss / float(total_obs)


    def evaluate_train_losses(self) -> Dict[str, float]:
        """Evaluate training MSE on both normalised and original scales.

        Returns
        -------
        dict
            - ``normalised_loss``: MSE computed directly on scaled targets.
            - ``actual_loss``: MSE after inverse-transforming predictions and
              targets back to the original target scale.

        This method is deliberately evaluation-only. It does not update model
        parameters and it does not use the test set.
        """
        if self.train_loader is None:
            return {"normalised_loss": float("nan"), "actual_loss": float("nan")}

        self.model.to(self.device)
        self.model.eval()

        preds, trues = [], []
        total_norm_loss = 0.0
        total_obs = 0

        def _ensure_2d(t: torch.Tensor) -> torch.Tensor:
            return t.unsqueeze(-1) if t.dim() == 1 else t

        with torch.no_grad():
            for batch in self.train_loader:
                x = batch["x"].to(self.device).float()
                y = _ensure_2d(batch["y"].to(self.device).float())
                pred = _ensure_2d(self.model(x))

                loss = self.criterion(pred, y)
                n = int(y.shape[0])
                total_norm_loss += float(loss.detach().cpu().item()) * n
                total_obs += n

                preds.append(pred.detach().cpu())
                trues.append(y.detach().cpu())

        if total_obs <= 0 or len(preds) == 0:
            return {"normalised_loss": float("nan"), "actual_loss": float("nan")}

        pred_s = torch.cat(preds, dim=0).numpy().astype(np.float64)
        true_s = torch.cat(trues, dim=0).numpy().astype(np.float64)
        if pred_s.ndim == 1:
            pred_s = pred_s.reshape(-1, 1)
        if true_s.ndim == 1:
            true_s = true_s.reshape(-1, 1)

        pred_actual = self.y_scaler.inverse_transform(pred_s)
        true_actual = self.y_scaler.inverse_transform(true_s)
        actual_loss = float(np.nanmean((true_actual - pred_actual) ** 2))

        return {
            "normalised_loss": float(total_norm_loss / float(total_obs)),
            "actual_loss": actual_loss,
        }

    def compute_base_forecast_and_residuals(self):
        self.model.to(self.device)
        self.model.eval()

        preds, trues = [], []
        tr_preds, tr_trues = [], []

        def _ensure_2d(t: torch.Tensor) -> torch.Tensor:
            return t.unsqueeze(-1) if t.dim() == 1 else t

        with torch.no_grad():
            for batch in self.test_loader:
                xb = batch["x"].to(self.device).float()
                yb = batch["y"].to(self.device).float()
                out = _ensure_2d(self.model(xb))
                yb2 = _ensure_2d(yb)
                preds.append(out.detach().cpu())
                trues.append(yb2.detach().cpu())

            for batch in self.train_loader_unshuffle:
                xb = batch["x"].to(self.device).float()
                yb = batch["y"].to(self.device).float()
                out = _ensure_2d(self.model(xb))
                yb2 = _ensure_2d(yb)
                tr_preds.append(out.detach().cpu())
                tr_trues.append(yb2.detach().cpu())

        def _cat_to_numpy_2d(tensor_list, fallback_h: int | None = None):
            if len(tensor_list) == 0:
                h = 1 if fallback_h is None else int(fallback_h)
                return np.empty((0, h), dtype=np.float64)
            arr = torch.cat(tensor_list, dim=0).numpy().astype(np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            return arr

        preds_arr = _cat_to_numpy_2d(preds)
        trues_arr = _cat_to_numpy_2d(trues, fallback_h=preds_arr.shape[1])
        tr_preds_arr = _cat_to_numpy_2d(tr_preds)
        tr_trues_arr = _cat_to_numpy_2d(tr_trues, fallback_h=tr_preds_arr.shape[1])

        yhat_test = self.y_scaler.inverse_transform(preds_arr)
        ytrue_test = self.y_scaler.inverse_transform(trues_arr)

        yhat_fit = self.y_scaler.inverse_transform(tr_preds_arr)
        ytrue_fit = self.y_scaler.inverse_transform(tr_trues_arr)

        self.base_forecast = yhat_test.astype(np.float64)
        self.test_true = ytrue_test.astype(np.float64)

        # Keep the full (T_fit, H) residual matrix.
        # This is what the cloud needs to estimate W horizon-by-horizon.
        resid = (ytrue_fit - yhat_fit)
        self.residual_samples = resid.astype(np.float64)
        self.y1 = yhat_fit.astype(np.float64)
        self.y2 = ytrue_fit.astype(np.float64)

    def set_P_column(self, P_col: np.ndarray):
        arr = np.asarray(P_col, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, None]
        self.P_col = arr

    def build_weighted_contributions(self) -> np.ndarray:
        """Build this client's outgoing contributions for every recipient.

        Returns
        -------
        contrib : np.ndarray, shape (n_series, T_test, H)
            contrib[i, t, h] = P_h[i, self] * base_forecast[t, h]
        """
        if self.P_col is None:
            raise ValueError("P_col has not been assigned to this client.")
        if self.base_forecast is None:
            raise ValueError("base_forecast has not been computed for this client.")

        base = np.asarray(self.base_forecast, dtype=np.float64)
        if base.ndim == 1:
            base = base[:, None]

        P_col = np.asarray(self.P_col, dtype=np.float64)
        if P_col.ndim == 1:
            P_col = P_col[:, None]

        if P_col.shape[1] == 1 and base.shape[1] > 1:
            P_col = np.repeat(P_col, base.shape[1], axis=1)
        if P_col.shape[1] != base.shape[1]:
            raise ValueError(
                f"P_col horizon dimension {P_col.shape[1]} does not match base forecast horizon {base.shape[1]}"
            )

        return P_col[:, None, :] * base[None, :, :]

    def reset_reconciled_forecasts(self):
        self.reconciled_forecast = None
        self.reconciled_forecasts = {}

    def add_reconciled_contribution(self, contrib: np.ndarray, method: str | None = None):
        contrib = np.asarray(contrib, dtype=np.float64)
        if method is None:
            if self.reconciled_forecast is None:
                self.reconciled_forecast = contrib.copy()
            else:
                self.reconciled_forecast += contrib
            return

        key = str(method)
        if key not in self.reconciled_forecasts:
            self.reconciled_forecasts[key] = contrib.copy()
        else:
            self.reconciled_forecasts[key] += contrib

        if len(self.reconciled_forecasts) == 1:
            self.reconciled_forecast = self.reconciled_forecasts[key]
