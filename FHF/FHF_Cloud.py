from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch.nn as nn


class Cloud:
    """Server/Cloud: FedAvg + compute reconciliation matrices P_h = S G_h."""

    def __init__(self, init_model: nn.Module, client_ids: List[int]):
        self.global_sd = {k: v.detach().cpu().clone() for k, v in init_model.state_dict().items()}
        self.client_ids = list(map(int, client_ids))

        self.all_nodes: List[str] = []
        self.bottoms: List[str] = []
        self.S: Optional[np.ndarray] = None

        self.W: Optional[np.ndarray] = None
        self.Ws: Optional[np.ndarray] = None          # (H, n_series, n_series)
        self.G: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self.Gs: Dict[str, np.ndarray] = {}
        self.Ps: Dict[str, np.ndarray] = {}
        self.Ws_by_method: Dict[str, np.ndarray] = {}

    def build_hierarchy_S(self, cid2series: Dict[int, str]):
        all_nodes = [str(cid2series[int(cid)]) for cid in self.client_ids]
        bottoms = []
        for n in all_nodes:
            has_child = any((m != n) and m.startswith(n + "/") for m in all_nodes)
            if not has_child:
                bottoms.append(n)

        n_all = len(all_nodes)
        n_bot = len(bottoms)
        S = np.zeros((n_all, n_bot), dtype=np.float64)
        for i, node in enumerate(all_nodes):
            for j, b in enumerate(bottoms):
                if (b == node) or b.startswith(node + "/"):
                    S[i, j] = 1.0

        self.all_nodes = all_nodes
        self.bottoms = bottoms
        self.S = S
        return S, all_nodes, bottoms

    def _coerce_residual_matrix(self, residuals: np.ndarray) -> np.ndarray:
        arr = np.asarray(residuals, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:
            raise ValueError(f"Residual array must be 1D or 2D, got shape {arr.shape}")
        return arr

    def _prepare_residual_cube(self, resid_by_cid: Dict[int, np.ndarray]) -> np.ndarray:
        mats = []
        lengths = []
        horizons = []
        for cid in self.client_ids:
            if int(cid) not in resid_by_cid:
                raise KeyError(f"Missing residuals for client {cid}")
            arr = self._coerce_residual_matrix(resid_by_cid[int(cid)])
            mats.append(arr)
            lengths.append(arr.shape[0])
            horizons.append(arr.shape[1])

        if not mats:
            raise ValueError("No residuals received from clients.")
        if len(set(horizons)) != 1:
            raise ValueError(f"All clients must share the same residual horizon. Got {sorted(set(horizons))}")

        T_eff = int(min(lengths))
        H = int(horizons[0])
        if T_eff <= 0:
            raise ValueError("Residual length must be positive for all clients.")

        cube = np.stack([m[-T_eff:, :] for m in mats], axis=0).astype(np.float64)  # (n_series, T_eff, H)
        return cube

    def _cov_from_residual_matrix(self, residuals_h: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
        """residuals_h shape: (T_eff, n_series), matching HierarchicalForecast's residual-matrix convention."""
        T_eff, n_series = residuals_h.shape
        if T_eff <= 1:
            return np.eye(n_series, dtype=np.float64)
        W = np.cov(residuals_h, rowvar=False)
        W = np.asarray(W, dtype=np.float64)
        W = W + float(ridge) * np.eye(n_series, dtype=np.float64)
        return W

    def _ledoit_cov_from_residual_matrix(self, residuals_h: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
        from sklearn.covariance import LedoitWolf

        T_eff, n_series = residuals_h.shape
        if T_eff <= 1:
            return np.eye(n_series, dtype=np.float64)
        W = LedoitWolf().fit(residuals_h).covariance_
        W = np.asarray(W, dtype=np.float64)
        W = W + float(ridge) * np.eye(n_series, dtype=np.float64)
        return W

    def compute_W_matrices_from_residual_samples(self, resid_by_cid: Dict[int, np.ndarray], ridge: float = 1e-6) -> np.ndarray:
        """Estimate one covariance matrix per horizon.

        Adaptation of HierarchicalForecast's MinTrace residual handling:
        for each horizon h, use a residual matrix with shape (obs, n_series)
        and estimate W_h from that matrix.
        """
        cube = self._prepare_residual_cube(resid_by_cid)  # (n_series, T_eff, H)
        _, _, H = cube.shape
        Ws = []
        for h in range(H):
            residuals_h = cube[:, :, h].T  # (T_eff, n_series)
            Ws.append(self._cov_from_residual_matrix(residuals_h, ridge=ridge))
        Ws = np.stack(Ws, axis=0)
        self.Ws = Ws
        self.W = Ws[0] if Ws.shape[0] == 1 else None
        return Ws

    def _mint_G_from_W(self, W: np.ndarray) -> np.ndarray:
        S = self.S
        if S is None:
            raise ValueError("S not built. Call build_hierarchy_S first.")
        W_inv = np.linalg.pinv(W)
        A = S.T @ W_inv @ S
        A_inv = np.linalg.pinv(A)
        G = A_inv @ S.T @ W_inv
        return G

    def _bottom_selection_G(self) -> np.ndarray:
        node2i = {n: i for i, n in enumerate(self.all_nodes)}
        J = np.zeros((len(self.bottoms), len(self.all_nodes)), dtype=np.float64)
        for j, b in enumerate(self.bottoms):
            J[j, node2i[b]] = 1.0
        return J

    def _canonical_method(self, method: str) -> str:
        m = str(method).strip().lower().replace('-', '_')
        alias = {
            'bu': 'bu',
            'bottom_up': 'bu',
            'bottomup': 'bu',
            'bottom': 'bu',
            'mint': 'mint_cov',
            'mint_cov': 'mint_cov',
            'mint_var': 'mint_var',
            'mint_diag': 'mint_var',
            'mint_ols': 'mint_ols',
            'ols': 'mint_ols',
            'mint_shrinkage': 'mint_shrinkage',
            'mint_shr': 'mint_shrinkage',
            'wls_structure': 'wls_structure',
            'mint_structure': 'wls_structure',
            'structure': 'wls_structure',
        }
        if m not in alias:
            raise ValueError(f"Unknown recon_method: {method}")
        return alias[m]

    def normalize_recon_methods(self, methods):
        if isinstance(methods, str):
            methods = [methods]
        out = []
        for method in methods:
            key = self._canonical_method(method)
            if key not in out:
                out.append(key)
        return out

    def _compute_G_for_method(self, method: str, W_base: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
        method = self._canonical_method(method)
        n_all = len(self.client_ids)

        if method == 'mint_cov':
            G = self._mint_G_from_W(W_base)
        elif method == 'mint_var':
            Wd = np.diag(np.diag(W_base))
            G = self._mint_G_from_W(Wd)
        elif method == 'mint_ols':
            G = self._mint_G_from_W(np.eye(n_all, dtype=np.float64))
        elif method == 'wls_structure':
            J = self._bottom_selection_G()
            var_bottom = np.diag(J @ W_base @ J.T)
            Wstr = self.S @ np.diag(var_bottom) @ self.S.T
            Wstr = Wstr + float(ridge) * np.eye(n_all, dtype=np.float64)
            G = self._mint_G_from_W(Wstr)
        elif method == 'mint_shrinkage':
            G = self._mint_G_from_W(W_base)
        elif method == 'bu':
            G = self._bottom_selection_G()
        else:
            raise ValueError(f"Unknown canonical method: {method}")
        return G

    def compute_reconciliation_matrices(self, methods, resid_by_cid: Dict[int, np.ndarray], ridge: float = 1e-6, td_eps: float = 1e-8):
        if self.S is None:
            raise ValueError("S not built. Call build_hierarchy_S first.")

        methods = self.normalize_recon_methods(methods)
        residual_cube = self._prepare_residual_cube(resid_by_cid)   # (n_series, T_eff, H)
        _, _, H = residual_cube.shape
        n_all = len(self.client_ids)

        # Plain covariance matrices, one per horizon.
        W_covs = self.compute_W_matrices_from_residual_samples(resid_by_cid, ridge=ridge)

        Ps: Dict[str, np.ndarray] = {}
        Gs: Dict[str, np.ndarray] = {}
        Ws_by_method: Dict[str, np.ndarray] = {}

        for method in methods:
            P_list = []
            G_list = []
            W_list = []

            for h in range(H):
                residuals_h = residual_cube[:, :, h].T  # (T_eff, n_series)

                if method == 'mint_shrinkage':
                    W_h = self._ledoit_cov_from_residual_matrix(residuals_h, ridge=ridge)
                else:
                    W_h = W_covs[h]

                G_h = self._compute_G_for_method(method=method, W_base=W_h, ridge=ridge)
                P_h = self.S @ G_h

                P_list.append(P_h)
                G_list.append(G_h)
                W_list.append(W_h)

            Ps[method] = np.stack(P_list, axis=0)   # (H, n_all, n_all)
            Gs[method] = np.stack(G_list, axis=0)   # (H, n_bottom, n_all)
            Ws_by_method[method] = np.stack(W_list, axis=0)

        self.Ws_by_method = Ws_by_method
        self.Gs = Gs
        self.Ps = Ps
        if len(methods) == 1:
            only = methods[0]
            self.Ws = Ws_by_method[only]
            self.W = self.Ws[0] if self.Ws.shape[0] == 1 else None
            self.G = Gs[only][0] if Gs[only].shape[0] == 1 else None
            self.P = Ps[only][0] if Ps[only].shape[0] == 1 else None
        return Ps, Gs

    def compute_reconciliation_matrix(self, method: str, resid_by_cid: Dict[int, np.ndarray], ridge: float = 1e-6, td_eps: float = 1e-8):
        methods = self.normalize_recon_methods(method)
        Ps, Gs = self.compute_reconciliation_matrices(methods, resid_by_cid=resid_by_cid, ridge=ridge, td_eps=td_eps)
        only = methods[0]
        self.P = Ps[only][0] if Ps[only].shape[0] == 1 else None
        self.G = Gs[only][0] if Gs[only].shape[0] == 1 else None
        return Ps[only], Gs[only]
