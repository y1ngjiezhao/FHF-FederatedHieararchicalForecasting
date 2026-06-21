from __future__ import annotations

from typing import Dict, Sequence
import numpy as np

from FHF_Cloud import Cloud as ReconEngine
from FHF_models import build_model
from HFL_servers import HFLSeriesNode


class ReconciliationCoordinator:
    def __init__(self, args, node_ids: Sequence[int], cid2series: Dict[int, str]):
        dummy_model = build_model(args).to(args.device)
        self.node_ids = [int(x) for x in node_ids]
        self.cid2series = {int(k): str(v) for k, v in cid2series.items()}
        self.engine = ReconEngine(init_model=dummy_model, client_ids=self.node_ids)
        self.engine.build_hierarchy_S(self.cid2series)

    @property
    def S(self):
        return self.engine.S

    @property
    def Gs(self):
        return self.engine.Gs

    @property
    def Ps(self):
        return self.engine.Ps

    @property
    def Ws_by_method(self):
        return self.engine.Ws_by_method

    def normalize_recon_methods(self, methods):
        return self.engine.normalize_recon_methods(methods)

    def compute_reconciliation_matrices(self, methods, residuals_by_node: Dict[int, np.ndarray], ridge: float = 1e-6, td_eps: float = 1e-8):
        return self.engine.compute_reconciliation_matrices(methods=methods, resid_by_cid=residuals_by_node, ridge=ridge, td_eps=td_eps)

    def distribute_columns(self, nodes: Dict[int, HFLSeriesNode], method: str):
        P_tensor = self.engine.Ps[str(method)]
        W_tensor = self.engine.Ws_by_method[str(method)]
        for j, nid in enumerate(self.node_ids):
            P_col = P_tensor[:, :, j].T
            W_col = W_tensor[:, :, j].T
            nodes[nid].set_P_column(P_col)
            nodes[nid].set_W_column(W_col, method=str(method))


class PeerNetwork:
    def __init__(self, node_ids: Sequence[int]):
        self.node_ids = [int(x) for x in node_ids]

    def distributed_reconcile(self, nodes: Dict[int, HFLSeriesNode], coordinator: ReconciliationCoordinator, methods: Sequence[str], validate: bool = True, atol: float = 1e-7):
        for nid in self.node_ids:
            nodes[nid].reset_reconciled_forecasts()

        test_lengths = []
        horizon_lengths = []
        for nid in self.node_ids:
            base = nodes[nid].base_forecast
            if base is None:
                raise ValueError(f'Node {nid} has no base forecast. Run compute_base_forecast_and_residuals first.')
            base = np.asarray(base, dtype=np.float64)
            if base.ndim == 1:
                base = base[:, None]
            test_lengths.append(base.shape[0])
            horizon_lengths.append(base.shape[1])
        if len(set(test_lengths)) != 1:
            raise ValueError(f'All nodes must share the same test length for distributed reconciliation. Got {sorted(set(test_lengths))}')
        if len(set(horizon_lengths)) != 1:
            raise ValueError(f'All nodes must share the same forecast horizon. Got {sorted(set(horizon_lengths))}')

        for method in methods:
            coordinator.distribute_columns(nodes, method=method)
            for sender_id in self.node_ids:
                outgoing = nodes[sender_id].build_weighted_contributions()
                for i, recipient_id in enumerate(self.node_ids):
                    nodes[recipient_id].add_reconciled_contribution(outgoing[i], method=str(method))

            if validate:
                P_tensor = coordinator.Ps[str(method)]
                base_stack = np.stack([np.asarray(nodes[nid].base_forecast, dtype=np.float64) for nid in self.node_ids], axis=0)
                rec_central = np.empty_like(base_stack)
                for h in range(base_stack.shape[2]):
                    rec_central[:, :, h] = P_tensor[h] @ base_stack[:, :, h]
                for i, nid in enumerate(self.node_ids):
                    diff = np.max(np.abs(nodes[nid].reconciled_forecasts[str(method)] - rec_central[i]))
                    if diff > float(atol):
                        raise AssertionError(f'P2P reconciliation mismatch for method={method}, node={nid}, max_diff={diff}, atol={atol}')
