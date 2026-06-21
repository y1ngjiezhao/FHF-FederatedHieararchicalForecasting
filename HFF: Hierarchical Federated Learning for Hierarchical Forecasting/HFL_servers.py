from __future__ import annotations

from typing import Dict, List, Optional, Sequence
import numpy as np
import torch

from FHF_Client import Client as BaseClient
from aggregation import fedavg_aggregate


VALID_HIER_TRAIN_MODES = {"post_agg_local_finetune", "self_train_then_aggregate"}


def clone_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in sd.items()}


def _safe_average(values) -> float:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _evaluate_node_losses(node: BaseClient) -> Dict[str, float]:
    """Evaluate a node's train loss on normalised and original scales."""
    if hasattr(node, "evaluate_train_losses"):
        losses = node.evaluate_train_losses()
        normalised = losses.get("normalised_loss", np.nan)
        actual = losses.get("actual_loss", np.nan)
        return {
            "normalised_loss": float(normalised) if np.isfinite(float(normalised)) else float("nan"),
            "actual_loss": float(actual) if np.isfinite(float(actual)) else float("nan"),
        }
    return {"normalised_loss": float("nan"), "actual_loss": float("nan")}


class HFLSeriesNode(BaseClient):
    def __init__(self, cid: int, series_name: str, df_one, args, role: str, parent_id: Optional[int], trainable: bool):
        super().__init__(cid=cid, series_name=series_name, df_one=df_one, args=args)
        self.role = str(role)
        self.parent_id = None if parent_id is None else int(parent_id)
        self.child_ids: List[int] = []
        self.trainable = bool(trainable)
        self.W_columns: Dict[str, np.ndarray] = {}

    def set_W_column(self, W_col: np.ndarray, method: str):
        arr = np.asarray(W_col, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, None]
        self.W_columns[str(method)] = arr


class EdgeServer:
    def __init__(self, edge_node_id: int, child_client_ids: Sequence[int], init_sd: Dict[str, torch.Tensor]):
        self.edge_node_id = int(edge_node_id)
        self.child_client_ids = [int(x) for x in child_client_ids]
        self.edge_sd = clone_state_dict(init_sd)
        self.last_upload_weight: int = max(1, len(self.child_client_ids))
        self.last_train_losses: Dict[int, float] = {}
        self.last_selected_client_ids: List[int] = []

    def sync_from_cloud(self, global_sd: Dict[str, torch.Tensor]):
        self.edge_sd = clone_state_dict(global_sd)

    def run_one_edge_round(self, nodes: Dict[int, HFLSeriesNode], client_frac: float, rng: np.random.Generator, hier_train_mode: str = "post_agg_local_finetune"):
        hier_train_mode = str(hier_train_mode)
        if hier_train_mode not in VALID_HIER_TRAIN_MODES:
            raise ValueError(f"Unsupported hier_train_mode={hier_train_mode}. Valid options: {sorted(VALID_HIER_TRAIN_MODES)}")
        if not self.child_client_ids:
            raise ValueError(f'Edge {self.edge_node_id} has no clients.')

        selected_num = max(1, int(round(len(self.child_client_ids) * float(client_frac))))
        if selected_num >= len(self.child_client_ids):
            selected = list(self.child_client_ids)
        else:
            selected = list(map(int, rng.choice(self.child_client_ids, size=selected_num, replace=False).tolist()))

        state_dicts: List[Dict[str, torch.Tensor]] = []
        weights: List[int] = []
        train_normalised_losses: Dict[int, float] = {}
        train_actual_losses: Dict[int, float] = {}
        for cid in selected:
            node = nodes[cid]
            node.set_state_dict(self.edge_sd)
            node.local_train()
            loss_info = _evaluate_node_losses(node)
            train_normalised_losses[int(cid)] = float(loss_info.get("normalised_loss", float("nan")))
            train_actual_losses[int(cid)] = float(loss_info.get("actual_loss", float("nan")))
            state_dicts.append(node.get_state_dict())
            weights.append(max(1, node.num_train_samples()))

        child_agg_sd = fedavg_aggregate(state_dicts, weights)
        total_child_weight = int(sum(weights))

        edge_node = nodes[self.edge_node_id]
        if edge_node.trainable:
            if hier_train_mode == "post_agg_local_finetune":
                edge_node.set_state_dict(child_agg_sd)
                edge_node.local_train()
                edge_loss_info = _evaluate_node_losses(edge_node)
                train_normalised_losses[int(self.edge_node_id)] = float(edge_loss_info.get("normalised_loss", float("nan")))
                train_actual_losses[int(self.edge_node_id)] = float(edge_loss_info.get("actual_loss", float("nan")))
                edge_local_sd = edge_node.get_state_dict()
                edge_weight = max(1, edge_node.num_train_samples())
                self.edge_sd = fedavg_aggregate([child_agg_sd, edge_local_sd], [max(1, total_child_weight), edge_weight])
                self.last_upload_weight = max(1, total_child_weight + edge_weight)
            else:
                # New mode: edge trains its own local model independently from the incoming shared model,
                # then participates in the same aggregation pool as its child clients.
                edge_node.set_state_dict(self.edge_sd)
                edge_node.local_train()
                edge_loss_info = _evaluate_node_losses(edge_node)
                train_normalised_losses[int(self.edge_node_id)] = float(edge_loss_info.get("normalised_loss", float("nan")))
                train_actual_losses[int(self.edge_node_id)] = float(edge_loss_info.get("actual_loss", float("nan")))
                edge_local_sd = edge_node.get_state_dict()
                edge_weight = max(1, edge_node.num_train_samples())
                self.edge_sd = fedavg_aggregate(state_dicts + [edge_local_sd], weights + [edge_weight])
                self.last_upload_weight = max(1, total_child_weight + edge_weight)
        else:
            self.edge_sd = clone_state_dict(child_agg_sd)
            self.last_upload_weight = max(1, total_child_weight)

        edge_node.set_state_dict(self.edge_sd)
        # Backward-compatible alias keeps the old name as the normalised loss.
        train_losses = dict(train_normalised_losses)
        self.last_train_losses = train_losses
        self.last_selected_client_ids = list(selected)

        return {
            "edge_id": int(self.edge_node_id),
            "selected_client_ids": list(selected),
            "train_losses": dict(train_losses),
            "train_normalised_losses": dict(train_normalised_losses),
            "train_actual_losses": dict(train_actual_losses),
            "avg_normalised_loss": _safe_average(train_normalised_losses.values()),
            "avg_actual_loss": _safe_average(train_actual_losses.values()),
            # Backward-compatible alias.
            "avg_train_loss": _safe_average(train_normalised_losses.values()),
            "num_train_events": len(train_normalised_losses),
        }


class CloudServer:
    def __init__(self, cloud_node_id: int, init_sd: Dict[str, torch.Tensor]):
        self.cloud_node_id = int(cloud_node_id)
        self.global_sd = clone_state_dict(init_sd)
        self.last_upload_weight: int = 1
        self.last_train_losses: Dict[int, float] = {}

    def aggregate_edges(self, edge_servers: Dict[int, EdgeServer], nodes: Dict[int, HFLSeriesNode], hier_train_mode: str = "post_agg_local_finetune"):
        hier_train_mode = str(hier_train_mode)
        if hier_train_mode not in VALID_HIER_TRAIN_MODES:
            raise ValueError(f"Unsupported hier_train_mode={hier_train_mode}. Valid options: {sorted(VALID_HIER_TRAIN_MODES)}")

        train_normalised_losses: Dict[int, float] = {}
        train_actual_losses: Dict[int, float] = {}
        state_dicts = [edge_servers[eid].edge_sd for eid in edge_servers]
        weights = [max(1, edge_servers[eid].last_upload_weight) for eid in edge_servers]
        tmp_global_sd = fedavg_aggregate(state_dicts, weights)
        total_edge_weight = int(sum(weights))

        cloud_node = nodes[self.cloud_node_id]
        if cloud_node.trainable:
            if hier_train_mode == "post_agg_local_finetune":
                cloud_node.set_state_dict(tmp_global_sd)
                cloud_node.local_train()
                cloud_loss_info = _evaluate_node_losses(cloud_node)
                train_normalised_losses[int(self.cloud_node_id)] = float(cloud_loss_info.get("normalised_loss", float("nan")))
                train_actual_losses[int(self.cloud_node_id)] = float(cloud_loss_info.get("actual_loss", float("nan")))
                cloud_local_sd = cloud_node.get_state_dict()
                cloud_weight = max(1, cloud_node.num_train_samples())
                self.global_sd = fedavg_aggregate([tmp_global_sd, cloud_local_sd], [max(1, total_edge_weight), cloud_weight])
                self.last_upload_weight = max(1, total_edge_weight + cloud_weight)
            else:
                # New mode: cloud trains its own total-series model independently from the incoming
                # shared model before aggregating with the edge uploads.
                cloud_node.set_state_dict(self.global_sd)
                cloud_node.local_train()
                cloud_loss_info = _evaluate_node_losses(cloud_node)
                train_normalised_losses[int(self.cloud_node_id)] = float(cloud_loss_info.get("normalised_loss", float("nan")))
                train_actual_losses[int(self.cloud_node_id)] = float(cloud_loss_info.get("actual_loss", float("nan")))
                cloud_local_sd = cloud_node.get_state_dict()
                cloud_weight = max(1, cloud_node.num_train_samples())
                self.global_sd = fedavg_aggregate(state_dicts + [cloud_local_sd], weights + [cloud_weight])
                self.last_upload_weight = max(1, total_edge_weight + cloud_weight)
        else:
            self.global_sd = clone_state_dict(tmp_global_sd)
            self.last_upload_weight = max(1, total_edge_weight)

        cloud_node.set_state_dict(self.global_sd)
        # Backward-compatible alias keeps the old name as the normalised loss.
        train_losses = dict(train_normalised_losses)
        self.last_train_losses = train_losses

        return {
            "cloud_id": int(self.cloud_node_id),
            "train_losses": dict(train_losses),
            "train_normalised_losses": dict(train_normalised_losses),
            "train_actual_losses": dict(train_actual_losses),
            "avg_normalised_loss": _safe_average(train_normalised_losses.values()),
            "avg_actual_loss": _safe_average(train_actual_losses.values()),
            # Backward-compatible alias.
            "avg_train_loss": _safe_average(train_normalised_losses.values()),
            "num_train_events": len(train_normalised_losses),
        }
