from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union


def parent_path(path: str) -> Optional[str]:
    return path.rsplit('/', 1)[0] if '/' in path else None


def path_depth(path: str) -> int:
    return path.count('/')


@dataclass
class HierarchyLayout:
    node_ids: List[int]
    client_ids: List[int]
    edge_ids: List[int]
    cloud_id: int
    path_by_id: Dict[int, str]
    role_by_id: Dict[int, str]
    parent_by_id: Dict[int, Optional[int]]
    children_by_id: Dict[int, List[int]]
    edge_to_clients: Dict[int, List[int]]



def infer_three_level_layout(node_ids: Sequence[int], cid2series: Dict[int, str]) -> HierarchyLayout:
    node_ids = [int(x) for x in node_ids]
    path_by_id = {int(cid): str(cid2series[int(cid)]) for cid in node_ids}
    depths = {cid: path_depth(path_by_id[cid]) for cid in node_ids}
    uniq_depths = sorted(set(depths.values()))

    if len(uniq_depths) != 3:
        raise ValueError(
            'This HFL implementation assumes exactly 3 hierarchy levels '
            f'(cloud/edge/client). Got depths={uniq_depths}.'
        )

    cloud_depth, edge_depth, client_depth = uniq_depths
    cloud_ids = [cid for cid in node_ids if depths[cid] == cloud_depth]
    edge_ids = [cid for cid in node_ids if depths[cid] == edge_depth]
    client_ids = [cid for cid in node_ids if depths[cid] == client_depth]

    if len(cloud_ids) != 1:
        raise ValueError(f'Expected exactly one cloud node, got {cloud_ids}')

    path_to_id = {path_by_id[cid]: cid for cid in node_ids}
    parent_by_id: Dict[int, Optional[int]] = {}
    children_by_id: Dict[int, List[int]] = {cid: [] for cid in node_ids}

    for cid in node_ids:
        ppath = parent_path(path_by_id[cid])
        pid = path_to_id.get(ppath) if ppath is not None else None
        parent_by_id[cid] = pid
        if pid is not None:
            children_by_id[pid].append(cid)

    role_by_id: Dict[int, str] = {}
    for cid in node_ids:
        d = depths[cid]
        role_by_id[cid] = 'cloud' if d == cloud_depth else ('edge' if d == edge_depth else 'client')

    edge_to_clients = {eid: [cid for cid in children_by_id[eid] if role_by_id[cid] == 'client'] for eid in edge_ids}
    cloud_id = cloud_ids[0]

    for eid in edge_ids:
        if parent_by_id[eid] != cloud_id:
            raise ValueError(
                f'Edge node {eid} ({path_by_id[eid]}) must have cloud node {cloud_id} as parent; '
                f'got parent={parent_by_id[eid]}.')
        if not edge_to_clients[eid]:
            raise ValueError(f'Edge node {eid} ({path_by_id[eid]}) has no client children.')

    for cid in client_ids:
        pid = parent_by_id[cid]
        if pid not in edge_ids:
            raise ValueError(f'Client node {cid} ({path_by_id[cid]}) must have an edge parent; got {pid}.')

    return HierarchyLayout(
        node_ids=node_ids,
        client_ids=client_ids,
        edge_ids=edge_ids,
        cloud_id=cloud_id,
        path_by_id=path_by_id,
        role_by_id=role_by_id,
        parent_by_id=parent_by_id,
        children_by_id=children_by_id,
        edge_to_clients=edge_to_clients,
    )



def resolve_edge_trainable_flags(edge_ids: Sequence[int], spec: Union[bool, Sequence[int], Dict[int, bool]]) -> Dict[int, bool]:
    edge_ids = [int(x) for x in edge_ids]
    if isinstance(spec, bool):
        return {eid: bool(spec) for eid in edge_ids}
    if isinstance(spec, dict):
        return {eid: bool(spec.get(eid, False)) for eid in edge_ids}
    selected = set(int(x) for x in spec)
    return {eid: (eid in selected) for eid in edge_ids}



def resolve_edge_aggregation_rounds(edge_ids: Sequence[int], default_rounds: int, per_edge_spec: Optional[Dict[int, int]] = None) -> Dict[int, int]:
    edge_ids = [int(x) for x in edge_ids]
    rounds = {eid: max(1, int(default_rounds)) for eid in edge_ids}
    if per_edge_spec is None:
        return rounds
    for raw_eid, raw_rounds in per_edge_spec.items():
        eid = int(raw_eid)
        if eid not in rounds:
            raise ValueError(f'edge_agg_rounds_by_edge contains unknown edge id {eid}')
        rounds[eid] = max(1, int(raw_rounds))
    return rounds
