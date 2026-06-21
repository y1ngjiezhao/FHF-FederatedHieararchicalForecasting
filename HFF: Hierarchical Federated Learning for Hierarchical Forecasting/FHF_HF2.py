from __future__ import annotations

"""
Compatibility entry point for modular HFL.

This file preserves the familiar public API:
    - HFLArgs
    - set_seed
    - build_hfl_nodes
    - run_hfl_with_third_party_reconciliation

Internally it delegates to the modular implementation in:
    - HFL_topology.py
    - HFL_servers.py
    - HFL_recon.py
    - HFL_runner_timed.py

So existing notebooks can keep using:
    from FHF_HFL_modular import HFLArgs, run_hfl_with_third_party_reconciliation

or you can rename this file to FHF_HFL.py to fully replace the old monolithic version.
"""

from HFL_runner_timed import HFLArgs, set_seed, build_hfl_nodes, run_hfl_timed


def run_hfl_with_third_party_reconciliation(args: HFLArgs):
    """Backward-compatible name for the modular/timed HFL runner."""
    return run_hfl_timed(args)


__all__ = [
    "HFLArgs",
    "set_seed",
    "build_hfl_nodes",
    "run_hfl_with_third_party_reconciliation",
]
