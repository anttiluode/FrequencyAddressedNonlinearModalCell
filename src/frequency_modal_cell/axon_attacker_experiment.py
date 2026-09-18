from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .axon import ephaptic_contacts, grow_branching_arbor, terminal_arrival_times
from .standard_attacker import FixedDelayGraph, max_arrival_error, rms_arrival_error


def _make_bundle(seed: int = 23, separation: float = 0.055):
    arbors = []
    base_targets = np.array([[1.15, -0.18], [1.25, 0.0], [1.15, 0.18]], dtype=float)
    for i, y in enumerate((-separation, 0.0, separation)):
        soma = np.array([0.0, y])
        targets = base_targets + np.array([0.0, y])
        arbors.append(grow_branching_arbor(soma, targets, arbor_id=i, seed=seed + i))
    return arbors


def _rounded(x: float) -> float:
    return float(np.round(x, 12))


def run_v2(seed: int = 23, omega: float = 0.42) -> dict:
    near = _make_bundle(seed=seed, separation=0.055)
    far = _make_bundle(seed=seed, separation=0.34)
    sync = {0: 0.0, 1: 0.35, 2: 0.70}
    staggered = {0: 0.0, 1: 12.0, 2: 24.0}

    near_contacts = ephaptic_contacts(near)
    far_contacts = ephaptic_contacts(far)
    near_graph = FixedDelayGraph.from_arbors(near)
    far_graph = FixedDelayGraph.from_arbors(far)

    near_uncoupled = terminal_arrival_times(near, sync, ephaptic_gain=0.0, contacts=near_contacts)
    near_coupled = terminal_arrival_times(near, sync, ephaptic_gain=0.006, contacts=near_contacts)
    far_coupled = terminal_arrival_times(far, sync, ephaptic_gain=0.006, contacts=far_contacts)
    staggered_coupled = terminal_arrival_times(near, staggered, ephaptic_gain=0.006, contacts=near_contacts)

    graph_near_sync = near_graph.arrivals(sync)
    graph_far_sync = far_graph.arrivals(sync)
    graph_near_staggered = near_graph.arrivals(staggered)

    uncoupled_max = _rounded(max_arrival_error(near_uncoupled, graph_near_sync))
    near_err = _rounded(rms_arrival_error(near_coupled, graph_near_sync))
    far_err = _rounded(rms_arrival_error(far_coupled, graph_far_sync))
    staggered_err = _rounded(rms_arrival_error(staggered_coupled, graph_near_staggered))
    phase_err = _rounded(omega * near_err)

    gate = {
        "fixed_graph_exact_when_uncoupled": bool(uncoupled_max < 1e-10),
        "near_coactivity_breaks_fixed_delay": bool(near_err > 1e-3),
        "spatial_control_suppresses_80pct": bool(far_err <= 0.2 * near_err),
        "temporal_control_suppresses_80pct": bool(staggered_err <= 0.2 * near_err),
    }

    return {
        "experiment": "v2_fixed_delay_graph_attacker",
        "seed": seed,
        "attacker": {
            "name": "sparse_fixed_delay_graph",
            "edge_count": len(near_graph.delays),
            "description": "one sparse edge per grown terminal, carrying only its frozen path delay",
        },
        "metrics": {
            "uncoupled_max_abs_arrival_error": uncoupled_max,
            "near_sync_coupled_rms_arrival_error": near_err,
            "far_sync_coupled_rms_arrival_error": far_err,
            "near_staggered_coupled_rms_arrival_error": staggered_err,
            "near_sync_phase_rms_error_rad": phase_err,
        },
        "gate": gate,
        "verdict": "PASS_FIXED_DELAY_BOUNDARY" if all(gate.values()) else "FAIL_FIXED_DELAY_BOUNDARY",
        "interpretation": (
            "Frozen branching geometry earns no new primitive by itself: it collapses exactly to a sparse graph with fixed delays. "
            "The residue in this synthetic model is the context-dependent timing perturbation caused by nearby co-active axonal matter. "
            "A richer conventional graph with activity-dependent delays could emulate that too, so the claim is decomposition, not irreducibility."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/v2_fixed_delay_attacker.json"))
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--omega", type=float, default=0.42)
    args = parser.parse_args()
    receipt = run_v2(seed=args.seed, omega=args.omega)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
