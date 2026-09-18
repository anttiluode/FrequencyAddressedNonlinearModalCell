from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .axon import arrival_phase_shift, ephaptic_contacts, grow_branching_arbor, terminal_arrival_times


def _make_bundle(seed: int = 23, separation: float = 0.06):
    arbors = []
    base_targets = np.array([[1.15, -0.18], [1.25, 0.0], [1.15, 0.18]], dtype=float)
    for i, y in enumerate((-separation, 0.0, separation)):
        soma = np.array([0.0, y])
        targets = base_targets + np.array([0.0, y])
        arbors.append(grow_branching_arbor(soma, targets, arbor_id=i, seed=seed + i))
    return arbors


def _mean_delay_shift(coupled, uncoupled):
    return float(np.round(np.mean([uncoupled[k] - coupled[k] for k in sorted(coupled)]), 12))


def run_v1(seed: int = 23, omega: float = 0.42) -> dict:
    near = _make_bundle(seed=seed, separation=0.055)
    far = _make_bundle(seed=seed, separation=0.34)
    sync = {0: 0.0, 1: 0.35, 2: 0.70}
    staggered = {0: 0.0, 1: 12.0, 2: 24.0}

    near_contacts = ephaptic_contacts(near)
    far_contacts = ephaptic_contacts(far)
    uncoupled = terminal_arrival_times(near, sync, ephaptic_gain=0.0, contacts=near_contacts)
    coupled = terminal_arrival_times(near, sync, ephaptic_gain=0.006, contacts=near_contacts)
    far_uncoupled = terminal_arrival_times(far, sync, ephaptic_gain=0.0, contacts=far_contacts)
    far_coupled = terminal_arrival_times(far, sync, ephaptic_gain=0.006, contacts=far_contacts)
    staggered_uncoupled = terminal_arrival_times(near, staggered, ephaptic_gain=0.0, contacts=near_contacts)
    staggered_coupled = terminal_arrival_times(near, staggered, ephaptic_gain=0.006, contacts=near_contacts)

    near_shift = _mean_delay_shift(coupled, uncoupled)
    far_shift = _mean_delay_shift(far_coupled, far_uncoupled)
    staggered_shift = _mean_delay_shift(staggered_coupled, staggered_uncoupled)
    phase = arrival_phase_shift(coupled, uncoupled, omega=omega)
    mean_phase_shift = float(np.round(np.mean(np.abs(list(phase.values()))), 12))
    mean_delay = float(np.mean([uncoupled[k] - sync[k[0]] for k in uncoupled]))
    shift_fraction = near_shift / mean_delay
    all_target_errors = np.concatenate([a.target_errors for a in near])
    branch_counts = [a.branch_count for a in near]

    gate = {
        "all_targets_reached": bool(np.max(all_target_errors) < 1e-8),
        "all_arbors_branch": bool(min(branch_counts) >= 1),
        "near_sync_ephaptic_effect_nonzero": bool(near_shift > 1e-4),
        "weak_not_hidden_wire": bool(0.0 < shift_fraction < 0.05),
        "spatial_control_suppresses_80pct": bool(far_shift <= 0.2 * near_shift),
        "temporal_control_suppresses_80pct": bool(staggered_shift <= 0.2 * near_shift),
        "zero_gain_exactly_uncoupled": bool(uncoupled == terminal_arrival_times(near, sync, ephaptic_gain=0.0, contacts={})),
    }
    verdict = "PASS_AXONAL_MATTER_GATE" if all(gate.values()) else "FAIL_AXONAL_MATTER_GATE"
    return {
        "experiment": "v1_branching_axonal_matter",
        "seed": seed,
        "geometry": {
            "arbors": len(near),
            "targets_per_arbor": len(near[0].targets),
            "branch_counts": branch_counts,
            "segment_counts": [len(a.segments) for a in near],
            "mean_total_length": float(np.round(np.mean([a.total_length for a in near]), 12)),
            "max_target_error": float(np.max(all_target_errors)),
            "near_ephaptic_contacts": len(near_contacts),
            "far_ephaptic_contacts": len(far_contacts),
        },
        "ephaptic": {
            "model": "weak geometry- and coactivity-dependent conduction-delay perturbation",
            "gain": 0.006,
            "near_sync_mean_delay_shift": near_shift,
            "far_sync_mean_delay_shift": far_shift,
            "near_staggered_mean_delay_shift": staggered_shift,
            "near_sync_delay_shift_fraction": shift_fraction,
            "carrier_omega_rad_per_step": omega,
            "mean_absolute_arrival_phase_shift_rad": mean_phase_shift,
        },
        "gate": gate,
        "verdict": verdict,
        "claim_boundary": (
            "This is a synthetic axonal-matter gate. Branch growth is chemoaffinity-like and ephaptic coupling is modeled only as a weak local timing perturbation between nearby co-active segments. "
            "It does not claim calibrated mammalian conduction velocities, field amplitudes, or that ephaptic coupling is a major source of neural computation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/v1_axonal_matter.json"))
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--omega", type=float, default=0.42)
    args = parser.parse_args()
    receipt = run_v1(seed=args.seed, omega=args.omega)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
