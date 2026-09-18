from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .active_language import (
    discover_active,
    discover_from_indices,
    discover_geometry_cover,
    discover_random,
)
from .virtual_organoid import StimulusAction, VirtualOrganoid, evaluate_codebook


def _candidates() -> list[StimulusAction]:
    return [
        StimulusAction(electrode=e, omega=omega, phase=phase, amplitude=0.12, duration=72)
        for e in range(8)
        for omega in (0.16, 0.24, 0.32, 0.40, 0.48, 0.56)
        for phase in (0.0, np.pi / 2.0)
    ]


def _probe() -> StimulusAction:
    return StimulusAction(electrode=0, omega=0.36, phase=0.30, amplitude=0.07, duration=64)


def _metrics(matter, discovery, *, wait_steps: int, probe: StimulusAction, seed: int) -> dict:
    measured = evaluate_codebook(
        matter,
        discovery.codebook,
        wait_steps=wait_steps,
        probe=probe,
        repeats=4,
        observation_noise=2e-4,
        seed=seed,
    )
    return {
        "accuracy": float(np.round(measured["accuracy"], 12)),
        "min_pair_distance": float(np.round(measured["min_calibration_pair_distance"], 12)),
        "median_margin": float(np.round(measured["median_classification_margin"], 12)),
    }


def _world(world_seed: int, *, budget: int, random_repeats: int, wait_steps: int) -> dict:
    candidates = _candidates()
    probe = _probe()
    codes = 6

    active = discover_active(
        VirtualOrganoid(seed=world_seed),
        candidates,
        wait_steps=wait_steps,
        probe=probe,
        budget=budget,
        codes=codes,
    )
    geometry = discover_geometry_cover(
        VirtualOrganoid(seed=world_seed),
        candidates,
        wait_steps=wait_steps,
        probe=probe,
        budget=budget,
        codes=codes,
    )
    oracle = discover_from_indices(
        VirtualOrganoid(seed=world_seed),
        candidates,
        list(range(len(candidates))),
        wait_steps=wait_steps,
        probe=probe,
        codes=codes,
        strategy="exhaustive_oracle",
    )

    active_metrics = _metrics(
        VirtualOrganoid(seed=world_seed),
        active,
        wait_steps=wait_steps,
        probe=probe,
        seed=1000 + world_seed,
    )
    geometry_metrics = _metrics(
        VirtualOrganoid(seed=world_seed),
        geometry,
        wait_steps=wait_steps,
        probe=probe,
        seed=2000 + world_seed,
    )
    oracle_metrics = _metrics(
        VirtualOrganoid(seed=world_seed),
        oracle,
        wait_steps=wait_steps,
        probe=probe,
        seed=3000 + world_seed,
    )

    random_metrics = []
    random_probe_sets = []
    for repeat in range(random_repeats):
        rng = np.random.default_rng(10000 + 101 * world_seed + repeat)
        discovery = discover_random(
            VirtualOrganoid(seed=world_seed),
            candidates,
            wait_steps=wait_steps,
            probe=probe,
            budget=budget,
            codes=codes,
            rng=rng,
        )
        random_probe_sets.append(list(discovery.probed_indices))
        random_metrics.append(
            _metrics(
                VirtualOrganoid(seed=world_seed),
                discovery,
                wait_steps=wait_steps,
                probe=probe,
                seed=4000 + 37 * world_seed + repeat,
            )
        )

    return {
        "world_seed": world_seed,
        "active": {
            "probed_indices": list(active.probed_indices),
            "metrics": active_metrics,
        },
        "geometry_cover": {
            "probed_indices": list(geometry.probed_indices),
            "metrics": geometry_metrics,
        },
        "exhaustive_oracle": {
            "probe_count": len(candidates),
            "metrics": oracle_metrics,
        },
        "random": {
            "repeats": random_repeats,
            "probe_sets": random_probe_sets,
            "accuracy_median": float(np.round(np.median([m["accuracy"] for m in random_metrics]), 12)),
            "min_pair_distance_median": float(
                np.round(np.median([m["min_pair_distance"] for m in random_metrics]), 12)
            ),
            "accuracy_all": [m["accuracy"] for m in random_metrics],
        },
    }


def run_v4(
    *,
    budget: int = 12,
    random_repeats: int = 5,
    wait_steps: int = 96,
    world_seeds: tuple[int, ...] = (4, 9, 14),
) -> dict:
    worlds = [
        _world(seed, budget=budget, random_repeats=random_repeats, wait_steps=wait_steps)
        for seed in world_seeds
    ]

    active_acc = np.asarray([w["active"]["metrics"]["accuracy"] for w in worlds])
    geom_acc = np.asarray([w["geometry_cover"]["metrics"]["accuracy"] for w in worlds])
    oracle_acc = np.asarray([w["exhaustive_oracle"]["metrics"]["accuracy"] for w in worlds])
    random_acc = np.asarray([w["random"]["accuracy_median"] for w in worlds])

    active_pair = np.asarray([w["active"]["metrics"]["min_pair_distance"] for w in worlds])
    geom_pair = np.asarray([w["geometry_cover"]["metrics"]["min_pair_distance"] for w in worlds])
    random_pair = np.asarray([w["random"]["min_pair_distance_median"] for w in worlds])

    summary = {
        "active_accuracy_median": float(np.round(np.median(active_acc), 12)),
        "geometry_accuracy_median": float(np.round(np.median(geom_acc), 12)),
        "random_accuracy_median": float(np.round(np.median(random_acc), 12)),
        "oracle_accuracy_median": float(np.round(np.median(oracle_acc), 12)),
        "active_min_pair_distance_median": float(np.round(np.median(active_pair), 12)),
        "geometry_min_pair_distance_median": float(np.round(np.median(geom_pair), 12)),
        "random_min_pair_distance_median": float(np.round(np.median(random_pair), 12)),
        "active_world_wins_vs_random_median": int(np.sum(active_acc > random_acc)),
    }

    base_gate = {
        "probe_budget_at_most_one_eighth_of_exhaustive": bool(budget * 8 <= len(_candidates())),
        "active_accuracy_at_least_0_75": bool(summary["active_accuracy_median"] >= 0.75),
        "active_beats_random_accuracy_by_0_05": bool(
            summary["active_accuracy_median"] >= summary["random_accuracy_median"] + 0.05
        ),
        "active_wins_at_least_two_of_three_worlds": bool(
            summary["active_world_wins_vs_random_median"] >= 2
        ),
        "active_within_0_10_of_exhaustive_oracle": bool(
            summary["active_accuracy_median"] >= summary["oracle_accuracy_median"] - 0.10
        ),
        "active_pair_distance_at_least_1_10x_random": bool(
            summary["active_min_pair_distance_median"]
            >= 1.10 * max(summary["random_min_pair_distance_median"], 1e-12)
        ),
    }

    adaptive_specific = bool(
        summary["active_accuracy_median"] >= summary["geometry_accuracy_median"] + 0.03
        or (
            summary["active_accuracy_median"] >= summary["geometry_accuracy_median"] - 1e-12
            and summary["active_min_pair_distance_median"]
            >= 1.10 * max(summary["geometry_min_pair_distance_median"], 1e-12)
        )
    )
    geometry_beats_random = bool(
        summary["geometry_accuracy_median"] >= summary["random_accuracy_median"] + 0.05
        or summary["geometry_min_pair_distance_median"]
        >= 1.10 * max(summary["random_min_pair_distance_median"], 1e-12)
    )

    if all(base_gate.values()) and adaptive_specific:
        verdict = "PASS_RESPONSE_ADAPTIVE_LANGUAGE_DISCOVERY"
    elif all(base_gate.values()) and geometry_beats_random:
        verdict = "PASS_BUDGETED_DESIGN_NOT_ADAPTIVE"
    else:
        verdict = "FAIL_BUDGETED_LANGUAGE_DISCOVERY"

    return {
        "experiment": "v4_budgeted_stimulation_language",
        "candidate_count": len(_candidates()),
        "code_symbols": 6,
        "probe_budget": budget,
        "budget_fraction": float(np.round(budget / len(_candidates()), 12)),
        "random_repeats_per_world": random_repeats,
        "world_seeds": list(world_seeds),
        "controller_access": "stimulation coordinates plus signatures of interventions actually purchased",
        "controller_denied": "unqueried responses, hidden cell/material/connectivity state, and gradients through matter",
        "worlds": worlds,
        "summary": summary,
        "base_gate": base_gate,
        "adaptive_specific": adaptive_specific,
        "geometry_beats_random": geometry_beats_random,
        "verdict": verdict,
        "claim_boundary": (
            "The active selector uses an ordinary ridge response surrogate over visible stimulation coordinates. "
            "Any advantage establishes probe efficiency only in this synthetic family. If the fixed geometry-cover "
            "attacker matches it, response adaptation has not earned a distinct claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/v4_budgeted_language.json"))
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--random-repeats", type=int, default=5)
    args = parser.parse_args()
    receipt = run_v4(budget=args.budget, random_repeats=args.random_repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
