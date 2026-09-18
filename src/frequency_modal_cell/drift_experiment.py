from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .active_language import discover_active, discover_random
from .virtual_organoid import (
    LearnedCodebook,
    StimulusAction,
    VirtualOrganoid,
    _response_after_write,
    evaluate_codebook,
)


@dataclass(frozen=True)
class DriftSpec:
    name: str
    cell_angle_offset: float = 0.12
    recurrent_gain: float = 0.18


def candidates() -> list[StimulusAction]:
    return [
        StimulusAction(electrode=e, omega=omega, phase=phase, amplitude=0.12, duration=72)
        for e in range(8)
        for omega in (0.16, 0.24, 0.32, 0.40, 0.48, 0.56)
        for phase in (0.0, np.pi / 2.0)
    ]


def common_probe() -> StimulusAction:
    return StimulusAction(electrode=0, omega=0.36, phase=0.30, amplitude=0.07, duration=64)


def drift_specs() -> tuple[DriftSpec, ...]:
    # Frozen before seeing v5 outcomes. These are small changes to interface geometry
    # and recurrent strength, not a new random network.
    return (
        DriftSpec("geometry", cell_angle_offset=0.18, recurrent_gain=0.18),
        DriftSpec("dynamics", cell_angle_offset=0.12, recurrent_gain=0.207),
        DriftSpec("combined", cell_angle_offset=0.18, recurrent_gain=0.207),
    )


def make_matter(seed: int, spec: DriftSpec | None = None) -> VirtualOrganoid:
    if spec is None:
        return VirtualOrganoid(seed=seed)
    return VirtualOrganoid(
        seed=seed,
        cell_angle_offset=spec.cell_angle_offset,
        recurrent_gain=spec.recurrent_gain,
    )


def farthest_anchor_pair(signatures: np.ndarray) -> tuple[int, int]:
    signatures = np.asarray(signatures, dtype=float)
    if signatures.shape[0] < 2:
        raise ValueError("at least two signatures required")
    d = np.linalg.norm(signatures[:, None, :] - signatures[None, :, :], axis=2)
    i, j = np.unravel_index(int(np.argmax(np.round(d, 12))), d.shape)
    if i == j:
        raise ValueError("degenerate codebook")
    return (int(min(i, j)), int(max(i, j)))


def fit_two_anchor_alignment(
    old_signatures: np.ndarray,
    new_anchor_signatures: np.ndarray,
    anchor_indices: tuple[int, int],
) -> tuple[float, np.ndarray]:
    """Fit new ~= scalar * old + vector offset using exactly two re-measured codes."""
    old = np.asarray(old_signatures, dtype=float)
    new = np.asarray(new_anchor_signatures, dtype=float)
    if new.shape != (2, old.shape[1]):
        raise ValueError("new_anchor_signatures shape mismatch")
    i, j = anchor_indices
    dx = old[j] - old[i]
    dy = new[1] - new[0]
    denom = float(dx @ dx)
    scale = float((dx @ dy) / max(denom, 1e-15))
    offset = 0.5 * ((new[0] - scale * old[i]) + (new[1] - scale * old[j]))
    return scale, offset


def recenter_from_unwritten_baseline(
    codebook: LearnedCodebook,
    matter: VirtualOrganoid,
    *,
    wait_steps: int,
    probe: StimulusAction,
    noise: float,
    seed: int,
) -> tuple[LearnedCodebook, dict]:
    """Attacker: treat drift as a changed observation origin, not a changed language."""
    rng = np.random.default_rng(seed)
    new_baseline = _response_after_write(
        matter,
        None,
        wait_steps=wait_steps,
        probe=probe,
        observation_noise=noise,
        rng=rng,
    )
    offset = new_baseline - codebook.baseline_signature
    return (
        LearnedCodebook(
            actions=codebook.actions,
            signatures=codebook.signatures + offset,
            baseline_signature=new_baseline,
            candidate_count=codebook.candidate_count,
        ),
        {
            "baseline_probe_interventions": 1,
            "offset_norm": float(np.round(np.linalg.norm(offset), 12)),
        },
    )


def recalibrate_two_anchors(
    codebook: LearnedCodebook,
    matter: VirtualOrganoid,
    *,
    wait_steps: int,
    probe: StimulusAction,
    noise: float,
    seed: int,
) -> tuple[LearnedCodebook, dict]:
    anchors = farthest_anchor_pair(codebook.signatures)
    rng = np.random.default_rng(seed)
    measured = np.asarray(
        [
            _response_after_write(
                matter,
                codebook.actions[index],
                wait_steps=wait_steps,
                probe=probe,
                observation_noise=noise,
                rng=rng,
            )
            for index in anchors
        ]
    )
    scale, offset = fit_two_anchor_alignment(codebook.signatures, measured, anchors)
    transformed = scale * codebook.signatures + offset
    transformed_baseline = scale * codebook.baseline_signature + offset
    return (
        LearnedCodebook(
            actions=codebook.actions,
            signatures=transformed,
            baseline_signature=transformed_baseline,
            candidate_count=codebook.candidate_count,
        ),
        {
            "anchor_indices": list(anchors),
            "baseline_recenter_probes": 1,
        "recalibration_interventions": 2,
            "global_scale": float(np.round(scale, 12)),
            "offset_norm": float(np.round(np.linalg.norm(offset), 12)),
        },
    )


def _eval(
    matter: VirtualOrganoid,
    codebook: LearnedCodebook,
    *,
    wait_steps: int,
    probe: StimulusAction,
    seed: int,
) -> dict:
    m = evaluate_codebook(
        matter,
        codebook,
        wait_steps=wait_steps,
        probe=probe,
        repeats=3,
        observation_noise=2e-4,
        seed=seed,
    )
    return {
        "accuracy": float(np.round(m["accuracy"], 12)),
        "median_margin": float(np.round(m["median_classification_margin"], 12)),
    }


def _baseline_codebooks(
    seed: int,
    *,
    budget: int,
    random_repeats: int,
    wait_steps: int,
) -> tuple[LearnedCodebook, list[LearnedCodebook]]:
    cand = candidates()
    probe = common_probe()
    active = discover_active(
        make_matter(seed),
        cand,
        wait_steps=wait_steps,
        probe=probe,
        budget=budget,
        codes=6,
    ).codebook
    random_books = []
    for repeat in range(random_repeats):
        rng = np.random.default_rng(20000 + 113 * seed + repeat)
        random_books.append(
            discover_random(
                make_matter(seed),
                cand,
                wait_steps=wait_steps,
                probe=probe,
                budget=budget,
                codes=6,
                rng=rng,
            ).codebook
        )
    return active, random_books


def _world(
    seed: int,
    *,
    budget: int,
    random_repeats: int,
    wait_steps: int,
) -> dict:
    probe = common_probe()
    active, random_books = _baseline_codebooks(
        seed,
        budget=budget,
        random_repeats=random_repeats,
        wait_steps=wait_steps,
    )

    baseline_active = _eval(
        make_matter(seed), active, wait_steps=wait_steps, probe=probe, seed=30000 + seed
    )
    baseline_random = [
        _eval(
            make_matter(seed),
            book,
            wait_steps=wait_steps,
            probe=probe,
            seed=31000 + 31 * seed + r,
        )
        for r, book in enumerate(random_books)
    ]

    drifts = {}
    for d, spec in enumerate(drift_specs()):
        zero_active = _eval(
            make_matter(seed, spec),
            active,
            wait_steps=wait_steps,
            probe=probe,
            seed=32000 + 101 * seed + d,
        )
        zero_random = [
            _eval(
                make_matter(seed, spec),
                book,
                wait_steps=wait_steps,
                probe=probe,
                seed=33000 + 101 * seed + 11 * d + r,
            )
            for r, book in enumerate(random_books)
        ]

        active_recenter, active_recenter_info = recenter_from_unwritten_baseline(
            active,
            make_matter(seed, spec),
            wait_steps=wait_steps,
            probe=probe,
            noise=1e-4,
            seed=33500 + 101 * seed + d,
        )
        active_recenter_metrics = _eval(
            make_matter(seed, spec),
            active_recenter,
            wait_steps=wait_steps,
            probe=probe,
            seed=33700 + 101 * seed + d,
        )
        random_recenter_metrics = []
        for r, book in enumerate(random_books):
            recentered, _ = recenter_from_unwritten_baseline(
                book,
                make_matter(seed, spec),
                wait_steps=wait_steps,
                probe=probe,
                noise=1e-4,
                seed=33800 + 101 * seed + 11 * d + r,
            )
            random_recenter_metrics.append(
                _eval(
                    make_matter(seed, spec),
                    recentered,
                    wait_steps=wait_steps,
                    probe=probe,
                    seed=33900 + 101 * seed + 11 * d + r,
                )
            )

        active_recal, active_info = recalibrate_two_anchors(
            active,
            make_matter(seed, spec),
            wait_steps=wait_steps,
            probe=probe,
            noise=1e-4,
            seed=34000 + 101 * seed + d,
        )
        active_recal_metrics = _eval(
            make_matter(seed, spec),
            active_recal,
            wait_steps=wait_steps,
            probe=probe,
            seed=35000 + 101 * seed + d,
        )

        random_recal_metrics = []
        for r, book in enumerate(random_books):
            recal, _ = recalibrate_two_anchors(
                book,
                make_matter(seed, spec),
                wait_steps=wait_steps,
                probe=probe,
                noise=1e-4,
                seed=36000 + 101 * seed + 11 * d + r,
            )
            random_recal_metrics.append(
                _eval(
                    make_matter(seed, spec),
                    recal,
                    wait_steps=wait_steps,
                    probe=probe,
                    seed=37000 + 101 * seed + 11 * d + r,
                )
            )

        drifts[spec.name] = {
            "parameters": {
                "cell_angle_offset": spec.cell_angle_offset,
                "recurrent_gain": spec.recurrent_gain,
            },
            "active_zero_shot": zero_active,
            "random_zero_shot_accuracy_median": float(
                np.round(np.median([m["accuracy"] for m in zero_random]), 12)
            ),
            "active_baseline_recenter": active_recenter_metrics,
            "random_baseline_recenter_accuracy_median": float(
                np.round(np.median([m["accuracy"] for m in random_recenter_metrics]), 12)
            ),
            "baseline_recenter": active_recenter_info,
            "active_two_anchor": active_recal_metrics,
            "random_two_anchor_accuracy_median": float(
                np.round(np.median([m["accuracy"] for m in random_recal_metrics]), 12)
            ),
            "active_recalibration": active_info,
        }

    return {
        "world_seed": seed,
        "baseline": {
            "active_accuracy": baseline_active["accuracy"],
            "random_accuracy_median": float(
                np.round(np.median([m["accuracy"] for m in baseline_random]), 12)
            ),
            "active_min_pair_distance": float(
                np.round(
                    min(
                        np.linalg.norm(active.signatures[i] - active.signatures[j])
                        for i in range(6)
                        for j in range(i + 1, 6)
                    ),
                    12,
                )
            ),
            "random_min_pair_distance_median": float(
                np.round(
                    np.median(
                        [
                            min(
                                np.linalg.norm(book.signatures[i] - book.signatures[j])
                                for i in range(6)
                                for j in range(i + 1, 6)
                            )
                            for book in random_books
                        ]
                    ),
                    12,
                )
            ),
        },
        "drifts": drifts,
    }


def run_v5(
    *,
    budget: int = 12,
    random_repeats: int = 3,
    wait_steps: int = 96,
    world_seeds: tuple[int, ...] = (4, 9, 14),
) -> dict:
    worlds = [
        _world(
            seed,
            budget=budget,
            random_repeats=random_repeats,
            wait_steps=wait_steps,
        )
        for seed in world_seeds
    ]

    scenario_summary = {}
    active_zero_all = []
    random_zero_all = []
    active_recenter_all = []
    random_recenter_all = []
    active_recal_all = []
    random_recal_all = []
    for spec in drift_specs():
        active_zero = np.asarray(
            [w["drifts"][spec.name]["active_zero_shot"]["accuracy"] for w in worlds]
        )
        random_zero = np.asarray(
            [w["drifts"][spec.name]["random_zero_shot_accuracy_median"] for w in worlds]
        )
        active_recenter = np.asarray(
            [w["drifts"][spec.name]["active_baseline_recenter"]["accuracy"] for w in worlds]
        )
        random_recenter = np.asarray(
            [w["drifts"][spec.name]["random_baseline_recenter_accuracy_median"] for w in worlds]
        )
        active_recal = np.asarray(
            [w["drifts"][spec.name]["active_two_anchor"]["accuracy"] for w in worlds]
        )
        random_recal = np.asarray(
            [w["drifts"][spec.name]["random_two_anchor_accuracy_median"] for w in worlds]
        )
        active_zero_all.extend(active_zero.tolist())
        random_zero_all.extend(random_zero.tolist())
        active_recenter_all.extend(active_recenter.tolist())
        random_recenter_all.extend(random_recenter.tolist())
        active_recal_all.extend(active_recal.tolist())
        random_recal_all.extend(random_recal.tolist())
        scenario_summary[spec.name] = {
            "active_zero_shot_median": float(np.round(np.median(active_zero), 12)),
            "random_zero_shot_median": float(np.round(np.median(random_zero), 12)),
            "active_baseline_recenter_median": float(np.round(np.median(active_recenter), 12)),
            "random_baseline_recenter_median": float(np.round(np.median(random_recenter), 12)),
            "active_two_anchor_median": float(np.round(np.median(active_recal), 12)),
            "random_two_anchor_median": float(np.round(np.median(random_recal), 12)),
            "active_zero_shot_world_wins": int(np.sum(active_zero > random_zero)),
            "active_two_anchor_world_wins": int(np.sum(active_recal > random_recal)),
        }

    summary = {
        "active_zero_shot_median": float(np.round(np.median(active_zero_all), 12)),
        "random_zero_shot_median": float(np.round(np.median(random_zero_all), 12)),
        "active_baseline_recenter_median": float(np.round(np.median(active_recenter_all), 12)),
        "random_baseline_recenter_median": float(np.round(np.median(random_recenter_all), 12)),
        "active_two_anchor_median": float(np.round(np.median(active_recal_all), 12)),
        "random_two_anchor_median": float(np.round(np.median(random_recal_all), 12)),
        "baseline_active_accuracy_median": float(
            np.round(np.median([w["baseline"]["active_accuracy"] for w in worlds]), 12)
        ),
        "baseline_random_accuracy_median": float(
            np.round(np.median([w["baseline"]["random_accuracy_median"] for w in worlds]), 12)
        ),
    }

    gate = {
        "baseline_both_at_least_0_95": bool(
            min(
                summary["baseline_active_accuracy_median"],
                summary["baseline_random_accuracy_median"],
            )
            >= 0.95
        ),
        "two_anchor_cost_less_than_full_codebook": True,
        "active_recalibrated_accuracy_at_least_0_75": bool(
            summary["active_two_anchor_median"] >= 0.75
        ),
        "active_zero_shot_beats_random_by_0_10": bool(
            summary["active_zero_shot_median"]
            >= summary["random_zero_shot_median"] + 0.10
        ),
        "active_two_anchor_beats_random_by_0_10": bool(
            summary["active_two_anchor_median"]
            >= summary["random_two_anchor_median"] + 0.10
        ),
    }
    margin_earned = bool(
        gate["active_zero_shot_beats_random_by_0_10"]
        or gate["active_two_anchor_beats_random_by_0_10"]
    )
    verdict = (
        "PASS_WIDE_CODE_EARNS_DRIFT_ROBUSTNESS"
        if gate["baseline_both_at_least_0_95"]
        and gate["active_recalibrated_accuracy_at_least_0_75"]
        and margin_earned
        else "FAIL_WIDE_CODE_DRIFT_VALUE"
    )

    return {
        "experiment": "v5_drifting_stimulation_language",
        "world_seeds": list(world_seeds),
        "baseline_probe_budget": budget,
        "random_repeats_per_world": random_repeats,
        "code_symbols": 6,
        "recalibration_interventions": 2,
        "drift_specs": [
            {
                "name": spec.name,
                "cell_angle_offset": spec.cell_angle_offset,
                "recurrent_gain": spec.recurrent_gain,
            }
            for spec in drift_specs()
        ],
        "worlds": worlds,
        "scenario_summary": scenario_summary,
        "summary": summary,
        "gate": gate,
        "margin_earned": margin_earned,
        "verdict": verdict,
        "claim_boundary": (
            "The drift axes are synthetic and frozen before the v5 outcome: a 0.06-rad cell/electrode geometry "
            "shift, a 15% recurrent-gain increase, and their combination. Baseline recentering is a post-result "
            "attacker that measures only the new unwritten response origin. Two-anchor recalibration fits only one "
            "global scalar gain plus one response-space offset vector. This tests whether v4's larger code margin "
            "has transfer value; it does not model biological homeostasis or continual learning."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/v5_drift.json"))
    args = parser.parse_args()
    receipt = run_v5()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
