from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .virtual_organoid import (
    LinearPortReservoir,
    StimulusAction,
    VirtualOrganoid,
    evaluate_codebook,
    learn_stimulation_codebook,
)


def _action_dict(action: StimulusAction) -> dict:
    return {
        "electrode": action.electrode,
        "omega": float(np.round(action.omega, 12)),
        "phase": float(np.round(action.phase, 12)),
        "amplitude": float(np.round(action.amplitude, 12)),
        "duration": action.duration,
    }


def _round_metrics(metrics: dict) -> dict:
    return {key: float(np.round(value, 12)) for key, value in metrics.items()}


def _candidate_actions() -> list[StimulusAction]:
    return [
        StimulusAction(electrode=e, omega=omega, phase=phase, amplitude=0.12, duration=72)
        for e in range(8)
        for omega in (0.18, 0.30, 0.42, 0.54)
        for phase in (0.0, np.pi / 2.0)
    ]


def _run_arm(name: str, matter, candidates, *, wait_steps: int, probe: StimulusAction, seed: int) -> dict:
    codebook = learn_stimulation_codebook(
        matter,
        candidates,
        wait_steps=wait_steps,
        probe=probe,
        codes=4,
    )
    metrics = evaluate_codebook(
        matter,
        codebook,
        wait_steps=wait_steps,
        probe=probe,
        repeats=12,
        observation_noise=2e-4,
        seed=seed,
    )
    return {
        "name": name,
        "codebook": [_action_dict(action) for action in codebook.actions],
        "metrics": _round_metrics(metrics),
    }


def run_v3(seed: int = 31, wait_steps: int = 96) -> dict:
    candidates = _candidate_actions()
    probe = StimulusAction(electrode=0, omega=0.36, phase=0.30, amplitude=0.07, duration=64)

    fanmc = _run_arm(
        "fanmc_hidden_matter",
        VirtualOrganoid(seed=4),
        candidates,
        wait_steps=wait_steps,
        probe=probe,
        seed=seed,
    )
    fast = _run_arm(
        "same_fast_matter_no_slow_write",
        VirtualOrganoid(seed=4, slow_write=0.0, slow_gain=0.0),
        candidates,
        wait_steps=wait_steps,
        probe=probe,
        seed=seed + 1,
    )
    linear = _run_arm(
        "linear_state_space_reservoir",
        LinearPortReservoir(seed=9),
        candidates,
        wait_steps=wait_steps,
        probe=probe,
        seed=seed + 2,
    )

    fanmc_acc = fanmc["metrics"]["accuracy"]
    fast_acc = fast["metrics"]["accuracy"]
    fanmc_persist = fanmc["metrics"]["max_write_vs_baseline_distance"]
    fast_persist = fast["metrics"]["max_write_vs_baseline_distance"]
    linear_acc = linear["metrics"]["accuracy"]

    gate = {
        "fanmc_learns_four_code_black_box_vocabulary": bool(fanmc_acc >= 0.90),
        "write_survives_96_step_wait": bool(fanmc_persist >= 5e-4),
        "slow_state_ablation_near_chance": bool(fast_acc <= 0.50),
        "persistent_signal_exceeds_fast_ablation_10x": bool(fanmc_persist >= 10.0 * max(fast_persist, 1e-12)),
    }

    return {
        "experiment": "v3_virtual_organoid_interface",
        "seed": seed,
        "interface": {
            "electrodes": 8,
            "controller_access": [
                "reset",
                "stimulate(electrode, frequency, phase, amplitude, duration)",
                "wait",
                "probe/read",
            ],
            "controller_denied": [
                "cell states",
                "material state",
                "recurrent matrix",
                "electrode-to-cell map",
                "gradients through internals",
            ],
            "candidate_stimulations": len(candidates),
            "codes_selected": 4,
            "wait_steps": wait_steps,
            "probe": _action_dict(probe),
            "heldout_perturbations": "1% frequency jitter, 0.08-rad phase jitter, 4% amplitude jitter, recording noise",
        },
        "arms": {
            "fanmc_hidden_matter": fanmc,
            "same_fast_matter_no_slow_write": fast,
            "linear_state_space_reservoir": linear,
        },
        "gate": gate,
        "verdict": "PASS_BLACK_BOX_STIMULATION_LANGUAGE" if all(gate.values()) else "FAIL_BLACK_BOX_STIMULATION_LANGUAGE",
        "attacker_note": (
            "The linear state-space reservoir uses the identical black-box eight-port interface. "
            "If it matches FANMC, v3 establishes the interface and persistent-address mechanism, not a FANMC advantage."
        ),
        "claim_boundary": (
            "This is a synthetic virtual-organoid task inspired by wetware-computing interfaces. "
            "It does not model an organoid, infer biological mechanisms, or establish that frequency-addressed matter "
            "outperforms conventional reservoirs. Internal substrate parameters are fixed; only the external stimulation "
            "codebook is learned."
        ),
        "source_inspiration": (
            "Jordan et al. (2024), Open and remotely accessible Neuroplatform for research in wetware computing, "
            "Frontiers in Artificial Intelligence 7:1376042. The paper motivates programming an unknown, non-stationary "
            "biological network through spatiotemporal stimulation and closed-loop observation rather than direct weight writes."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/v3_virtual_organoid.json"))
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--wait", type=int, default=96)
    args = parser.parse_args()
    receipt = run_v3(seed=args.seed, wait_steps=args.wait)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
