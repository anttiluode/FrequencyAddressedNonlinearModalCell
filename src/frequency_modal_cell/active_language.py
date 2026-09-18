from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .virtual_organoid import (
    BlackBoxMatter,
    LearnedCodebook,
    StimulusAction,
    _response_after_write,
)


@dataclass(frozen=True)
class BudgetedDiscovery:
    codebook: LearnedCodebook
    probed_indices: tuple[int, ...]
    probe_budget: int
    strategy: str


def action_features(action: StimulusAction, n_electrodes: int) -> np.ndarray:
    """Controller-visible stimulation coordinates; no substrate state is used."""
    one_hot = np.zeros(n_electrodes, dtype=float)
    one_hot[action.electrode] = 1.0
    omega = (float(action.omega) - 0.36) / 0.20
    phase = float(action.phase)
    phase_pair = np.array([np.sin(phase), np.cos(phase)], dtype=float)
    # The interaction lets a cheap surrogate learn that frequency can mean different
    # things at different physical ports without seeing the hidden port-to-cell map.
    return np.concatenate(
        [
            np.array([1.0], dtype=float),
            one_hot,
            np.array([omega, omega * omega], dtype=float),
            phase_pair,
            one_hot * omega,
        ]
    )


def _feature_matrix(candidates: Sequence[StimulusAction], n_electrodes: int) -> np.ndarray:
    return np.asarray([action_features(action, n_electrodes) for action in candidates], dtype=float)


def geometry_cover_indices(
    candidates: Sequence[StimulusAction],
    *,
    n_electrodes: int,
    budget: int,
) -> list[int]:
    """Fixed farthest-point design in visible stimulus space only."""
    if budget < 1 or budget > len(candidates):
        raise ValueError("invalid probe budget")
    features = _feature_matrix(candidates, n_electrodes)
    varying = np.std(features, axis=0) > 1e-12
    f = features[:, varying]
    if f.shape[1] == 0:
        return list(range(budget))
    scale = np.maximum(np.std(f, axis=0), 1e-9)
    f = (f - np.mean(f, axis=0)) / scale

    center = np.mean(f, axis=0)
    selected = [int(np.argmin(np.linalg.norm(f - center, axis=1)))]
    while len(selected) < budget:
        d = np.linalg.norm(f[:, None, :] - f[selected][None, :, :], axis=2)
        nearest = np.min(d, axis=1)
        nearest[selected] = -1.0
        selected.append(int(np.argmax(np.round(nearest, 12))))
    return selected


def _select_codebook(
    candidates: Sequence[StimulusAction],
    indices: Sequence[int],
    signatures: np.ndarray,
    baseline: np.ndarray,
    *,
    codes: int,
) -> LearnedCodebook:
    if codes > len(indices):
        raise ValueError("codes exceed observed interventions")
    stable = np.round(np.asarray(signatures, dtype=float), 12)
    stable_baseline = np.round(np.asarray(baseline, dtype=float), 12)

    chosen_local = [int(np.argmax(np.linalg.norm(stable - stable_baseline, axis=1)))]
    while len(chosen_local) < codes:
        d = np.linalg.norm(stable[:, None, :] - stable[chosen_local][None, :, :], axis=2)
        nearest = np.min(d, axis=1)
        nearest[chosen_local] = -1.0
        chosen_local.append(int(np.argmax(np.round(nearest, 12))))

    return LearnedCodebook(
        actions=tuple(candidates[int(indices[i])] for i in chosen_local),
        signatures=np.asarray(signatures)[chosen_local],
        baseline_signature=np.asarray(baseline),
        candidate_count=len(candidates),
    )


def discover_from_indices(
    matter: BlackBoxMatter,
    candidates: Sequence[StimulusAction],
    indices: Sequence[int],
    *,
    wait_steps: int,
    probe: StimulusAction,
    codes: int,
    strategy: str,
) -> BudgetedDiscovery:
    baseline = _response_after_write(matter, None, wait_steps=wait_steps, probe=probe)
    signatures = np.asarray(
        [
            _response_after_write(matter, candidates[int(i)], wait_steps=wait_steps, probe=probe)
            for i in indices
        ]
    )
    return BudgetedDiscovery(
        codebook=_select_codebook(candidates, indices, signatures, baseline, codes=codes),
        probed_indices=tuple(int(i) for i in indices),
        probe_budget=len(indices),
        strategy=strategy,
    )


def discover_geometry_cover(
    matter: BlackBoxMatter,
    candidates: Sequence[StimulusAction],
    *,
    wait_steps: int,
    probe: StimulusAction,
    budget: int,
    codes: int,
) -> BudgetedDiscovery:
    indices = geometry_cover_indices(candidates, n_electrodes=matter.n_electrodes, budget=budget)
    return discover_from_indices(
        matter,
        candidates,
        indices,
        wait_steps=wait_steps,
        probe=probe,
        codes=codes,
        strategy="stimulus_geometry_cover",
    )


def discover_random(
    matter: BlackBoxMatter,
    candidates: Sequence[StimulusAction],
    *,
    wait_steps: int,
    probe: StimulusAction,
    budget: int,
    codes: int,
    rng: np.random.Generator,
) -> BudgetedDiscovery:
    indices = rng.choice(len(candidates), size=budget, replace=False).tolist()
    return discover_from_indices(
        matter,
        candidates,
        indices,
        wait_steps=wait_steps,
        probe=probe,
        codes=codes,
        strategy="random",
    )


def discover_active(
    matter: BlackBoxMatter,
    candidates: Sequence[StimulusAction],
    *,
    wait_steps: int,
    probe: StimulusAction,
    budget: int,
    codes: int,
    warmup: int = 4,
    ridge: float = 2e-3,
    uncertainty_weight: float = 0.35,
) -> BudgetedDiscovery:
    """Choose new interventions from a response surrogate plus model uncertainty.

    Every selected action is actually executed before the next action is chosen.
    Candidate response signatures are never read unless that candidate consumes one
    unit of the intervention budget.
    """
    if warmup < 2 or warmup >= budget:
        raise ValueError("warmup must be at least two and smaller than budget")
    features = _feature_matrix(candidates, matter.n_electrodes)
    initial = geometry_cover_indices(candidates, n_electrodes=matter.n_electrodes, budget=warmup)

    baseline = _response_after_write(matter, None, wait_steps=wait_steps, probe=probe)
    observed = list(initial)
    signatures = [
        _response_after_write(matter, candidates[i], wait_steps=wait_steps, probe=probe)
        for i in observed
    ]

    while len(observed) < budget:
        phi = features[observed]
        y = np.asarray(signatures)
        gram = phi.T @ phi + ridge * np.eye(phi.shape[1])
        inv = np.linalg.pinv(gram, rcond=1e-10)
        coef = inv @ phi.T @ y
        pred = features @ coef

        novelty = np.min(
            np.linalg.norm(pred[:, None, :] - y[None, :, :], axis=2),
            axis=1,
        )
        uncertainty = np.sqrt(
            np.maximum(np.einsum("ij,jk,ik->i", features, inv, features), 0.0)
        )
        uncertainty /= max(float(np.max(uncertainty)), 1e-12)

        if y.shape[0] > 1:
            pair = np.linalg.norm(y[:, None, :] - y[None, :, :], axis=2)
            response_scale = float(np.median(pair[np.triu_indices(y.shape[0], 1)]))
        else:
            response_scale = float(np.linalg.norm(y[0] - baseline))
        response_scale = max(response_scale, 1e-6)

        score = novelty + uncertainty_weight * response_scale * uncertainty
        score[np.asarray(observed, dtype=int)] = -np.inf
        # Acquisition ties are scientifically equivalent here but tiny LAPACK
        # differences can otherwise flip a discrete next-probe choice across Python
        # versions. Quantize well below the measured response margins, then use
        # np.argmax's stable first-index tie break.
        next_index = int(np.argmax(np.round(score, 9)))
        observed.append(next_index)
        signatures.append(
            _response_after_write(
                matter,
                candidates[next_index],
                wait_steps=wait_steps,
                probe=probe,
            )
        )

    return BudgetedDiscovery(
        codebook=_select_codebook(
            candidates,
            observed,
            np.asarray(signatures),
            baseline,
            codes=codes,
        ),
        probed_indices=tuple(observed),
        probe_budget=budget,
        strategy="response_adaptive_surrogate",
    )
