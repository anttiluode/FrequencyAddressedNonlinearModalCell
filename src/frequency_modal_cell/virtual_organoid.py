from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .core import ResonantActiveCable


@dataclass(frozen=True)
class StimulusAction:
    electrode: int
    omega: float
    phase: float = 0.0
    amplitude: float = 0.12
    duration: int = 72


class BlackBoxMatter(Protocol):
    n_electrodes: int

    def reset(self) -> None: ...
    def stimulate(self, action: StimulusAction) -> None: ...
    def wait(self, steps: int) -> None: ...
    def probe_signature(
        self,
        action: StimulusAction,
        *,
        observation_noise: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray: ...


def _carrier(
    steps: int,
    n_electrodes: int,
    electrode: int,
    omega: float,
    phase: float,
    amplitude: float,
    taper: int = 6,
) -> np.ndarray:
    if electrode < 0 or electrode >= n_electrodes:
        raise ValueError("invalid electrode")
    out = np.zeros((steps, n_electrodes), dtype=float)
    idx = np.arange(steps, dtype=float)
    env = np.ones(steps, dtype=float)
    taper = min(int(taper), steps // 2)
    if taper > 0:
        ramp = np.sin(np.linspace(0.0, np.pi / 2.0, taper, endpoint=False)) ** 2
        env[:taper] = ramp
        env[-taper:] = ramp[::-1]
    out[:, electrode] = amplitude * env * np.sin(omega * idx + phase)
    return out


def _fourier_signature(trace: np.ndarray, omega: float) -> np.ndarray:
    trace = np.asarray(trace, dtype=float)
    t = np.arange(trace.shape[0], dtype=float)
    c = np.cos(omega * t)
    s = np.sin(omega * t)
    scale = 2.0 / max(trace.shape[0], 1)
    return np.concatenate([scale * trace.T @ c, scale * trace.T @ s])


class VirtualOrganoid:
    """A hidden population of FANMC-like cells behind an eight-electrode interface.

    The controller is intentionally denied the internal cable states, material state,
    recurrent matrix and electrode-to-cell map. The public interface is reset,
    stimulate, wait and probe_signature.

    The slow material variable is synthetic. It is included to test whether a
    stimulation language can write and later read persistent substrate state; it is
    not a fitted biological plasticity rule.
    """

    def __init__(
        self,
        *,
        n_cells: int = 6,
        n_electrodes: int = 8,
        cell_compartments: int = 8,
        slow_decay: float = 0.9975,
        slow_write: float = 0.006,
        slow_gain: float = 1.8,
        cell_angle_offset: float = 0.12,
        recurrent_gain: float = 0.18,
        seed: int = 4,
    ) -> None:
        self.n_electrodes = int(n_electrodes)
        self._cells = [ResonantActiveCable(n=cell_compartments) for _ in range(n_cells)]
        self._slow_decay = float(slow_decay)
        self._slow_write = float(slow_write)
        self._slow_gain = float(slow_gain)
        self._cell_angle_offset = float(cell_angle_offset)
        self._recurrent_gain = float(recurrent_gain)

        electrode_angle = np.linspace(0.0, 2.0 * np.pi, self.n_electrodes, endpoint=False)
        cell_angle = np.linspace(
            self._cell_angle_offset,
            2.0 * np.pi + self._cell_angle_offset,
            n_cells,
            endpoint=False,
        )
        electrode_pos = np.column_stack([np.cos(electrode_angle), np.sin(electrode_angle)])
        cell_pos = 0.62 * np.column_stack([np.cos(cell_angle), np.sin(cell_angle)])
        d2 = np.sum((electrode_pos[:, None, :] - cell_pos[None, :, :]) ** 2, axis=2)
        coupling = np.exp(-d2 / (2.0 * 0.5**2))
        coupling /= np.linalg.norm(coupling, axis=1, keepdims=True)
        self._electrode_to_cell = coupling

        rng = np.random.default_rng(seed)
        recurrent = rng.normal(size=(n_cells, n_cells))
        recurrent *= rng.random((n_cells, n_cells)) < 0.35
        np.fill_diagonal(recurrent, 0.0)
        radius = float(np.max(np.abs(np.linalg.eigvals(recurrent))))
        self._recurrent = self._recurrent_gain * recurrent / max(radius, 1e-12)
        self.reset()

    def reset(self) -> None:
        for cell in self._cells:
            cell.reset()
        self._material = np.zeros(len(self._cells), dtype=float)
        self._last_output = np.zeros(len(self._cells), dtype=float)

    def _step(self, electrode_drive: np.ndarray) -> np.ndarray:
        electrode_drive = np.asarray(electrode_drive, dtype=float)
        if electrode_drive.shape != (self.n_electrodes,):
            raise ValueError("electrode drive shape mismatch")

        external = self._electrode_to_cell.T @ electrode_drive
        recurrent = self._recurrent @ self._last_output
        output = []
        for i, cell in enumerate(self._cells):
            gain = 1.0 + self._slow_gain * self._material[i]
            local = np.array(
                [
                    gain * external[i] + 0.35 * recurrent[i],
                    0.65 * gain * external[i] - 0.20 * recurrent[i],
                ],
                dtype=float,
            )
            cell.step(local)
            output.append(0.6 * cell.v[cell.n // 2] + 0.4 * cell.v[cell.n // 2 - 1])

        output = np.asarray(output)
        energy = np.clip(output**2, 0.0, 4.0)
        self._material = self._slow_decay * self._material + self._slow_write * energy
        self._material = np.clip(self._material, 0.0, 1.5)
        self._last_output = output
        return self._electrode_to_cell @ output

    def _run(self, drive: np.ndarray) -> np.ndarray:
        return np.asarray([self._step(row) for row in np.asarray(drive, dtype=float)])

    def stimulate(self, action: StimulusAction) -> None:
        self._run(
            _carrier(
                action.duration,
                self.n_electrodes,
                action.electrode,
                action.omega,
                action.phase,
                action.amplitude,
            )
        )

    def wait(self, steps: int) -> None:
        self._run(np.zeros((int(steps), self.n_electrodes), dtype=float))

    def probe_signature(
        self,
        action: StimulusAction,
        *,
        observation_noise: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        trace = self._run(
            _carrier(
                action.duration,
                self.n_electrodes,
                action.electrode,
                action.omega,
                action.phase,
                action.amplitude,
            )
        )
        if observation_noise > 0.0:
            rng = np.random.default_rng(0) if rng is None else rng
            trace = trace + rng.normal(0.0, observation_noise, size=trace.shape)
        return _fourier_signature(trace, action.omega)


class LinearPortReservoir:
    """Boring state-space attacker with the identical eight-port public interface."""

    def __init__(self, *, n_electrodes: int = 8, state_dim: int = 48, seed: int = 9) -> None:
        self.n_electrodes = int(n_electrodes)
        rng = np.random.default_rng(seed)
        q, _ = np.linalg.qr(rng.normal(size=(state_dim, state_dim)))
        eigenvalues = np.linspace(0.94, 0.995, state_dim)
        self._a = q @ np.diag(eigenvalues) @ q.T

        b = rng.normal(size=(state_dim, self.n_electrodes))
        b /= np.linalg.norm(b, axis=0, keepdims=True)
        self._b = 0.08 * b

        c = rng.normal(size=(self.n_electrodes, state_dim))
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        self._c = c
        self.reset()

    def reset(self) -> None:
        self._x = np.zeros(self._a.shape[0], dtype=float)

    def _step(self, electrode_drive: np.ndarray) -> np.ndarray:
        self._x = self._a @ self._x + self._b @ electrode_drive
        return self._c @ self._x

    def _run(self, drive: np.ndarray) -> np.ndarray:
        return np.asarray([self._step(row) for row in np.asarray(drive, dtype=float)])

    def stimulate(self, action: StimulusAction) -> None:
        self._run(
            _carrier(
                action.duration,
                self.n_electrodes,
                action.electrode,
                action.omega,
                action.phase,
                action.amplitude,
            )
        )

    def wait(self, steps: int) -> None:
        self._run(np.zeros((int(steps), self.n_electrodes), dtype=float))

    def probe_signature(
        self,
        action: StimulusAction,
        *,
        observation_noise: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        trace = self._run(
            _carrier(
                action.duration,
                self.n_electrodes,
                action.electrode,
                action.omega,
                action.phase,
                action.amplitude,
            )
        )
        if observation_noise > 0.0:
            rng = np.random.default_rng(0) if rng is None else rng
            trace = trace + rng.normal(0.0, observation_noise, size=trace.shape)
        return _fourier_signature(trace, action.omega)


@dataclass(frozen=True)
class LearnedCodebook:
    actions: tuple[StimulusAction, ...]
    signatures: np.ndarray
    baseline_signature: np.ndarray
    candidate_count: int


def _response_after_write(
    matter: BlackBoxMatter,
    action: StimulusAction | None,
    *,
    wait_steps: int,
    probe: StimulusAction,
    observation_noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    matter.reset()
    if action is not None:
        matter.stimulate(action)
    matter.wait(wait_steps)
    return matter.probe_signature(probe, observation_noise=observation_noise, rng=rng)


def learn_stimulation_codebook(
    matter: BlackBoxMatter,
    candidates: Sequence[StimulusAction],
    *,
    wait_steps: int,
    probe: StimulusAction,
    codes: int = 4,
) -> LearnedCodebook:
    """Learn a stimulation vocabulary using only black-box stimulation and recording."""
    if codes < 2 or codes > len(candidates):
        raise ValueError("invalid code count")

    baseline = _response_after_write(matter, None, wait_steps=wait_steps, probe=probe)
    signatures = np.asarray(
        [_response_after_write(matter, action, wait_steps=wait_steps, probe=probe) for action in candidates]
    )
    # Round before discrete selection so tiny BLAS/version differences cannot change
    # which action wins a near-tie in the frozen receipt.
    stable = np.round(signatures, 12)
    stable_baseline = np.round(baseline, 12)

    selected = [int(np.argmax(np.linalg.norm(stable - stable_baseline, axis=1)))]
    while len(selected) < codes:
        distance = np.linalg.norm(stable[:, None, :] - stable[selected][None, :, :], axis=2)
        nearest = np.min(distance, axis=1)
        nearest[selected] = -1.0
        selected.append(int(np.argmax(nearest)))

    return LearnedCodebook(
        actions=tuple(candidates[i] for i in selected),
        signatures=signatures[selected],
        baseline_signature=baseline,
        candidate_count=len(candidates),
    )


def jitter_action(action: StimulusAction, rng: np.random.Generator) -> StimulusAction:
    return StimulusAction(
        electrode=action.electrode,
        omega=float(action.omega * (1.0 + rng.normal(0.0, 0.01))),
        phase=float(action.phase + rng.normal(0.0, 0.08)),
        amplitude=float(action.amplitude * (1.0 + rng.normal(0.0, 0.04))),
        duration=action.duration,
    )


def evaluate_codebook(
    matter: BlackBoxMatter,
    codebook: LearnedCodebook,
    *,
    wait_steps: int,
    probe: StimulusAction,
    repeats: int = 12,
    observation_noise: float = 2e-4,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    correct = 0
    total = 0
    margins = []
    for target, action in enumerate(codebook.actions):
        for _ in range(repeats):
            perturbed = jitter_action(action, rng)
            signature = _response_after_write(
                matter,
                perturbed,
                wait_steps=wait_steps,
                probe=probe,
                observation_noise=observation_noise,
                rng=rng,
            )
            distances = np.linalg.norm(codebook.signatures - signature, axis=1)
            order = np.argsort(distances)
            pred = int(order[0])
            correct += int(pred == target)
            total += 1
            margins.append(float(distances[order[1]] - distances[order[0]]))
    pair_distances = []
    for i in range(len(codebook.signatures)):
        for j in range(i + 1, len(codebook.signatures)):
            pair_distances.append(float(np.linalg.norm(codebook.signatures[i] - codebook.signatures[j])))
    return {
        "accuracy": float(correct / max(total, 1)),
        "median_classification_margin": float(np.median(margins)),
        "min_calibration_pair_distance": float(min(pair_distances)),
        "max_write_vs_baseline_distance": float(
            max(np.linalg.norm(sig - codebook.baseline_signature) for sig in codebook.signatures)
        ),
    }
