from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class AxonSegment:
    arbor: int
    index: int
    parent: int
    start: np.ndarray
    end: np.ndarray
    depth: int

    @property
    def vector(self) -> np.ndarray:
        return self.end - self.start

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.vector))

    @property
    def midpoint(self) -> np.ndarray:
        return 0.5 * (self.start + self.end)


@dataclass
class AxonArbor:
    arbor_id: int
    soma: np.ndarray
    targets: np.ndarray
    segments: list[AxonSegment]
    terminal_segment_indices: list[int]
    target_errors: np.ndarray

    @property
    def total_length(self) -> float:
        return float(sum(seg.length for seg in self.segments))

    @property
    def branch_count(self) -> int:
        children: dict[int, int] = {}
        for seg in self.segments:
            children[seg.parent] = children.get(seg.parent, 0) + 1
        return int(sum(v > 1 for v in children.values()))

    def path_to_terminal(self, terminal_index: int) -> list[AxonSegment]:
        by_index = {seg.index: seg for seg in self.segments}
        out: list[AxonSegment] = []
        cur = int(terminal_index)
        while cur >= 0:
            seg = by_index[cur]
            out.append(seg)
            cur = seg.parent
        return list(reversed(out))


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


def _grow_polyline(
    arbor_id: int,
    segments: list[AxonSegment],
    start: np.ndarray,
    target: np.ndarray,
    *,
    parent: int,
    depth: int,
    step_length: float,
    guidance: float,
    inertia: float,
    noise: float,
    rng: np.random.Generator,
    initial_direction: np.ndarray,
) -> int:
    pos = np.asarray(start, dtype=float).copy()
    direction = _unit(np.asarray(initial_direction, dtype=float))
    max_steps = int(np.ceil(np.linalg.norm(target - pos) / step_length)) + 12
    last = parent
    for _ in range(max_steps):
        delta = target - pos
        dist = float(np.linalg.norm(delta))
        if dist <= step_length:
            new_pos = target.copy()
        else:
            jitter = rng.normal(size=pos.size)
            jitter -= np.dot(jitter, direction) * direction
            new_dir = _unit(inertia * direction + guidance * _unit(delta) + noise * jitter)
            if not np.any(new_dir):
                new_dir = _unit(delta)
            new_pos = pos + min(step_length, dist) * new_dir
            direction = new_dir
        idx = len(segments)
        segments.append(
            AxonSegment(
                arbor=arbor_id,
                index=idx,
                parent=last,
                start=pos.copy(),
                end=new_pos.copy(),
                depth=depth,
            )
        )
        last = idx
        pos = new_pos
        if float(np.linalg.norm(target - pos)) < 1e-10:
            break
    return last


def grow_branching_arbor(
    soma: np.ndarray,
    targets: np.ndarray,
    *,
    arbor_id: int = 0,
    step_length: float = 0.055,
    trunk_fraction: float = 0.46,
    guidance: float = 0.78,
    inertia: float = 0.62,
    noise: float = 0.035,
    seed: int = 0,
) -> AxonArbor:
    """Grow a long trunk and branch it toward multiple distant targets.

    A shared growth cone follows a chemoaffinity-like target direction with inertia
    and weak stochastic wandering. Daughter cones inherit the parent direction and
    then seek individual targets. This is a computational growth model, not a fitted
    developmental mechanism.
    """
    soma = np.asarray(soma, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if soma.shape != (2,) or targets.ndim != 2 or targets.shape[1] != 2 or len(targets) < 2:
        raise ValueError("soma must be (2,) and targets must be [n>=2,2]")
    rng = np.random.default_rng(seed)
    centroid = targets.mean(axis=0)
    branch_point = soma + trunk_fraction * (centroid - soma)
    segments: list[AxonSegment] = []
    trunk_terminal = _grow_polyline(
        arbor_id, segments, soma, branch_point, parent=-1, depth=0,
        step_length=step_length, guidance=guidance, inertia=inertia, noise=noise,
        rng=rng, initial_direction=centroid - soma,
    )
    branch_pos = segments[trunk_terminal].end.copy()
    inherited = segments[trunk_terminal].vector
    terminals: list[int] = []
    errors: list[float] = []
    for target in targets:
        term = _grow_polyline(
            arbor_id, segments, branch_pos, target, parent=trunk_terminal, depth=1,
            step_length=step_length, guidance=guidance, inertia=inertia, noise=noise,
            rng=rng, initial_direction=inherited,
        )
        terminals.append(term)
        errors.append(float(np.linalg.norm(segments[term].end - target)))
    return AxonArbor(arbor_id, soma.copy(), targets.copy(), segments, terminals, np.asarray(errors))


def _segment_alignment(a: AxonSegment, b: AxonSegment) -> float:
    return float(abs(np.dot(_unit(a.vector), _unit(b.vector))))


def ephaptic_contacts(
    arbors: Iterable[AxonArbor],
    *,
    length_scale: float = 0.055,
    orientation_power: float = 4.0,
    cutoff: float = 0.16,
) -> dict[tuple[tuple[int, int], tuple[int, int]], float]:
    """Sparse local extracellular coupling between nearby segments of different axons."""
    segments = [seg for arbor in arbors for seg in arbor.segments]
    out: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}
    for i, a in enumerate(segments):
        for b in segments[i + 1:]:
            if a.arbor == b.arbor:
                continue
            dist = float(np.linalg.norm(a.midpoint - b.midpoint))
            if dist > cutoff:
                continue
            weight = float(np.exp(-dist / length_scale) * _segment_alignment(a, b) ** orientation_power)
            if weight > 1e-4:
                out[((a.arbor, a.index), (b.arbor, b.index))] = weight
    return out


def _contact_lookup(contacts):
    lookup: dict[tuple[int, int], list[tuple[tuple[int, int], float]]] = {}
    for (a, b), w in contacts.items():
        lookup.setdefault(a, []).append((b, w))
        lookup.setdefault(b, []).append((a, w))
    return lookup


def terminal_arrival_times(
    arbors: list[AxonArbor],
    emission_times: dict[int, float],
    *,
    speed: float = 0.045,
    ephaptic_gain: float = 0.006,
    temporal_scale: float = 3.0,
    contacts=None,
) -> dict[tuple[int, int], float]:
    """Propagate along grown branches with a weak local coactivity-dependent delay shift.

    Ephaptic coupling changes timing only; it never creates a target connection.
    Coupling is evaluated from uncoupled passage times to keep this first gate explicit.
    """
    if speed <= 0:
        raise ValueError("speed must be positive")
    contacts = ephaptic_contacts(arbors) if contacts is None else contacts
    lookup = _contact_lookup(contacts)
    base_start = {}
    by_key = {}
    for arbor in arbors:
        t0 = float(emission_times.get(arbor.arbor_id, 0.0))
        end_times = {}
        for seg in arbor.segments:
            start_t = t0 if seg.parent < 0 else end_times[seg.parent]
            key = (arbor.arbor_id, seg.index)
            base_start[key] = start_t
            by_key[key] = seg
            end_times[seg.index] = start_t + seg.length / speed

    adjusted_duration = {}
    for key, seg in by_key.items():
        mid_t = base_start[key] + 0.5 * seg.length / speed
        influence = 0.0
        for other_key, weight in lookup.get(key, []):
            other = by_key[other_key]
            other_mid = base_start[other_key] + 0.5 * other.length / speed
            overlap = np.exp(-0.5 * ((mid_t - other_mid) / temporal_scale) ** 2)
            influence += weight * float(overlap)
        frac = min(max(ephaptic_gain * influence, 0.0), 0.05)
        adjusted_duration[key] = (seg.length / speed) * (1.0 - frac)

    arrivals = {}
    for arbor in arbors:
        for target_idx, terminal in enumerate(arbor.terminal_segment_indices):
            delay = sum(adjusted_duration[(arbor.arbor_id, seg.index)] for seg in arbor.path_to_terminal(terminal))
            arrivals[(arbor.arbor_id, target_idx)] = float(emission_times.get(arbor.arbor_id, 0.0) + delay)
    return arrivals


def arrival_phase_shift(coupled, uncoupled, *, omega: float):
    return {key: float(omega * (coupled[key] - uncoupled[key])) for key in coupled}
