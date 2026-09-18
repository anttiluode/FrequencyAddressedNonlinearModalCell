from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .axon import AxonArbor


@dataclass
class FixedDelayGraph:
    """Boring attacker for a frozen branching arbor.

    Each anatomical terminal becomes one sparse directed edge carrying only a fixed
    propagation delay. If branching geometry has no runtime state, this representation
    should be exactly sufficient for arrival timing.
    """

    delays: dict[tuple[int, int], float]

    @classmethod
    def from_arbors(cls, arbors: list[AxonArbor], *, speed: float = 0.045) -> "FixedDelayGraph":
        if speed <= 0:
            raise ValueError("speed must be positive")
        delays: dict[tuple[int, int], float] = {}
        for arbor in arbors:
            for target_idx, terminal in enumerate(arbor.terminal_segment_indices):
                length = sum(seg.length for seg in arbor.path_to_terminal(terminal))
                delays[(arbor.arbor_id, target_idx)] = float(length / speed)
        return cls(delays=delays)

    def arrivals(self, emission_times: dict[int, float]) -> dict[tuple[int, int], float]:
        return {
            key: float(emission_times.get(key[0], 0.0) + delay)
            for key, delay in self.delays.items()
        }


def rms_arrival_error(target: dict, pred: dict) -> float:
    keys = sorted(target)
    if keys != sorted(pred):
        raise ValueError("arrival dictionaries must have identical keys")
    return float(np.sqrt(np.mean([(target[k] - pred[k]) ** 2 for k in keys])))


def max_arrival_error(target: dict, pred: dict) -> float:
    keys = sorted(target)
    if keys != sorted(pred):
        raise ValueError("arrival dictionaries must have identical keys")
    return float(max(abs(target[k] - pred[k]) for k in keys))
