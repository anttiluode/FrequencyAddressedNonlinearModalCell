import numpy as np

from frequency_modal_cell.drift_experiment import (
    farthest_anchor_pair,
    fit_two_anchor_alignment,
    recenter_from_unwritten_baseline,
)
from frequency_modal_cell.virtual_organoid import LearnedCodebook, StimulusAction


def test_two_anchor_alignment_recovers_global_gain_and_offset():
    old = np.array(
        [
            [0.0, 1.0, 2.0],
            [2.0, -1.0, 0.5],
            [0.3, 0.4, -0.2],
        ],
        dtype=float,
    )
    anchors = (0, 1)
    scale_true = 1.17
    offset_true = np.array([0.2, -0.1, 0.05])
    new = np.vstack(
        [
            scale_true * old[anchors[0]] + offset_true,
            scale_true * old[anchors[1]] + offset_true,
        ]
    )
    scale, offset = fit_two_anchor_alignment(old, new, anchors)
    assert np.isclose(scale, scale_true)
    assert np.allclose(offset, offset_true)


def test_farthest_anchor_pair_uses_baseline_geometry_only():
    signatures = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 3.0], [0.2, 0.2]])
    assert farthest_anchor_pair(signatures) == (1, 2)


class BaselineMatter:
    n_electrodes = 1

    def reset(self):
        pass

    def stimulate(self, action):
        raise AssertionError("baseline recenter must not stimulate a code symbol")

    def wait(self, steps):
        pass

    def probe_signature(self, action, *, observation_noise=0.0, rng=None):
        return np.array([2.0, 4.0], dtype=float)


def test_baseline_recenter_moves_only_the_observation_origin():
    action = StimulusAction(electrode=0, omega=0.3, duration=8)
    book = LearnedCodebook(
        actions=(action, action),
        signatures=np.array([[1.5, 2.5], [0.5, 3.5]], dtype=float),
        baseline_signature=np.array([1.0, 2.0], dtype=float),
        candidate_count=2,
    )
    shifted, info = recenter_from_unwritten_baseline(
        book,
        BaselineMatter(),
        wait_steps=3,
        probe=action,
        noise=0.0,
        seed=1,
    )
    assert np.allclose(shifted.baseline_signature, [2.0, 4.0])
    assert np.allclose(shifted.signatures, book.signatures + [1.0, 2.0])
    assert info["baseline_probe_interventions"] == 1
