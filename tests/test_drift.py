import numpy as np

from frequency_modal_cell.drift_experiment import (
    farthest_anchor_pair,
    fit_two_anchor_alignment,
)


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
