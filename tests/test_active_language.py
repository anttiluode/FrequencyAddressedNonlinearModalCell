import numpy as np

from frequency_modal_cell.active_language import (
    action_features,
    discover_active,
    geometry_cover_indices,
)
from frequency_modal_cell.virtual_organoid import StimulusAction


class ToyMatter:
    n_electrodes = 4

    def reset(self):
        self.state = np.zeros(4, dtype=float)

    def stimulate(self, action):
        self.state[action.electrode] += action.omega * (1.0 + 0.2 * np.cos(action.phase))

    def wait(self, steps):
        self.state *= 0.995 ** int(steps)

    def probe_signature(self, action, *, observation_noise=0.0, rng=None):
        return np.concatenate(
            [
                self.state,
                np.array([np.sum(self.state), np.linalg.norm(self.state)], dtype=float),
            ]
        )


def _small_candidates():
    return [
        StimulusAction(electrode=e, omega=w, phase=p, amplitude=0.09, duration=20)
        for e in range(4)
        for w in (0.20, 0.36, 0.52)
        for p in (0.0, np.pi / 2.0)
    ]


def test_action_features_are_controller_visible_and_finite():
    action = StimulusAction(electrode=2, omega=0.42, phase=0.7)
    features = action_features(action, 8)
    assert features.shape == (21,)
    assert np.isfinite(features).all()


def test_geometry_cover_spends_exact_unique_budget():
    candidates = _small_candidates()
    indices = geometry_cover_indices(candidates, n_electrodes=4, budget=7)
    assert len(indices) == 7
    assert len(set(indices)) == 7
    assert all(0 <= i < len(candidates) for i in indices)


def test_active_discovery_never_exceeds_purchased_probes():
    candidates = _small_candidates()
    probe = StimulusAction(electrode=0, omega=0.31, phase=0.2, amplitude=0.05, duration=20)
    discovery = discover_active(
        ToyMatter(),
        candidates,
        wait_steps=12,
        probe=probe,
        budget=7,
        codes=3,
        warmup=3,
    )
    assert discovery.probe_budget == 7
    assert len(discovery.probed_indices) == 7
    assert len(set(discovery.probed_indices)) == 7
    assert len(discovery.codebook.actions) == 3
