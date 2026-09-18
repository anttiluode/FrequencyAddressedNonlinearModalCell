import numpy as np

from frequency_modal_cell.virtual_organoid import (
    LinearPortReservoir,
    StimulusAction,
    VirtualOrganoid,
    evaluate_codebook,
    learn_stimulation_codebook,
)
from frequency_modal_cell.virtual_organoid_experiment import run_v3


def test_virtual_organoid_exposes_only_port_level_shapes():
    matter = VirtualOrganoid(n_cells=4, cell_compartments=6, seed=2)
    action = StimulusAction(electrode=2, omega=0.30, phase=0.2, amplitude=0.08, duration=24)
    matter.stimulate(action)
    matter.wait(12)
    signature = matter.probe_signature(
        StimulusAction(electrode=0, omega=0.36, phase=0.0, amplitude=0.05, duration=32)
    )
    assert matter.n_electrodes == 8
    assert signature.shape == (16,)
    assert np.isfinite(signature).all()


def test_codebook_learning_uses_same_interface_for_boring_reservoir():
    candidates = [
        StimulusAction(electrode=e, omega=w, phase=0.0, amplitude=0.10, duration=32)
        for e in range(4)
        for w in (0.20, 0.36)
    ]
    probe = StimulusAction(electrode=0, omega=0.30, phase=0.1, amplitude=0.05, duration=32)
    matter = LinearPortReservoir(n_electrodes=8, state_dim=24, seed=3)
    codebook = learn_stimulation_codebook(matter, candidates, wait_steps=24, probe=probe, codes=3)
    result = evaluate_codebook(
        matter,
        codebook,
        wait_steps=24,
        probe=probe,
        repeats=2,
        observation_noise=1e-5,
        seed=4,
    )
    assert len(codebook.actions) == 3
    assert 0.0 <= result["accuracy"] <= 1.0


def test_frequency_and_phase_can_be_presented_through_same_black_box_interface():
    matter = VirtualOrganoid(n_cells=4, cell_compartments=6, seed=5)
    probe = StimulusAction(electrode=0, omega=0.31, phase=0.1, amplitude=0.05, duration=24)
    candidates = [
        StimulusAction(electrode=0, omega=omega, phase=phase, amplitude=0.10, duration=28)
        for omega, phase in ((0.18, 0.0), (0.30, 0.0), (0.42, 0.0), (0.30, np.pi / 2.0))
    ]
    codebook = learn_stimulation_codebook(matter, candidates, wait_steps=20, probe=probe, codes=3)
    assert len(codebook.actions) == 3
    assert all(action.electrode == 0 for action in codebook.actions)
    assert len({(round(action.omega, 6), round(action.phase, 6)) for action in codebook.actions}) == 3
