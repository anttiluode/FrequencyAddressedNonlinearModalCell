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


def test_v3_receipt_keeps_fanmc_and_attackers_together():
    receipt = run_v3(seed=31, wait_steps=96)
    assert receipt["interface"]["electrodes"] == 8
    assert set(receipt["arms"]) == {
        "fanmc_hidden_matter",
        "same_fast_matter_no_slow_write",
        "linear_state_space_reservoir",
        "fanmc_spatial_only",
        "fanmc_frequency_phase_only",
    }
    assert set(receipt["address_ablation"]) >= {
        "full_accuracy",
        "spatial_only_accuracy",
        "frequency_phase_only_accuracy",
    }
    assert receipt["verdict"] in {
        "PASS_BLACK_BOX_STIMULATION_LANGUAGE",
        "FAIL_BLACK_BOX_STIMULATION_LANGUAGE",
    }
