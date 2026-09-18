import numpy as np

from frequency_modal_cell.axon import ephaptic_contacts, grow_branching_arbor, terminal_arrival_times
from frequency_modal_cell.axon_experiment import run_v1


def test_branching_arbor_reaches_multiple_distant_targets():
    targets = np.array([[1.1, -0.2], [1.2, 0.0], [1.1, 0.2]])
    arbor = grow_branching_arbor(np.array([0.0, 0.0]), targets, seed=4)
    assert len(arbor.terminal_segment_indices) == 3
    assert arbor.branch_count >= 1
    assert np.max(arbor.target_errors) < 1e-8
    assert arbor.total_length > 1.0


def test_ephaptic_coupling_is_local_weak_and_zeroable():
    a0 = grow_branching_arbor(np.array([0.0, -0.03]), np.array([[1.1, -0.17], [1.2, -0.03], [1.1, 0.11]]), arbor_id=0, seed=2)
    a1 = grow_branching_arbor(np.array([0.0, 0.03]), np.array([[1.1, -0.11], [1.2, 0.03], [1.1, 0.17]]), arbor_id=1, seed=3)
    contacts = ephaptic_contacts([a0, a1])
    assert contacts
    emissions = {0: 0.0, 1: 0.4}
    base = terminal_arrival_times([a0, a1], emissions, ephaptic_gain=0.0, contacts=contacts)
    coupled = terminal_arrival_times([a0, a1], emissions, ephaptic_gain=0.006, contacts=contacts)
    shifts = np.array([base[k] - coupled[k] for k in base])
    assert np.all(shifts >= 0.0)
    assert np.max(shifts) > 0.0
    assert np.max(shifts) < 0.05 * np.mean([base[k] - emissions[k[0]] for k in base])


def test_v1_receipt_has_spatial_and_temporal_attackers():
    receipt = run_v1(seed=23)
    assert receipt["experiment"] == "v1_branching_axonal_matter"
    assert receipt["geometry"]["arbors"] == 3
    assert receipt["gate"]["all_targets_reached"]
    assert "spatial_control_suppresses_80pct" in receipt["gate"]
    assert "temporal_control_suppresses_80pct" in receipt["gate"]
    assert receipt["verdict"] in {"PASS_AXONAL_MATTER_GATE", "FAIL_AXONAL_MATTER_GATE"}
