from frequency_modal_cell.axon_attacker_experiment import run_v2


def test_v2_fixed_delay_attacker_sets_the_expected_boundary():
    receipt = run_v2(seed=23)
    assert receipt["experiment"] == "v2_fixed_delay_graph_attacker"
    assert receipt["attacker"]["edge_count"] == 9
    assert receipt["gate"]["fixed_graph_exact_when_uncoupled"]
    assert receipt["gate"]["near_coactivity_breaks_fixed_delay"]
    assert receipt["gate"]["spatial_control_suppresses_80pct"]
    assert receipt["gate"]["temporal_control_suppresses_80pct"]
    assert receipt["verdict"] == "PASS_FIXED_DELAY_BOUNDARY"
