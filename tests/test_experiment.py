from frequency_modal_cell.experiment import run_v0


def test_v0_receipt_has_preregistered_sections():
    receipt = run_v0(seed=17, train_steps=144, rank=8)
    assert receipt["experiment"] == "v0_frequency_addressed_reduction"
    assert set(receipt["models"]) == {"nonlinear_modal", "linear_modal", "random_projection_nonlinear"}
    assert {"heldout_single", "heldout_pair", "order"} <= set(receipt["metrics"])
    assert receipt["teacher"]["state_dim"] == 36
    assert receipt["models"]["nonlinear_modal"]["state_dim"] == 8
    assert receipt["verdict"] in {
        "PASS_NONLINEAR_MODAL_COMPRESSION",
        "FAIL_NONLINEAR_MODAL_COMPRESSION",
    }
