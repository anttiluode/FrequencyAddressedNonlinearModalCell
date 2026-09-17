import numpy as np

from frequency_modal_cell.core import (
    ResonantActiveCable,
    carrier_packet,
    fit_reduced_model,
    rollout_reduced,
)


def test_teacher_has_finite_state_and_two_ports():
    teacher = ResonantActiveCable(n=12)
    x = teacher.reset()
    assert x.shape == (36,)
    assert teacher.port_matrix.shape == (12, 2)
    for _ in range(20):
        x = teacher.step(np.array([0.1, 0.0]))
    assert np.isfinite(x).all()


def test_carrier_packet_supports_frequency_phase_and_port():
    u = carrier_packet(64, omega=0.31, phase=0.7, amplitude=0.12, port=1, n_ports=2)
    assert u.shape == (64, 2)
    assert np.allclose(u[:, 0], 0.0)
    assert np.max(np.abs(u[:, 1])) <= 0.1200001
    assert not np.allclose(u[:, 1], 0.0)


def test_reduced_model_rolls_out_with_lower_state_dimension():
    teacher = ResonantActiveCable(n=12)
    drives = [carrier_packet(120, omega=w, phase=0.0, amplitude=0.08, port=0) for w in (0.18, 0.30, 0.42)]
    traces = [teacher.simulate(u) for u in drives]
    model = fit_reduced_model(traces, drives, rank=8, nonlinear=True, ridge=1e-5)
    assert model.rank == 8
    pred = rollout_reduced(model, drives[0], traces[0][0])
    assert pred.shape == traces[0].shape
    assert np.isfinite(pred).all()
