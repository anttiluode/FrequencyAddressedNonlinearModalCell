from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .core import (
    ResonantActiveCable,
    _features,
    carrier_packet,
    combine_packets,
    fit_reduced_model,
    nrmse,
    observable,
    rollout_reduced,
)


def _simulate(teacher: ResonantActiveCable, drive: np.ndarray) -> np.ndarray:
    return teacher.simulate(drive)


def _single_bank(steps: int, freqs, phases, amp: float = 0.095):
    bank = []
    for port in (0, 1):
        for omega in freqs:
            for phase in phases:
                bank.append(carrier_packet(steps, omega=omega, phase=phase, amplitude=amp, port=port))
    return bank


def _pair(steps: int, wa: float, wb: float, phase_b: float, amp: float = 0.068) -> np.ndarray:
    return combine_packets(
        carrier_packet(steps, omega=wa, phase=0.0, amplitude=amp, port=0),
        carrier_packet(steps, omega=wb, phase=phase_b, amplitude=amp, port=1),
    )


def _order_drive(steps: int, first: tuple[int, float], second: tuple[int, float], amp: float = 0.115):
    half = steps // 2
    gap = 6
    port_a, wa = first
    port_b, wb = second
    return combine_packets(
        carrier_packet(steps, omega=wa, phase=0.25, amplitude=amp, port=port_a, start=4, stop=half - gap),
        carrier_packet(steps, omega=wb, phase=0.90, amplitude=amp, port=port_b, start=half + gap, stop=steps - 4),
    )


def _training_one_step_nrmse(model, traces, drives):
    errs = []
    targets = []
    for trace, drive in zip(traces, drives):
        z0 = ((trace[:-1] - model.mean) @ model.basis) / model.z_scale
        z1 = ((trace[1:] - model.mean) @ model.basis) / model.z_scale
        us = drive / model.u_scale
        pred = np.asarray([_features(z, u, model.nonlinear) @ model.coef for z, u in zip(z0, us)])
        errs.append((pred - z1).reshape(-1))
        targets.append(z1.reshape(-1))
    err = np.concatenate(errs)
    target = np.concatenate(targets)
    return float(np.sqrt(np.mean(err**2)) / (np.sqrt(np.mean(target**2)) + 1e-12))


def _pod_energy_fraction(traces, rank):
    states = np.concatenate([trace[:-1] for trace in traces], axis=0)
    states = states - states.mean(axis=0)
    singular = np.linalg.svd(states, compute_uv=False)
    return float(np.sum(singular[:rank] ** 2) / np.sum(singular**2))


def _eval_bank(teacher, models, bank):
    by_model = {name: [] for name in models}
    for drive in bank:
        target = _simulate(teacher, drive)
        y = observable(target, teacher.n)
        for name, model in models.items():
            pred = rollout_reduced(model, drive, target[0])
            by_model[name].append(nrmse(y, observable(pred, teacher.n)))
    return {name: float(np.median(vals)) for name, vals in by_model.items()}


def _interaction_metric(teacher, models, steps: int):
    pairs = [(0.24, 0.36, np.pi / 3), (0.36, 0.48, np.pi), (0.24, 0.48, np.pi / 2)]
    out = {name: [] for name in models}
    teacher_ref = []
    for wa, wb, ph in pairs:
        ua = carrier_packet(steps, omega=wa, phase=0.0, amplitude=0.068, port=0)
        ub = carrier_packet(steps, omega=wb, phase=ph, amplitude=0.068, port=1)
        up = ua + ub
        ta, tb, tp = (_simulate(teacher, u) for u in (ua, ub, up))
        base = _simulate(teacher, np.zeros_like(up))
        true_inter = observable(tp, teacher.n) - observable(ta, teacher.n) - observable(tb, teacher.n) + observable(base, teacher.n)
        teacher_ref.append(float(np.sqrt(np.mean(true_inter**2))))
        for name, model in models.items():
            pa = rollout_reduced(model, ua, ta[0])
            pb = rollout_reduced(model, ub, tb[0])
            pp = rollout_reduced(model, up, tp[0])
            p0 = rollout_reduced(model, np.zeros_like(up), base[0])
            pred_inter = observable(pp, teacher.n) - observable(pa, teacher.n) - observable(pb, teacher.n) + observable(p0, teacher.n)
            out[name].append(nrmse(true_inter, pred_inter))
    return {
        "teacher_interaction_rms": float(np.median(teacher_ref)),
        "model_nrmse": {name: float(np.median(vals)) for name, vals in out.items()},
    }


def _order_metric(teacher, models, steps: int):
    ab = _order_drive(steps, (0, 0.24), (1, 0.44))
    ba = _order_drive(steps, (1, 0.44), (0, 0.24))
    ta = _simulate(teacher, ab)
    tb = _simulate(teacher, ba)
    true_gap = observable(ta, teacher.n) - observable(tb, teacher.n)
    model_err = {}
    model_gap_rms = {}
    for name, model in models.items():
        pa = rollout_reduced(model, ab, ta[0])
        pb = rollout_reduced(model, ba, tb[0])
        gap = observable(pa, teacher.n) - observable(pb, teacher.n)
        model_err[name] = nrmse(true_gap, gap)
        model_gap_rms[name] = float(np.sqrt(np.mean(gap**2)))
    return {
        "teacher_gap_rms": float(np.sqrt(np.mean(true_gap**2))),
        "model_gap_rms": model_gap_rms,
        "model_nrmse": model_err,
    }


def run_v0(seed: int = 17, train_steps: int = 192, rank: int = 8) -> dict:
    rng = np.random.default_rng(seed)
    teacher = ResonantActiveCable(n=12)

    train_freqs = (0.18, 0.30, 0.42, 0.54)
    train_phases = (0.0, np.pi / 2)
    train_drives = _single_bank(train_steps, train_freqs, train_phases)
    train_drives += [
        _pair(train_steps, 0.18, 0.42, 0.0),
        _pair(train_steps, 0.30, 0.54, np.pi / 2),
        _pair(train_steps, 0.18, 0.54, np.pi),
    ]
    train_traces = [_simulate(teacher, drive) for drive in train_drives]

    nonlinear = fit_reduced_model(train_traces, train_drives, rank=rank, nonlinear=True, ridge=2e-3)
    linear = fit_reduced_model(
        train_traces,
        train_drives,
        rank=rank,
        nonlinear=False,
        ridge=2e-3,
        basis=nonlinear.basis,
        basis_kind="pod",
    )
    random_model = fit_reduced_model(
        train_traces,
        train_drives,
        rank=rank,
        nonlinear=True,
        ridge=2e-3,
        basis_kind="random",
        rng=rng,
    )
    models = {
        "nonlinear_modal": nonlinear,
        "linear_modal": linear,
        "random_projection_nonlinear": random_model,
    }

    heldout_single = _single_bank(
        train_steps,
        freqs=(0.24, 0.36, 0.48),
        phases=(np.pi / 4, 3 * np.pi / 4),
        amp=0.095,
    )
    single_metrics = _eval_bank(teacher, models, heldout_single)

    heldout_pairs = [
        _pair(train_steps, 0.24, 0.36, np.pi / 3),
        _pair(train_steps, 0.36, 0.48, np.pi),
        _pair(train_steps, 0.24, 0.48, np.pi / 2),
    ]
    pair_metrics = _eval_bank(teacher, models, heldout_pairs)
    interaction = _interaction_metric(teacher, models, train_steps)
    order = _order_metric(teacher, models, train_steps)
    one_step = {name: _training_one_step_nrmse(model, train_traces, train_drives) for name, model in models.items()}
    pod_energy = _pod_energy_fraction(train_traces, rank)

    passes = {
        "single_vs_linear_15pct": single_metrics["nonlinear_modal"] <= 0.85 * single_metrics["linear_modal"],
        "pair_vs_linear_15pct": pair_metrics["nonlinear_modal"] <= 0.85 * pair_metrics["linear_modal"],
        "order_vs_linear_15pct": order["model_nrmse"]["nonlinear_modal"] <= 0.85 * order["model_nrmse"]["linear_modal"],
        "single_vs_random": single_metrics["nonlinear_modal"] <= single_metrics["random_projection_nonlinear"],
        "finite_local_model": bool(np.isfinite(nonlinear.coef).all()),
    }
    verdict = "PASS_NONLINEAR_MODAL_COMPRESSION" if all(passes.values()) else "FAIL_NONLINEAR_MODAL_COMPRESSION"

    return {
        "experiment": "v0_frequency_addressed_reduction",
        "seed": seed,
        "teacher": {
            "name": "resonant_active_cable",
            "compartments": teacher.n,
            "state_dim": teacher.state_dim,
            "ports": 2,
            "provenance": "NotSoSimpleNeuron quasi-active frequency gate + synthetic local active interaction",
        },
        "training": {
            "steps_per_probe": train_steps,
            "frequencies_rad_per_step": list(train_freqs),
            "phases_rad": list(train_phases),
            "probe_count": len(train_drives),
            "pod_energy_fraction_rank8": pod_energy,
        },
        "models": {
            name: {
                "state_dim": model.rank,
                "basis": model.basis_kind,
                "nonlinear": model.nonlinear,
                "parameter_count": model.parameter_count,
            }
            for name, model in models.items()
        },
        "metrics": {
            "training_one_step": one_step,
            "heldout_single": single_metrics,
            "heldout_pair": pair_metrics,
            "interaction": interaction,
            "order": order,
        },
        "gate": passes,
        "verdict": verdict,
        "claim_boundary": (
            "This tests local reduced-order fidelity inside one synthetic teacher family. "
            "It is not evidence of a biological Lagrangian, unique morphology advantage, "
            "or superiority to trained RNN/transformer architectures."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/v0.json"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int, default=192)
    parser.add_argument("--rank", type=int, default=8)
    args = parser.parse_args()
    receipt = run_v0(seed=args.seed, train_steps=args.steps, rank=args.rank)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
