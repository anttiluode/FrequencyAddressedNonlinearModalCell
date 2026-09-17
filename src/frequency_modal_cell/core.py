from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def passive_cable_operator(n: int, leak: float = 0.08, coupling: float = 0.18) -> np.ndarray:
    """Stable nearest-neighbour passive cable update used by the NSSN lineage."""
    lap = np.zeros((n, n), dtype=float)
    for i in range(n):
        if i > 0:
            lap[i, i - 1] = -1.0
            lap[i, i] += 1.0
        if i < n - 1:
            lap[i, i + 1] = -1.0
            lap[i, i] += 1.0
    op = np.eye(n) - leak * np.eye(n) - coupling * lap
    if float(np.max(np.abs(np.linalg.eigvals(op)))) >= 1.0:
        raise ValueError("unstable passive cable operator")
    return op


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


class ResonantActiveCable:
    """A transparent 36-state teacher: cable voltage + recovery + local conductance.

    It deliberately combines two already-tested NotSoSimpleNeuron ingredients:
    quasi-active resonant recovery and a synthetic local voltage-dependent excitatory
    interaction. It is a computational teacher, not a fitted biological cell.
    """

    def __init__(
        self,
        n: int = 12,
        *,
        leak: float = 0.08,
        coupling: float = 0.18,
        recovery_gain: float = 0.48,
        recovery_decay: float = 0.80,
        recovery_coupling: float = 0.15,
        conductance_decay: float = 0.72,
        active_gain: float = 0.42,
        threshold: float = 0.055,
        slope: float = 0.022,
        reversal: float = 0.75,
    ) -> None:
        self.n = int(n)
        self.operator = passive_cable_operator(self.n, leak=leak, coupling=coupling)
        self.recovery_gain = float(recovery_gain)
        self.recovery_decay = float(recovery_decay)
        self.recovery_coupling = float(recovery_coupling)
        self.conductance_decay = float(conductance_decay)
        self.active_gain = float(active_gain)
        self.threshold = float(threshold)
        self.slope = float(slope)
        self.reversal = float(reversal)

        # Two spatially distinct, slightly distributed receiver ports.
        x = np.arange(self.n, dtype=float)
        centers = (0.28 * (self.n - 1), 0.72 * (self.n - 1))
        cols = []
        for center in centers:
            p = np.exp(-0.5 * ((x - center) / 1.05) ** 2)
            p /= np.linalg.norm(p)
            cols.append(p)
        self.port_matrix = np.column_stack(cols)
        self.reset()

    @property
    def state_dim(self) -> int:
        return 3 * self.n

    def reset(self, state: np.ndarray | None = None) -> np.ndarray:
        if state is None:
            self.v = np.zeros(self.n, dtype=float)
            self.w = np.zeros(self.n, dtype=float)
            self.g = np.zeros(self.n, dtype=float)
        else:
            state = np.asarray(state, dtype=float)
            if state.shape != (self.state_dim,):
                raise ValueError(f"state must have shape ({self.state_dim},)")
            self.v = state[: self.n].copy()
            self.w = state[self.n : 2 * self.n].copy()
            self.g = state[2 * self.n :].copy()
        return self.state.copy()

    @property
    def state(self) -> np.ndarray:
        return np.concatenate([self.v, self.w, self.g])

    def step(self, port_signal: np.ndarray) -> np.ndarray:
        port_signal = np.asarray(port_signal, dtype=float)
        if port_signal.shape != (2,):
            raise ValueError("port_signal must have shape (2,)")
        drive = self.port_matrix @ port_signal

        old_v = self.v.copy()
        old_w = self.w.copy()
        # Conductance is excitatory: signed carrier affects voltage, positive half-cycle
        # deposits a local conductance trace. This keeps frequency/phase meaningful while
        # avoiding a fictitious negative conductance.
        self.g = self.conductance_decay * self.g + np.maximum(drive, 0.0)
        pre = self.operator @ old_v - self.recovery_gain * old_w + drive
        gate = _sigmoid((pre - self.threshold) / self.slope)
        headroom = np.maximum(self.reversal - pre, 0.0)
        active_current = self.active_gain * self.g * gate * headroom
        self.v = pre + active_current
        self.w = self.recovery_decay * old_w + self.recovery_coupling * old_v

        state = self.state
        if not np.isfinite(state).all() or np.max(np.abs(state)) > 50.0:
            raise FloatingPointError("teacher left its bounded operating regime")
        return state.copy()

    def simulate(self, inputs: np.ndarray, initial_state: np.ndarray | None = None) -> np.ndarray:
        inputs = np.asarray(inputs, dtype=float)
        if inputs.ndim != 2 or inputs.shape[1] != 2:
            raise ValueError("inputs must be [time, 2]")
        self.reset(initial_state)
        states = [self.state.copy()]
        for u in inputs:
            states.append(self.step(u))
        return np.asarray(states)


def carrier_packet(
    steps: int,
    *,
    omega: float,
    phase: float = 0.0,
    amplitude: float = 0.10,
    port: int = 0,
    n_ports: int = 2,
    start: int = 0,
    stop: int | None = None,
    taper: int = 6,
) -> np.ndarray:
    """Create a finite sinusoidal carrier packet addressed to one input port."""
    if port < 0 or port >= n_ports:
        raise ValueError("invalid port")
    stop = steps if stop is None else min(int(stop), steps)
    start = max(int(start), 0)
    out = np.zeros((steps, n_ports), dtype=float)
    if stop <= start:
        return out
    idx = np.arange(stop - start, dtype=float)
    env = np.ones(stop - start, dtype=float)
    taper = min(int(taper), (stop - start) // 2)
    if taper > 0:
        ramp = np.sin(np.linspace(0.0, np.pi / 2.0, taper, endpoint=False)) ** 2
        env[:taper] = ramp
        env[-taper:] = ramp[::-1]
    out[start:stop, port] = amplitude * env * np.sin(omega * idx + phase)
    return out


def combine_packets(*packets: np.ndarray) -> np.ndarray:
    if not packets:
        raise ValueError("at least one packet required")
    base = np.zeros_like(np.asarray(packets[0], dtype=float))
    for packet in packets:
        packet = np.asarray(packet, dtype=float)
        if packet.shape != base.shape:
            raise ValueError("packet shapes differ")
        base += packet
    return base


@dataclass
class ReducedModel:
    rank: int
    state_dim: int
    mean: np.ndarray
    basis: np.ndarray
    z_scale: np.ndarray
    u_scale: np.ndarray
    coef: np.ndarray
    nonlinear: bool
    ridge: float
    z_clip: np.ndarray
    basis_kind: str = "pod"

    @property
    def parameter_count(self) -> int:
        return int(self.coef.size)


def _features(z: np.ndarray, u: np.ndarray, nonlinear: bool) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    u = np.asarray(u, dtype=float)
    pieces = [np.array([1.0]), z, u]
    if nonlinear:
        # Symmetric quadratic latent terms and state-input interactions. This is a
        # small polynomial operator-inference library, not a generic neural net.
        q = [z[i] * z[j] for i in range(z.size) for j in range(i, z.size)]
        zu = [z[i] * u[j] for i in range(z.size) for j in range(u.size)]
        pieces.extend([np.asarray(q), np.asarray(zu)])
    return np.concatenate(pieces)


def _ridge_solve(phi: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    gram = phi.T @ phi
    reg = ridge * np.eye(gram.shape[0])
    reg[0, 0] = 0.0
    return np.linalg.solve(gram + reg, phi.T @ target)


def fit_reduced_model(
    traces: Sequence[np.ndarray],
    inputs: Sequence[np.ndarray],
    *,
    rank: int = 8,
    nonlinear: bool = True,
    ridge: float = 1e-4,
    basis: np.ndarray | None = None,
    basis_kind: str = "pod",
    rng: np.random.Generator | None = None,
) -> ReducedModel:
    if len(traces) != len(inputs) or not traces:
        raise ValueError("traces and inputs must be non-empty and matched")
    state_dim = int(np.asarray(traces[0]).shape[1])
    all_states = np.concatenate([np.asarray(t, dtype=float)[:-1] for t in traces], axis=0)
    mean = all_states.mean(axis=0)

    if basis is None:
        if basis_kind == "pod":
            _, _, vt = np.linalg.svd(all_states - mean, full_matrices=False)
            basis = vt[:rank].T
        elif basis_kind == "random":
            rng = np.random.default_rng(0) if rng is None else rng
            q, _ = np.linalg.qr(rng.normal(size=(state_dim, rank)))
            basis = q[:, :rank]
        else:
            raise ValueError("basis_kind must be 'pod' or 'random'")
    basis = np.asarray(basis, dtype=float)
    if basis.shape != (state_dim, rank):
        raise ValueError("basis shape mismatch")

    z_train = (all_states - mean) @ basis
    z_scale = np.maximum(z_train.std(axis=0), 1e-6)
    all_u = np.concatenate([np.asarray(u, dtype=float) for u in inputs], axis=0)
    u_scale = np.maximum(all_u.std(axis=0), 1e-6)

    rows = []
    targets = []
    for trace, drive in zip(traces, inputs):
        trace = np.asarray(trace, dtype=float)
        drive = np.asarray(drive, dtype=float)
        z0 = ((trace[:-1] - mean) @ basis) / z_scale
        z1 = ((trace[1:] - mean) @ basis) / z_scale
        us = drive / u_scale
        rows.extend(_features(z, u, nonlinear) for z, u in zip(z0, us))
        targets.append(z1)
    phi = np.asarray(rows)
    target = np.concatenate(targets, axis=0)
    coef = _ridge_solve(phi, target, ridge)
    z_clip = np.maximum(np.max(np.abs(z_train / z_scale), axis=0) * 1.35, 3.0)
    return ReducedModel(
        rank=rank,
        state_dim=state_dim,
        mean=mean,
        basis=basis,
        z_scale=z_scale,
        u_scale=u_scale,
        coef=coef,
        nonlinear=nonlinear,
        ridge=ridge,
        z_clip=z_clip,
        basis_kind=basis_kind,
    )


def rollout_reduced(model: ReducedModel, inputs: np.ndarray, initial_state: np.ndarray) -> np.ndarray:
    inputs = np.asarray(inputs, dtype=float)
    x0 = np.asarray(initial_state, dtype=float)
    z = ((x0 - model.mean) @ model.basis) / model.z_scale
    states = [model.mean + model.basis @ (z * model.z_scale)]
    for u in inputs:
        us = u / model.u_scale
        z = _features(z, us, model.nonlinear) @ model.coef
        # v0 is explicitly local to the sampled operating envelope. Bounded latent
        # coordinates prevent numerical blow-up from being mistaken for the only
        # scientific failure mode; rollout fidelity is still measured after clipping.
        z = np.clip(z, -model.z_clip, model.z_clip)
        states.append(model.mean + model.basis @ (z * model.z_scale))
    return np.asarray(states)


def nrmse(target: np.ndarray, pred: np.ndarray) -> float:
    target = np.asarray(target, dtype=float)
    pred = np.asarray(pred, dtype=float)
    scale = float(np.sqrt(np.mean(target**2))) + 1e-12
    return float(np.sqrt(np.mean((pred - target) ** 2)) / scale)


def observable(trace: np.ndarray, n: int = 12) -> np.ndarray:
    """Three fixed voltage observations: proximal, middle, distal."""
    idx = [2, n // 2, n - 3]
    return np.asarray(trace)[:, idx]
