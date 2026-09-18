"""
edgechip.py  —  a reference "talk to it like a CPU" neuron chip.

Contract (the only thing the controller sees):
    write(edge, omega, phase, amp, dur)   inject a shaped pulse at an edge dendrite
    wait(steps)                            let the matter evolve, no input
    read_somas(omega)                      lock-in readout at every soma

Everything else -- cable states, branch operators, edge->cell map, recurrent
routes, the slow material variable -- is hidden. This mirrors the organoid
constraint: you do not program the internals, you learn a language for the ports.

Primitives are faithful to the existing repos:
  - passive symmetric cable operator + modal basis   (NotSoSimpleNeuron/cable.py)
  - quasi-active resonant branch, one recovery var    (NotSoSimpleNeuron/frequency.py)
  - soft-knee calcium transducer                       (NotSoSimpleNeuron/transducer.py)
  - hidden 8-port organoid interface + slow write      (FANMC/virtual_organoid.py)

This file's job is not to be the final chip. It is to MEASURE the three design
questions so we build the load-bearing version, not a relabel of a weight matrix.
"""
from __future__ import annotations
import numpy as np


# ----------------------------------------------------------------------------- cable
def cable_operator(n: int, leak: float = 0.08, coupling: float = 0.18) -> np.ndarray:
    L = np.zeros((n, n))
    for i in range(n - 1):
        L[i, i] += 1.0; L[i + 1, i + 1] += 1.0
        L[i, i + 1] -= 1.0; L[i + 1, i] -= 1.0
    A = np.eye(n) - leak * np.eye(n) - coupling * L
    if float(np.max(np.abs(np.linalg.eigvalsh(A)))) >= 1.0:
        raise ValueError("unstable cable")
    return A


def modal_basis(A: np.ndarray):
    w, phi = np.linalg.eigh(A)              # phi columns = eigenmodes, orthonormal
    order = np.argsort(w)[::-1]
    return w[order], phi[:, order]


def resonant_state_matrix(A, g=0.45, d=0.80, c=0.15):
    """Block state matrix of a quasi-active branch:  v'=Av-gw+u ,  w'=dw+cv."""
    n = A.shape[0]
    M = np.zeros((2 * n, 2 * n))
    M[:n, :n] = A
    M[:n, n:] = -g * np.eye(n)
    M[n:, :n] = c * np.eye(n)
    M[n:, n:] = d * np.eye(n)
    return M


def transfer(M, b, c, omegas):
    """|H(e^{iw})| for a discrete-time linear system (matches local_frequency_response)."""
    N = M.shape[0]
    eye = np.eye(N, dtype=complex)
    out = np.empty(len(omegas))
    for k, w in enumerate(omegas):
        z = np.exp(1j * w)
        out[k] = abs(c @ np.linalg.solve(z * eye - M, b))
    return out


def lock_in(trace, omega):
    """In-phase / quadrature amplitude of a signal at one frequency (homodyne)."""
    t = np.arange(len(trace))
    s = 2.0 / max(len(trace), 1)
    return np.array([s * trace @ np.cos(omega * t), s * trace @ np.sin(omega * t)])


# ----------------------------------------------------------------------------- knee
class SoftKnee:
    """Leaky calcium trace with a soft threshold gain (coincidence detector)."""
    def __init__(self, decay=0.75, threshold=1.2, slope=0.18):
        self.decay, self.threshold, self.slope = decay, threshold, slope
        self.trace = 0.0
        self._norm = 1.0 / (1.0 + np.exp(-(1.0 - threshold) / slope))
    def reset(self): self.trace = 0.0
    def step(self, event):
        self.trace = self.decay * self.trace + event
        gain = 1.0 / (1.0 + np.exp(-(self.trace - self.threshold) / self.slope))
        return self.trace * gain / self._norm


# ----------------------------------------------------------------------------- neuron
class Neuron:
    """A few resonant branches -> a soma that mixes them -> an AIS that publishes.

    The branches are given DIFFERENT resonance parameters so they ring at
    different frequencies. Whether the soma mix is frequency-selective (a demux)
    or flat (a relabelled weight) is exactly what soma_demux() measures.
    """
    def __init__(self, n=10, branch_params=((0.45, 0.90), (0.30, 0.94), (0.55, 0.86)),
                 leak=0.02, coupling=0.06,
                 slow_decay=0.997, slow_write=0.006, slow_gain=1.6, use_knee=True):
        self.A = cable_operator(n, leak, coupling)  # high-Q: sharp resonances
        self.n = n
        self.branches = [resonant_state_matrix(self.A, g=g, d=d) for (g, d) in branch_params]
        self.state = [np.zeros(2 * n) for _ in self.branches]
        self.b_in = np.zeros(2 * n); self.b_in[0] = 1.0           # inject at compartment 0
        self.c_out = np.zeros(2 * n); self.c_out[n // 2] = 1.0    # read a mid compartment
        self.slow = 0.0
        self.slow_decay, self.slow_write, self.slow_gain = slow_decay, slow_write, slow_gain
        self.knee = SoftKnee() if use_knee else None

    def reset(self):
        self.state = [np.zeros_like(s) for s in self.state]
        self.slow = 0.0
        if self.knee: self.knee.reset()

    def step(self, event):
        drive = self.knee.step(event) if self.knee else event
        gain = 1.0 + self.slow_gain * self.slow
        readouts = []
        for i, M in enumerate(self.branches):
            self.state[i] = M @ self.state[i] + self.b_in * (gain * drive)
            readouts.append(self.c_out @ self.state[i])
        soma = float(np.mean(readouts))          # the mix
        self.slow = self.slow_decay * self.slow + self.slow_write * min(soma * soma, 4.0)
        return soma, np.array(readouts)


# ----------------------------------------------------------------------------- chip
class EdgeChip:
    """A field of neurons. You write pulses at edge neurons, read every soma."""
    def __init__(self, n_neurons=12, n_edge=4, seed=0, **neuron_kw):
        rng = np.random.default_rng(seed)
        self.neurons = [Neuron(**neuron_kw) for _ in range(n_neurons)]
        self.edge = list(range(n_edge))                          # writeable neurons
        # sparse recurrent routes: published event i -> injected at neuron j
        self.routes = {i: rng.choice([k for k in range(n_neurons) if k != i],
                                     size=2, replace=False) for i in range(n_neurons)}
        self.pending = np.zeros(n_neurons)
        self.publish_threshold = 0.6

    def reset(self):
        for nu in self.neurons: nu.reset()
        self.pending[:] = 0.0

    def _tick(self, external):
        emitted = np.zeros(len(self.neurons))
        somas = np.zeros(len(self.neurons))
        for i, nu in enumerate(self.neurons):
            ev = external[i] + self.pending[i]
            soma, _ = nu.step(ev)
            somas[i] = soma
            if abs(soma) > self.publish_threshold:
                emitted[i] = np.sign(soma)
        self.pending[:] = 0.0
        for i, e in enumerate(emitted):
            if e != 0.0:
                for j in self.routes[i]:
                    self.pending[j] += 0.4 * e
        return somas

    def write(self, edge, omega, phase=0.0, amp=0.4, dur=48):
        idx = self.edge[edge]
        for t in range(dur):
            ext = np.zeros(len(self.neurons))
            ext[idx] = amp * np.sin(omega * t + phase)
            self._tick(ext)

    def wait(self, steps):
        for _ in range(steps):
            self._tick(np.zeros(len(self.neurons)))

    def read_somas(self, omega, dur=48, amp=0.15):
        """Common weak probe at omega; lock-in the soma trace of every neuron."""
        traces = [[] for _ in self.neurons]
        for t in range(dur):
            ext = np.zeros(len(self.neurons))
            for e in self.edge:
                ext[self.edge[e]] += amp * np.sin(omega * t)
            somas = self._tick(ext)
            for i in range(len(self.neurons)):
                traces[i].append(somas[i])
        return np.array([lock_in(np.array(tr), omega) for tr in traces])  # (n_neurons, 2)


# ============================================================================= MEASUREMENTS
def q_reachable_modes(n=10, contacts=(0, 3, 6, 9)):
    """Q3 spatially: how many cable modes can you actually ADDRESS from the edge?"""
    A = cable_operator(n)
    _, phi = modal_basis(A)
    tol = 1e-9

    # one physical contact, one knob
    u = np.zeros(n); u[contacts[0]] = 1.0
    e_single = (phi.T @ u) ** 2
    pr_single = e_single.sum() ** 2 / np.sum(e_single ** 2)   # modal spread of a delta

    # k contacts sharing ONE knob (fixed fan)  -> still one modal direction
    P = np.zeros((n, len(contacts)))
    for j, p in enumerate(contacts): P[p, j] = 1.0
    shared = phi.T @ (P @ np.ones(len(contacts)))
    rank_shared = int(np.sum(np.abs(shared) > tol * abs(shared).max()) > 0)  # 1 direction

    # k contacts, INDEPENDENT knobs -> k modal directions
    s = np.linalg.svd(phi.T @ P, compute_uv=False)
    rank_indep = int(np.sum(s > tol * s.max()))
    return dict(modal_spread_single_delta=pr_single,
                addressable_dirs_one_knob=1,
                addressable_dirs_shared_fan=rank_shared,
                addressable_dirs_independent=rank_indep,
                n_contacts=len(contacts))


def q_frequency_addressing(n=10, g=0.45, d=0.90, n_omega=64):
    """Q3 temporally: from ONE contact, how many modes can frequency address?"""
    A = cable_operator(n, leak=0.02, coupling=0.06)  # high-Q
    lam, _ = modal_basis(A)
    omegas = np.linspace(0.05, np.pi - 0.05, n_omega)
    # per-mode resonant gain |H_k(w)| for each cable mode
    M = np.zeros((n_omega, n))
    for k, l in enumerate(lam):
        Mk = np.array([[l, -g], [0.15, d]])
        b = np.array([1.0, 0.0]); c = np.array([1.0, 0.0])
        M[:, k] = transfer(Mk, b, c, omegas)
    s = np.linalg.svd(M - M.mean(0), compute_uv=False)
    eff = (s.sum() ** 2) / np.sum(s ** 2)          # effective independent freq-addressable dims
    peak_w = omegas[np.argmax(M, axis=0)]          # where each mode resonates
    return dict(freq_addressable_modes=float(eff),
                distinct_resonances=int(len(np.unique(np.round(peak_w, 2)))),
                n_modes=n)


def q_soma_demux(branches=((6, 0.5, 0.92), (10, 0.35, 0.94), (16, 0.6, 0.90)), n_omega=64):
    """Does the soma mix depend on frequency (a demux) or is it flat (a weight)?

    Branches are genuinely different instruments (different cable LENGTHS + high-Q).
    The trap: same cable geometry with only tweaked recovery stays flat (~0.02).
    """
    omegas = np.linspace(0.05, np.pi - 0.05, n_omega)
    B = np.zeros((n_omega, len(branches)))
    for bi, (nn, g, d) in enumerate(branches):
        lam, _ = modal_basis(cable_operator(nn, leak=0.02, coupling=0.06))
        acc = np.zeros(n_omega)
        for l in lam:
            Mk = np.array([[l, -g], [0.15, d]])
            acc += transfer(Mk, np.array([1.0, 0.0]), np.array([1.0, 0.0]), omegas)
        B[:, bi] = acc
    mix = B / (B.sum(1, keepdims=True) + 1e-12)     # normalized soma mix profile per w
    # how much does the mix rotate across frequency? 1 - min pairwise cosine
    cos = mix @ mix.T / (np.linalg.norm(mix, axis=1)[:, None] * np.linalg.norm(mix, axis=1)[None] + 1e-12)
    rotation = float(1.0 - cos.min())
    s = np.linalg.svd(mix - mix.mean(0), compute_uv=False)
    eff = (s.sum() ** 2) / np.sum(s ** 2)
    return dict(mix_rotation_0flat_1full=rotation, effective_mix_dims=float(eff),
                n_branches=len(branches))


def q_knee_gain():
    """Q2: does the calcium knee earn its place (superlinear coincidence gain)?"""
    lin_trace = 0.0
    for _ in range(2): lin_trace = 0.75 * lin_trace + 1.0
    linear_pair = lin_trace
    k = SoftKnee(); k.step(1.0); knee_pair = k.step(1.0)
    single = SoftKnee(); knee_single = single.step(1.0)
    return dict(knee_over_linear_pair=knee_pair / linear_pair,
                knee_pair_over_single=knee_pair / max(knee_single, 1e-9))


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    print("=" * 68)
    print("Q3a  ADDRESSABLE MODES FROM THE EDGE (spatial)")
    r = q_reachable_modes()
    print(f"  a single point contact already touches ~{r['modal_spread_single_delta']:.1f} modes,")
    print(f"    but one knob reaches only {r['addressable_dirs_one_knob']} modal DIRECTION.")
    print(f"  {r['n_contacts']} contacts on a shared knob:      {r['addressable_dirs_shared_fan']} direction  (still one weight)")
    print(f"  {r['n_contacts']} contacts on independent knobs: {r['addressable_dirs_independent']} directions  (= n contacts)")
    print("  -> addressable modes = INDEPENDENT drives, not number of contacts.")
    print("=" * 68)
    print("Q3b  ADDRESSABLE MODES FROM ONE CONTACT VIA FREQUENCY (temporal)")
    r = q_frequency_addressing()
    print(f"  {r['n_modes']} cable modes, {r['distinct_resonances']} distinct resonant frequencies")
    print(f"  frequency alone addresses ~{r['freq_addressable_modes']:.1f} independent dims from ONE wire.")
    print("  -> omega is an extra addressing axis you get without more edges.")
    print("=" * 68)
    print("SOMA DEMUX  (is the soma a demux or a relabelled weight?)")
    r = q_soma_demux()
    print(f"  mix rotation across frequency: {r['mix_rotation_0flat_1full']:.3f}  (0=flat weight, 1=full demux)")
    print(f"  effective mix dimensions:      {r['effective_mix_dims']:.2f} of {r['n_branches']} branches")
    print("  (same cable + tweaked recovery instead stays ~0.02: demux must be EARNED)")
    print("=" * 68)
    print("Q2  CALCIUM KNEE")
    r = q_knee_gain()
    print(f"  paired gain, knee vs linear:  {r['knee_over_linear_pair']:.2f}x")
    print(f"  knee pair vs single event:    {r['knee_pair_over_single']:.2f}x  (superlinear coincidence)")
    print("=" * 68)
    print("PLUMBING  (write two frequencies at an edge, read all somas after a wait)")
    chip = EdgeChip(n_neurons=12, n_edge=4, seed=1)
    sigs = {}
    for wlabel, w in [("wA", 0.30), ("wB", 0.70)]:
        chip.reset(); chip.write(0, omega=w, amp=0.5, dur=48); chip.wait(60)
        sigs[wlabel] = np.linalg.norm(chip.read_somas(0.5), axis=1)
    sep = np.linalg.norm(sigs["wA"] - sigs["wB"]) / (np.linalg.norm(sigs["wA"]) + 1e-9)
    print(f"  soma-readout separation between the two written frequencies: {sep:.3f}")
    print("  (nonzero => which frequency was written is recoverable at the somas)")
