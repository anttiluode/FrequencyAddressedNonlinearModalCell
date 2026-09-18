# FrequencyAddressedNonlinearModalCell

> **The field is not the hypothesis anymore. The field is the compressed representation of computational matter.**

This repository asks one narrow question that emerged from the `NotSoSimpleNeuron`, `Entrain`, `InformationFlow`, `Saturday`, `Operaattori`, and `NSSN2` lines:

> Can a rich stateful receiver be compressed into a much smaller dynamical cell that is still addressable by carrier frequency/phase and still preserves the nonlinear history effects that made the original receiver interesting?

The goal is **not** to put a detailed biological dendrite into every artificial neuron. The biological/physical model is the mine. The desired product is a cheap reduced operator that could eventually be used as a trainable primitive at scale.

## Why this is not another oscillator toy

Earlier projects already established several boundaries:

- `SpectralNeuron`: frequency-selective addressing is useful, but by itself it is ordinary FDM.
- `Entrain`: Stuart-Landau resonators can route by entrainment and phase, with a measured Arnold-tongue resolution limit.
- `InformationFlow`: space/frequency/phase can select a nonlinear write, but the useful coupling was explicitly supplied.
- `NotSoSimpleNeuron`: a tuned point resonator beats a quasi-active cable on simple resonance, while the stronger residue is **state-conditioned cross-mode coupling and noncommuting event-order operators**.
- `Operaattori`: complicated morphology can collapse to reusable transport/nonlinear operator objects, but its passive cable is not intrinsically an oscillator bank.
- `NSSN2`: changing a resident machine is not enough; development must beat the frozen birth machine.

This repo reverses the old field-computing direction. Instead of inventing a field and asking what it computes, it starts with richer computational matter and asks what **small field/operator description can be extracted without losing the computation**.


## How this differs from ordinary neuron models

The unusual claim here is **not** merely that biology has dendrites and axons. Detailed compartmental models already know that. The difference is where the computation is allowed to live.

| ordinary abstraction | this repository's working abstraction |
|---|---|
| artificial neuron: `y = sigma(w^T x + b)` | arriving carrier perturbs a resident state-dependent receiver |
| spiking neuron: integrate current, threshold, emit spike | local state/history changes the operator encountered by the next event |
| connection = scalar weight or fixed kernel | effective connection can factor into receiver state, carrier, grown path and delay |
| axon = ideal wire or fixed delay | grown axonal geometry is an output delay/routing operator |
| network topology is specified directly | development can manufacture the topology before runtime uses it |
| coupling is explicit synaptic connectivity | weak local ephaptic timing can depend on nearby axonal traffic |

A useful shorthand is:

```text
normal neural abstraction:
    computation in node + communication in edge

this project:
    receiving matter computes
            +
    grown transmission matter transforms
            +
    the effective connection is assembled at runtime
```

That does **not** mean the physical description automatically earns a better computational primitive. A conventional model is the default attacker. In particular, a frozen branching axon should collapse to a sparse graph with fixed delays. The next gate below tests exactly that boundary.

### v2 — normal-model attacker: sparse graph + fixed delays

The first deliberately boring attacker removes the spatial axon after development and replaces every grown terminal route with one graph edge carrying the measured path delay.

If the axon has no runtime state, the two descriptions should be equivalent:

```text
grown branch geometry  ->  path length / speed
                         ↓
                  fixed graph delay
```

The only residue allowed to survive is **context-dependent timing**: in v1, nearby co-active axons slightly change one another's propagation delay. If the fixed-delay graph matches that too, then the ephaptic story has added no computational consequence.

This is intentionally a reduction test, not a performance benchmark. Even if fixed delays fail under ephaptic coupling, a richer conventional graph with activity-dependent delays could emulate the effect. The question is simply **which pieces of the physical story survive abstraction**.


## v0 — compress a resonant + active receiver

The teacher is deliberately synthetic and transparent. It combines two mechanisms already isolated in `NotSoSimpleNeuron`:

```text
12-compartment passive cable
        +
quasi-active recovery state         -> non-zero resonant response
        +
local voltage-dependent conductance -> state-dependent nonlinear interaction
```

Each compartment therefore carries three state variables:

```text
voltage v_i
recovery w_i
conductance g_i
```

for a **36-state teacher**. Two distributed spatial ports receive signed carrier packets. Positive carrier half-cycles also deposit an excitatory conductance trace; local voltage controls how much of that trace becomes active current.

This is a computational teacher, **not a claim about a fitted biological channel model**.

The reduced cell uses an 8-dimensional POD/PCA coordinate system learned only from training trajectories. Three frozen competitors use exactly the same training probes:

```text
nonlinear_modal             rank-8 POD + quadratic state/input operator inference
linear_modal                same rank-8 POD + linear dynamics
random_projection_nonlinear rank-8 random coordinates + same quadratic library
```

The quadratic regression is intentionally ordinary reduced-order modelling: it is close in spirit to **Operator Inference** and **SINDy**, not a claimed new mathematical method.

### Frozen training / test split

Training:

```text
frequencies  0.18, 0.30, 0.42, 0.54 rad/step
phases       0, pi/2
ports        both
plus three in-distribution two-tone interactions
```

Held out before fitting:

```text
frequencies  0.24, 0.36, 0.48 rad/step
phases       pi/4, 3pi/4
new two-tone combinations and relative phases
A->B versus B->A packet order
```

The v0 pass condition was frozen before the result: the nonlinear modal surrogate had to beat the same-coordinate linear model by at least 15% on held-out single carriers, held-out pairs, and the order probe, while not losing the single-carrier test to the same-rank random-coordinate nonlinear attacker.

## v0 result: FAIL

Committed receipt: [`results/v0.json`](results/v0.json)

The state itself is extremely compressible:

```text
rank-8 POD energy fraction     0.998602
```

And the nonlinear library really does improve the **one-step fit** in POD coordinates:

| model | training one-step NRMSE |
|---|---:|
| nonlinear modal | **0.10428** |
| linear modal | 0.19102 |
| random nonlinear | **0.02816** |

But that local fit does **not** survive autonomous rollout:

| model | unseen single carriers | unseen tone pairs | A/B order-gap error |
|---|---:|---:|---:|
| nonlinear modal | 0.66566 | 0.54811 | 0.66584 |
| **linear modal** | **0.10545** | **0.08203** | **0.13328** |
| random nonlinear | 1.02463 | 0.97686 | 0.97628 |

Verdict:

```text
FAIL_NONLINEAR_MODAL_COMPRESSION
```

The two-tone interaction residual is especially diagnostic. The teacher's nonlinear interaction is real but small in the current separated-port probe (`~1.10e-3` RMS). The linear model correctly predicts essentially none of that residual, yet remains far better on the total held-out trajectory. The naive quadratic reduced model overfits local transitions and compounds error when iterated.

### What v0 killed

It kills the easiest version of today's idea:

> “If the rich receiver lives near a low-dimensional manifold, fit a polynomial law in those modal coordinates and you have the computational substance.”

No. **Low-dimensional state geometry is weaker than a stable reduced computational law.**

That distinction is exactly the `MatrixInMatrix` / algorithm-decoding warning in dynamical form: a compact representation of observed state is not automatically the mechanism that generates future state.

### What survives

The experiment is still useful because it cleanly separates three questions:

1. **State compression:** yes — rank 8 captures 99.86% of sampled variance.
2. **Local dynamic regression:** partially — quadratic features improve one-step prediction.
3. **Autonomous causal reduction:** no — the fitted nonlinear law does not survive unseen rollout.

The next gate therefore should not add more decorative frequencies. It should ask whether a **structure-preserving closure** can retain the teacher's local nonlinear interaction without destabilizing the reduced dynamics, and it should strengthen the interaction witness with overlapping-vs-separated controls before any large-network experiment is attempted.

## Run it

```bash
python -m pip install -e '.[test]'
pytest -q
python -m frequency_modal_cell.experiment --output results/v0.json
```

The run is deterministic at seed 17.

## Repository structure

```text
src/frequency_modal_cell/core.py        teacher + carrier packets + reduced models
src/frequency_modal_cell/experiment.py  frozen v0 training/evaluation/receipt
tests/                                  mechanism and receipt tests
results/v0.json                         committed deterministic result
docs/superpowers/specs/                 frozen design
```

## Prior-art fence

Nothing here establishes a new Lagrangian law or a new reduced-order method.

Relevant established machinery includes:

- Brunton, Proctor & Kutz (2016), **Sparse Identification of Nonlinear Dynamics (SINDy)** — sparse regression over nonlinear candidate functions to identify dynamical laws.
- Peherstorfer & Willcox (2016), **Data-driven operator inference for nonintrusive projection-based model reduction** — infer reduced polynomial operators from high-dimensional trajectories without requiring the full operators.
- POD / projection-based model reduction generally — compress high-dimensional trajectory families into a low-dimensional coordinate system before learning or projecting dynamics.

The narrower research question here is whether those ordinary reduction ideas can extract a **cheap frequency-addressed state-conditioned computational primitive** from the particular resident-matter mechanisms isolated in this repository lineage.

## Claim boundary

This repository does **not** establish that:

- real dendrites are frequency-addressed modal computers;
- a Lagrangian or Hamiltonian is the unique right representation;
- the reduced coordinates are physically unique;
- frequency channels are better than ordinary vector communication;
- the current reduced cell beats GRUs, state-space models, attention, or transformers;
- the v0 nonlinear surrogate works — it explicitly fails its frozen gate.

The live question is smaller:

> **What is the cheapest stable dynamical object that preserves the useful computation of a richer receiving substrate?**


## v1 — branching axonal matter and weak ephaptic timing

The cell now has a second spatial scale after its local receiving dynamics:

```text
local resonant / active cell
          ↓ emitted event train
branching axonal arbor grown through space
          ↓
distant target terminals
          +
weak local ephaptic interaction where nearby axon segments run together
```

The axonal arbor is not an extra dense matrix. A shared growth cone advances toward the target region, then daughter branches inherit the parent direction and grow toward individual distant targets under a small chemoaffinity-like guidance rule. This intentionally rhymes with `GrowingAnttisNeuron`: **development manufactures the route before runtime uses it**.

The ephaptic term is deliberately conservative. It is **not** a tiny wireless synapse. Nearby co-active, similarly oriented axon segments slightly alter one another's propagation delay. That matches the more defensible literature-scale effect: extracellular fields in axon bundles can modulate conduction timing, while ephaptic effects have also been measured/modelled near terminals. The synthetic model therefore changes arrival timing/phase only; a branch must still physically reach a target for information to be delivered.

Frozen v1 receipt: [`results/v1_axonal_matter.json`](results/v1_axonal_matter.json)

At seed 23:

| quantity | value |
|---|---:|
| arbors | 3 |
| targets per arbor | 3 |
| segments per arbor | 47 |
| max target error | 0 |
| near-bundle ephaptic contacts | 1307 |
| mean synchronous delay shift | 0.39349 step |
| mean delay-shift fraction | **1.473%** |
| arrival phase shift at ω=0.42 | **0.16527 rad** |
| far-separated delay shift | 0.00530 step |
| temporally staggered delay shift | 0.00040 step |

The spatial and temporal attackers both suppress the effect by more than 98%, and turning ephaptic gain to zero reproduces the uncoupled timing exactly.

Verdict:

```text
PASS_AXONAL_MATTER_GATE
```

This earns a narrow addition to the picture:

```text
frequency / phase does not only address the receiving cell.

cell state → axonal event train → grown path geometry
                                 ↓
                     weak bundle-dependent delay
                                 ↓
                   changed arrival phase downstream
```

So the **axon itself can become part of the address operator** without carrying a second explicit weight matrix. The strong connection is still the branch topology. The ephaptic field is a weak contextual perturbation supplied by neighboring axonal traffic.

### v1 claim boundary

This is not a calibrated mammalian axon model. The growth law is synthetic and chemoaffinity-like; the conduction speed and ephaptic gain are dimensionless experiment parameters. The result establishes only that the proposed decomposition is executable, local, weak, geometry-dependent, and falsifiable. It does not establish that ephaptic coupling is a major source of biological computation.
