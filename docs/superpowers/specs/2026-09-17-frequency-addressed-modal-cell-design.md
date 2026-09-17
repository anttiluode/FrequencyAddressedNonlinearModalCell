# Frequency-Addressed Nonlinear Modal Cell — v0 Design

## Goal

Test whether an already-interesting stateful receiving substrate can be compressed into a substantially smaller autonomous dynamical cell without losing its frequency/phase addressing or history-dependent behavior.

The experiment deliberately reverses the older field-computing workflow:

```text
old: choose field law -> ask what it computes
v0:  choose richer teacher matter -> infer small law -> attack the inferred law
```

## Teacher

Use a 12-compartment synthetic cable derived from mechanisms already isolated in `NotSoSimpleNeuron`:

1. stable nearest-neighbor passive cable update;
2. one recovery variable per compartment, producing quasi-active resonance;
3. one decaying conductance trace per compartment;
4. local voltage-dependent gate and saturating reversal current.

Total state dimension: 36.

Two fixed distributed input ports are placed at different cable locations. Carrier packets are described by port, angular frequency, phase, amplitude and finite time support.

The teacher is intentionally not a detailed receptor/channel model. Its purpose is to contain both resonant temporal structure and local state-dependent nonlinear interaction in a transparent fixture.

## Reduction

Collect trajectories from a frozen training bank. Learn an 8-D POD basis from teacher states.

Fit two models in the exact same POD coordinates:

- linear reduced state-space map;
- polynomial nonlinear map with constant, linear state, input, symmetric quadratic state terms, and state-input terms.

Fit a third same-rank nonlinear model in a deterministic random orthogonal projection as a coordinate attacker.

All fits use ridge regression and are frozen before held-out evaluation.

## Training / evaluation separation

Training carrier frequencies: 0.18, 0.30, 0.42, 0.54 rad/step.
Training phases: 0, pi/2.
Both ports plus three two-tone training interactions.

Held out:

- 0.24, 0.36, 0.48 rad/step;
- pi/4 and 3pi/4;
- new tone pairs / relative phases;
- A->B versus B->A packet order.

## Frozen v0 gate

The nonlinear POD surrogate passes only if:

1. held-out single-carrier NRMSE <= 85% of the linear POD NRMSE;
2. held-out pair NRMSE <= 85% of linear;
3. order-gap NRMSE <= 85% of linear;
4. held-out single-carrier NRMSE <= same-rank random nonlinear attacker;
5. fitted coefficients are finite.

Scientific disappointment is a valid result. No result-driven parameter rescue is part of v0.

## Interpretation boundary

A pass would establish only that this teacher family admits a small nonlinear reduced surrogate over the declared operating envelope. It would not establish biological correctness, a unique Lagrangian, morphology advantage, or superiority to standard RNN/SSM/transformer architectures.

A failure separates representation compression from autonomous dynamical reduction and determines the next gate.
