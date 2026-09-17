# Frequency-Addressed Nonlinear Modal Cell v0 Implementation Plan

1. Write tests first for teacher state/ports, carrier addressing, reduced-rank rollout, and receipt schema. Confirm RED because implementation does not exist.
2. Implement a NumPy-only resonant + active cable teacher and finite carrier-packet generator.
3. Implement POD projection, linear and quadratic operator-inference models, deterministic random-projection attacker, bounded local rollout, and trace metrics.
4. Implement the frozen training/held-out probe banks and v0 gate.
5. Run unit tests and deterministic seed-17 v0 receipt. Do not tune after observing the scientific result.
6. Document result, boundaries, established reduced-order prior art, and the next discriminating experiment.
7. Add CI that reruns tests and the deterministic receipt.
8. Add the new repository and the missing cross-family frequency-addressed/compressed-matter bridge to Genealogy without duplicating existing Entrain/SpectralNeuron/InformationFlow/NSSN nodes.
