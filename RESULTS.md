# v0 receipt — state compresses, naive nonlinear closure does not

The frozen v0 experiment compresses a 36-state resonant + active cable teacher into rank-8 models and evaluates them on carrier frequencies, phases, tone pairs and event order not used to fit the dynamics.

## Result

- rank-8 POD energy fraction: **0.998602**
- nonlinear modal training one-step NRMSE: **0.10428**
- linear modal training one-step NRMSE: **0.19102**
- nonlinear modal held-out single-carrier NRMSE: **0.66566**
- linear modal held-out single-carrier NRMSE: **0.10545**
- nonlinear modal held-out pair NRMSE: **0.54811**
- linear modal held-out pair NRMSE: **0.08203**
- nonlinear modal A/B order-gap NRMSE: **0.66584**
- linear modal A/B order-gap NRMSE: **0.13328**

`FAIL_NONLINEAR_MODAL_COMPRESSION`

The important boundary is that **state compressibility is not dynamic-law compressibility**. The quadratic operator-inference model fits local transitions better, then accumulates large rollout error. The linear reduced model misses the tiny explicit nonlinear interaction residual but predicts the total trajectory much better in the current regime.

No post-result rescue tuning is included in v0.


# v1 receipt — axonal geometry routes; ephaptic coupling perturbs timing

The first axonal-matter gate adds three grown branching arbors, each reaching three distant targets. Weak local ephaptic coupling is applied only between nearby, similarly oriented, co-active segments and changes conduction delay rather than creating new target connections.

- mean synchronous delay shift: **0.39349 step**
- mean shift fraction: **1.473%**
- mean arrival-phase shift at ω=0.42: **0.16527 rad**
- far-separated control: **0.00530 step**
- temporally staggered control: **0.00040 step**

`PASS_AXONAL_MATTER_GATE`

The useful distinction is now: **branch topology carries the strong route; local axonal field interactions weakly modulate when that route delivers its event.**


# v2 receipt — branching collapses to a graph; runtime axonal context does not

A sparse fixed-delay graph with one edge per terminal reproduces the uncoupled grown arbor with **zero arrival-time error**. This kills any claim that frozen branching geometry alone earns a new computational primitive.

When weak ephaptic timing is enabled for nearby synchronous axons, the fixed-delay attacker misses arrival times by **0.40235 step RMS**, or **0.16899 rad RMS** at the test carrier. Spatial separation reduces the mismatch to **0.00801** and temporal staggering to **0.00044**.

`PASS_FIXED_DELAY_BOUNDARY`

The surviving distinction is therefore not “biology has branching axons.” It is narrower: **transmission matter can make an otherwise fixed edge operator depend weakly on current local traffic.** A conventional graph with activity-dependent delays remains a valid stronger attacker.


# v3 receipt — learn a language for hidden matter

v3 wraps a hidden FANMC population behind eight stimulation/recording ports. The controller sees no cell states, material variables, recurrent weights or gradients. It calibrates an external write vocabulary, waits 96 steps, then reads written state using one common probe.

- full FANMC four-code accuracy: **1.0000**
- slow-write ablation: **0.1875**
- 48-state linear state-space attacker: **0.9792**
- space-only FANMC: **1.0000**
- frequency+phase only at one electrode: **0.9167**
- frequency only at one electrode: **0.8125**
- phase only at one electrode: **0.2708**
- persistent FANMC write-vs-baseline distance: **0.006396**
- fast-only write-vs-baseline distance: **2.86e-6**

`PASS_BLACK_BOX_STIMULATION_LANGUAGE`

This earns an interface result, not an architecture victory. Slow hidden state is necessary for persistent FANMC writes; frequency is a real one-port address coordinate in this task; but space is easier and a boring linear state-space reservoir nearly matches the full codebook.
