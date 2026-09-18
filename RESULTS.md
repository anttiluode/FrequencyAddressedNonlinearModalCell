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
