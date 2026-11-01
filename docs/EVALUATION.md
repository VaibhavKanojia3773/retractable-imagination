# Evaluation protocol

The paper uses saved measurements, not estimated performance. `results/study.json` contains 480 per-episode records, four family summaries, and every backend timing sample. `results/evaluation.json` contains the sensor fit, state-repair kernels, and fixed-route cache control.

## Task study

Seed 20261102; 120 independently sampled episodes per predeclared family:

| Family | Audit | Intended distinction |
|---|---|---|
| False free space | Remove one invalid free-space report | A stale route can cross an occupied passage |
| False obstacle | Remove one invalid occupied-space report | Success can remain unchanged while route cost improves |
| Source packet group | Revoke 3–4 reports from one source | Test a batch audit across multiple factors |
| No corruption | Empty audit | Retraction must leave state unchanged |

Widths, heights, partition/passage coordinates, priors, background measurements, and incorrect response strengths vary. The agent starts with structural map information and uncertain selected passages; this is not mapping from images or SLAM. The observation encoder and planner are shared and frozen. Up to five realized safe actions precede the audit. Ground truth scores routes and enforces execution collisions; it is not fed to the planner.

Full replay, checkpoint replay, explicit subtraction and indexed retraction obtain identical corrected decisions. The stale-cache ablation retains the old route after repairing its state, deliberately illustrating incomplete invalidation; it is not presented as a competitive inference method.

## Backend timing

Seed 20261103; 24 independently generated encoded histories per history length. Method order is randomized. The timer covers revision, 20 route forecasts, and global A*. Every method recomputes the same downstream forecasts for this comparison. Acquisition, encoding, initial history ingestion, browser rendering, and network time are excluded. Report medians and interquartile ranges, not a microsecond-kernel ratio as overall performance.

Indexed retraction and explicit subtraction implement the same sufficient-statistic identity. The explicit baseline can be faster. No neural retraction operator or coupled factor-graph solver is evaluated.

## Correctness tests

Python tests cover duplicate IDs, atomic failure, idempotent removal, batch removal, replay equivalence and cache retention. The JavaScript core checks corrected state, changed route, collision freedom and retained current position. Both normal and standalone interfaces have been exercised through their controls.

## Scope

This is a controlled quasi-static occupancy and planning study. It does not establish real-robot transfer, general scene understanding, automatic fault discovery, calibrated real-camera uncertainty, or learned correction of an arbitrary recurrent state. The selected task distribution is not a deployment prevalence estimate. Source-group revocation is not a statistical model of correlated observations. Numeric memory counts in the kernel experiment exclude Python object and hash-table overhead.
