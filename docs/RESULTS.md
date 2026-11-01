# What the measured results establish

The evaluation uses a **procedural spatial-navigation suite**, not a public VQA benchmark or real-image dataset. Four frozen samplers produce 120 episodes each, for 480 total. Seeds, all per-episode records, methods and code are included.

## Evidence to action

1. **Invalid free-space evidence causes unsafe plans.** In the false-free-space family, retaining the report yields 0/120 collision-free goal routes; every exact correction method yields 120/120. The same result holds for the source packet-group family.
2. **Correct state is not enough if an old route remains cached.** The stale-cache ablation repairs the state but retains the pre-audit route, preserving those failures. This is an intentionally incomplete system, used to illustrate cache coherence rather than a competitive baseline.
3. **Success hides unnecessary motion.** In the false-obstacle family, every method succeeds, but median remaining route length decreases from 39 to 23 steps after correction: 16 fewer steps, or a 41.0% difference between medians. This is not a claim of a 41% mean per-episode improvement.
4. **Unaffected state is retained.** Across the expanded study, maximum log-odds error against replay is 4.44e-16 and unaffected-factor drift is zero. An empty audit makes no state change.
5. **An exact sufficient statistic is an important control.** Explicit subtraction matches indexed repair's corrected decisions. On this factorized model, indexed retraction is not a new inference algorithm and does not beat explicit inference.

## Backend cost

At 32,768 encoded observations, median backend revision + 20 route forecasts + global A* is 12.62 ms for full replay, 10.95 ms for checkpoint replay, 0.45 ms for explicit subtraction, and 0.56 ms for indexed retraction. Medians are over 24 independent histories with randomized method order. These are implementation-specific local CPU measurements. Initial encoding/index construction, display, network and acquisition are excluded.

The indexed ledger gives addressable revocation; its metadata and API overhead are real costs. The explicit baseline can be faster. No claim of learned-operator superiority follows from the replay comparison.

## Contribution boundary

The demonstrated contribution is a complete evidence-to-forecast-to-action execution contract, an inspectable artifact, and reproducible controls. The study does not establish a novel neural memory-retraction method, general 3D understanding, autonomous fault discovery, or real-robot generalization. A learned non-additive revision operator remains outside the reported results. These boundaries are stated in the paper rather than hidden behind the percentages.
