# Gate 3 linked pilot 2 (attempt 3) — instrument hold, 2026-10-13

**HOLD: instrument-failed. Software valid, execution valid, no scientific
headroom result. No further attempt is authorised; John must disposition.**

Launched from certificate commit `7d358bd` with implementation source
`ea33585` under protocol identity `gate3-v4.1`, after a clean pre-launch
host check. Wall 11:06, GNU time exit 2 (the launcher's stop code), peak RSS
1,565,424 KiB. This attempt charged **1461.08 process-seconds**; cumulative
**2934.34 s** of the unchanged **10,800 s** cap; **7865.66 s** remain.
Receipts: [published evidence](../results/gate3-pilot-2026-10-13-linked-2/manifest.json),
[execution summary](../results/gate3-pilot-2026-10-13-linked-2/execution-summary.json).
Both cards were idle with no compute process afterwards; all three prior
custody receipts and the new one are intact.

## What passed

- All twelve development hosts trained (about 40 s each, 251 MiB peak).
  Same-card, cross-card and three-per-card co-tenancy replays were bitwise;
  the B3 fit replay was bitwise; timing admission passed with three oracle
  seeds and no seed reduction. Thirty-six child receipts, every return
  code zero.
- **The v4.1 repair held on GPU.** The global-permutation no-op twin, which
  stopped attempt 2, passed bitwise on all six development anchors under the
  padded fixed-128 evaluator. The exact-inverse endpoint check (1e-4 nats)
  passed for every P1, P2 and P3 rung on every anchor.
- P1 and P2 are material on every anchor at the first rung of their
  ladders: under the v4 §5 rule the α-pair is {0.1, 0.3} and s_min is 0.1
  (derived by the publisher from the receipts; the launcher stopped before
  recording its choices).

## What stopped it

v4 §5: for each post-hoc family the magnitude is the first ladder rung at
which the exact-inverse gain A1 is at least 0.05 nats on **every**
development anchor, and §9 forbids the pilot from adding rungs. P3 (bias
shift of norm κ·RMS(z)) exhausted its ladder {0.25, 0.5, 1.0}:

| Anchor | κ = 0.25 | κ = 0.5 | κ = 1.0 |
|---|---|---|---|
| 3000 | 0.030 | 0.080 | 0.251 |
| 3001 | 0.011 | 0.038 | 0.141 |
| 3002 | −0.008 | −0.004 | **0.040** |
| 3003 | −0.000 | 0.021 | 0.130 |
| 3004 | −0.007 | 0.019 | 0.173 |
| 3005 | 0.029 | 0.071 | 0.198 |

Five anchors clear the bar at κ = 1.0; anchor 3002 does not. The small and
occasionally negative gains at low κ say that a bias shift of a
batch-normalised, rectified feature is largely absorbed by the downstream;
the gain grows faster than linearly in κ on every anchor. This is a
calibration finding about one instrument family, not about the headroom
families: P5, P6 and P8 were never opened.

## Disposition requested from John

The protocol's own rule (§9 item 4: an instrument finding requires a new
draft before freeze) applies. Options, with my recommendation first:

1. **Recommended: amendment v4.2 extending the P3 ladder** by appending
   κ ∈ {2.0, 4.0}, nothing else changed, then one linked attempt 4 with
   fresh hosts. The rule "first rung that clears 0.05 on every anchor" is
   unchanged, so the three failed rungs stay failed on the record. From the
   measured curvature, κ = 2.0 is expected to clear 0.05 on anchor 3002;
   that is a prediction, not evidence. Cost: attempt 3 reached this point
   in 1461 s, so a fresh attempt 4 is projected at roughly 3,000 to 4,000 s
   against 7,866 s remaining. Inheriting the attempt-3 hosts would save
   about 480 s and needs a reviewed provenance bridge across the identity
   change; not recommended.
2. Amendment dropping P3 and P3′ from the materiality gate (descriptive
   only at κ = 1.0). Keeps budget but loses the bias-only known answer and
   the representability control for the former arm. Not recommended.
3. Stop and archive with this finding.

Confirmation remains blocked regardless. Nothing in this hold changes X, ρ,
Hm, U, the η sets, counts, arms, verdict logic or the seed rules.
