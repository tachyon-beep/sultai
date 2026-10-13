# J-retire: Retire the Gate 3 repair families and arms from M0; keep only the P1 instrument
Agent: product-shaping-architect   Date: 2026-10-13

## Options considered
1. **Retire all as §2 proposes** (P2, P3, P3′, P5, P6, A7/A8, B4, B1/B2/B2′/B6); P1 (A1, A2, A3, B3 ≥ ρ·A3) stays as V0 validation.
2. **Keep one bottleneck family (P5 at w = 2 or 4) as a descriptive bridge arm.**
3. Dropped: retire P1 too. V0 then has no fitter known-answer, rule 5 forbids it, and it costs only 340 s (draft §8:323).

Not lost: P8 (intact host) is M-sq; B5 (wide adapter) is M-wide; the Gate 3 oracle kernel survives inside the P1 instrument.

## Recommendation
Accept J-retire as §2 states it, with one precision: "retired" means **no verdict, report or confirmatory role** (J7), and the P1 instrument runs through the existing `posthoc_worker` unchanged, so P2/P3/P3′/A7/A8 still land in receipts as unread bytes. Continuity with attempt 4 is carried by two core arms, not by a bridge: the V0 P1 B3 replay against published hashes, and M-sq-fresh, which is Gate 3's never-opened P8 under Gate 3's held-out data regime.

## Strongest case for
- **Clarity of §1.** Every retired item is a repair host or repair arm. Keeping one puts "how much lost tensor does the module recover" back into a study asking "does the module beat every cheaper route on an unimpaired host". Rule 5 (STEERING.md:94) makes retirement the default.
- **The bottleneck hosts are barely impaired.** P5/P6 no-op losses 0.632–0.718 (oracle4-*/result.json) sit inside the intact spread 0.601–0.765 (posthoc-*/result.json, A1), and P6 − P5 ≤ 0.026 nats everywhere (hold:43–50). A P5 bridge would re-measure the calibration/finishing effect the scientific review named (scientific-review-v1:25) on a host that differs from the reference by a pinch worth under 0.03 nats.
- **A bridge has one leg.** M0 fits on seen data against an R-bn no-op with a 1-seed/3-rate/3,200-step grid (draft §4:137); attempt 4 fitted held-out data against the raw host with 3 seeds × 4 rates × {200, 800}. Under M0's regime P5 no longer matches attempt 4; under Gate 3's it matches nothing in M0.
- **Budget.** A P5/P6 bridge at Gate 3 kernels costs ≈ 6 × (43 s host + 100 s grid) ≈ 860 s (execution-summary host_costs; oracle4 wall 100–104 s), 10 percent of the 8,800 s core (§8:330). Under ceiling A it displaces R-all (≈ 900 s, §8:332), the one extended arm that is a rule-2 cheaper route. Wrong trade.
- **A7/A8, B4, B6 and the cheap bank belong to M1**: its row names retrieval and few-step SGD as comparators (STEERING.md:105). At M0 they answer nothing.

## Strongest case against
- The only unimpaired-host reading in the receipts is an inference (B3 landing 0.048–0.207 nats below A1 on P1, alignment-review-v1:28). If M-sq reads 0.00–0.03 nats where attempt 4 read 0.04–0.26 on bottleneck hosts, a reader cannot tell "data regime changed" from "kernel changed" without a same-instrument arm. Answer: M-sq-fresh (draft §5:198) and D4 isolate the data regime; the V0 P1 replay pins the kernel. A better bridge than P5, at no extra cost.
- Retiring P2/P3 loses full-rank and bias-only known answers. Answer: M0's new code has its own (slicing, complement hash, zero-module identity, §7); P1 is the fitter check rule 5 asks for.

## Dependencies on other decisions
- **J6**: option B (4 GPU-h) removes the displacement argument but not the clarity argument; the recommendation holds under both.
- **J-host**: if the 0.5 host becomes primary, the bridge is even less relevant (F(0.5) is the cleaner starved host, Gate 3 §5:252 P10).
- **J7**: no retired arm enters the confirmatory set.
- **J9**: the retired families stay Gate 3 evidence under the closed gate (final-disposition table).

## Evidence
- docs/STEERING.md:94 (rule 5), :105–106 (M1/M2 comparators), :109.
- docs/plans/2026-10-13-capacity-frontier-screen.md:86–87 (retirement list), :137 (seen-data regime), :198 (M-sq-fresh), :208 (P1 instrument), :323 (340 s), :330 (8,800 s), :332–337 (extended set, R-all 900 s), :342 (ceiling).
- docs/plans/2026-10-10-gate3-headroom-screen.md:245–252 (families), :317–318 (A7/A8), :335–340 (B1–B6).
- src/sultai_screen/screen.py:252–271 (posthoc_worker runs P1/P2/P3/P3′ and A7/A8 in one call), :306–308 (bottleneck_worker runs P5 and P6 together), :327; src/sultai_screen/host.py:59–62 (widths 0/2/4/8 exist).
- docs/certificates/2026-10-13-gate3-linked-pilot-3-hold.md:43–50, :57.
- docs/results/gate3-pilot-2026-10-13-linked-3/: execution-summary.json host_costs (bottle 42–44 s); oracle4-*/result.json (wall 100–104 s, noop_loss 0.632–0.718); posthoc-*/result.json (A1 0.601–0.765, B3 0.547–0.582).
- docs/reviews/m0-capacity/scientific-review-v1-2026-10-13.md:25; alignment-review-v1-2026-10-13.md:28, :32.

## Versus the recorded §12 default
Match. One operational addition: "retired" means no verdict, report or confirmatory role, not code removal, so the P1 stage stays bitwise the attempt-4 posthoc worker and the V0 replay keeps its meaning.

## What would change this recommendation
- **Bottleneck family (P5/P6):** M0 reads re-tuning-equivalent, calibration or harvests-undertraining on the reference host **and** the 0.5-host Q1 pair is also null. The next screen then needs a host with a known class-relevant deficit as positive control; a redesigned bottleneck (hold option 3) returns at that starved-host screen, known answer stated net of re-tuning. M2 may reuse a known pinch site for site discovery.
- **A7/A8, B4, B6/cheap bank:** V1 present opens M1; they return as its retrieval and few-step baselines.
- **P2/P3/P3′:** V0 fails B3 ≥ ρ·A3 on P1; the full post-hoc bank becomes the diagnostic under separate authorisation, before any V1 reading.
- John re-states the objective to include repair (rule 5's own escape clause).

## Confidence
High: clarity and budget arguments stand independently, and the bridge the alternative wants already exists more cheaply (M-sq-fresh plus the V0 replay).
