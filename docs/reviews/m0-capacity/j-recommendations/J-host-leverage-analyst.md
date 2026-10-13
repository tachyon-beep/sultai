# J-host: primary host for M0
Agent: yzmir-systems-thinking:leverage-analyst   Date: 2026-10-13

## Options considered
- **H1 (recorded default):** the reference host carries every core arm. The 0.5 Q1 pair (M-sq, M-aff) is sixth and last in the extended order.
- **H2:** the 0.5 host as primary.
- **H3:** the reference host stays primary, and the 0.5 Q1 pair gains its own finishing control (N0/R-temp/R-cal, M-sq, M-aff, T-cool, M-sq | T-cool) and moves to **first** in the extended order.
- Dropped: s = 0.75 as primary. It has no receipts or hashes, more headroom than 1.0 but less than 0.5, and none of 1.0's validation advantage. Also dropped: "both hosts in full core". R-aff and R-all re-run a front that is identical at every width, so they cost the same on 0.5, and the full set does not fit ceiling A.

## Recommendation
Keep the reference host as primary for V1 and V2. Upgrade the 0.5-host extended pair to the H3 set (adding T-cool and M-sq | T-cool) and admit it first by the overrun rule. It reports descriptive D1 and D3 bounds with no verdict role.

## Strongest case for
1. **Validation and reversibility.** Only the 1.0 host has published hashes for V0 to replay, an attempt-4 prior on gain and spread, and a recipe tuned on it. A 0.5 host has never been trained here, and `Host` admits no tail width. A 0.5 primary would put V1 on unreviewed new code and an unmeasured deficit. The TCD itself required a measured starved deficit (≥ 0.05 nats) before freeze.
2. **The default's 0.5 reading never runs under J6 = A.** The core is 8,800 s. The extended items ahead of the 0.5 pair add 900 + 360 + 360 + 210 = 10,630 s, and the 20-epoch point takes the total to 11,200 s, which is over 10,800. Under A, the fallback screen would start with no module evidence on 0.5.
3. **The controls are informative only when the loop is read in two places.** Within one host family, starting loss drives apparent gain (the undertraining loop). The six attempt-4 hosts (spread 0.164 nats) were compressed onto a 0.035-nat floor. Across widths, a higher starting loss on 0.5 may be capacity the host lacks. One host cannot separate "nothing missing at S1 on this host" from "a module at S1 never adds capacity". The frontier says width helps, not that a module captures it. D3 on both hosts can: harvest predicts D3 < X on both; capacity predicts D3 ≥ X after T-cool, at least on 0.5. A mixed reading tells you whether the fallback screen is needed and which population M1 should train on.
4. **Cost per bit is low.** The 0.5 hosts are already in the core frontier. The added cost is 6 × (T-cool + 3 grids + calibrations): 1,320 s at the 1.0 bookings, or roughly 600–750 s by FLOP scaling. The tail is 7.08 M vs 25.95 M MACs per example, and grid arms run the tail only. Either figure fits the 2,000 s of headroom under A, and it runs ahead of R-all, which is one seed, never certified and descriptive. Free extra: report the within-host slope of G on L(N0), before and after T-cool, on both hosts.

## Strongest case against
The plan itself expects the 1.0 host to turn out "not where capacity is missing". A recipe-tuned, overparameterised host is the most likely place to read "harvests undertraining" or indeterminate, so H1/H3 may spend about 3 GPU-hours learning to move. H2 goes straight to the host with the steeper frontier and the starved-host story. Per arm it is about 3.7× cheaper in tail FLOPs, though host training only about 2×, and per-layer overhead caps even that. As a descriptive extended item, the 0.5 reading carries no verdict, so a "present" there still needs a separate preregistration. H3 also displaces R-all under A.

## Dependencies on other decisions
- **J6:** under A, H1's 0.5 pair is effectively dropped, so H3's reordering is what keeps a 0.5 reading. Under B, everything runs, but T-cool on 0.5 is still missing unless it is added.
- **J4:** with a 0.5 co-read, F(0.75) gives the local slope between 0.5 and 1.0 and becomes worth keeping. Under H2, F(0.5) becomes the anchor and the frontier needs a lower point (about 0.25).
- **J-D02 / M2:** anchors 6000–6005 become development anchors, as already noted in §5. M2 and any fallback confirmatory set draw fresh seeds.
- **M1** trains on hosts from the population that read "present", so H3 is the only option that can name that population at M0.
- **J7:** the confirmatory set stays reference-only.

## Evidence
- Reference host 284,634 parameters, seeds 3000–3005, published hashes: plan:92-95. Frontier: plan:100-111. 0.5 pair is last in the extended order and burns the anchors: plan:210-217. Fallback: plan:52-58, 302-305.
- Ledger 8,800 s; extended 900/360/360/210/570/720; ceiling A 10,800: plan:309-346. `PILOT_SECONDS`: src/sultai_screen/contracts.py:47.
- Loop evidence (six 1.0 hosts, impaired only): docs/reviews/m0-capacity/scientific-review-v1-2026-10-13.md:14-25; docs/certificates/2026-10-13-gate3-linked-pilot-3-hold.md:41-53.
- Widths admitted are {0,2,4,8}, the front is shared across widths and only the tail scales: src/sultai_screen/host.py:59-70. Recipe (lr 0.05, momentum 0.9, batch 32, 10 epochs): host.py:129-159. The FLOP figures are my own arithmetic from these shapes.
- Kernels: fit_1x 0.94 s, fit_4x 3.21 s, dev evaluation 0.038 s (01-timing-replay-decision.json `.timing`). Intact training wall 36.9–37.6 s against a 44.5 s charge: intact-300*/result.json. Bottleneck hosts take 42.2–43.5 s for negligible extra FLOPs (bottle*-300*/result.json). That suggests per-layer overhead [inference], so the 25 s §8 booking for 0.5 is derived, not measured, and probably low.
- TCD starved-deficit gate: docs/research/claude-handoff/sultai-concept-demonstrator-design.md:179. D02 near-empty: docs/DECISIONS.md:10. M1/M2 coupling: docs/STEERING.md:100-117.

## Versus the recorded §12 default
This agrees with the default on the primary host. It differs on what the 0.5 component contains (adding its own finishing control) and where it sits in the order (first, not sixth). As recorded, the default gives no 0.5 module reading under J6 = A, and even under B its 0.5 pair cannot separate finishing from capacity.

## What would change this recommendation
- If the measured R-aff kernel pushes the core near 10,800 s, H3 does not run under A, and the choice becomes B or H2.
- If John fixes J-D02 to the 0.5 host and wants M0 to read on M2's start population, choose H2, after a measured starved deficit and review of the tail-width code.
- If a cheap pre-pilot T-cool measurement on 1.0 already shows G(T-cool) ≥ the attempt-4 module gains, choose H2.

## Confidence
Medium. The cost and validation arithmetic is checked against source. The cross-host discriminating predictions are hypotheses, and no 0.5 host has ever been measured.
