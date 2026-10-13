# J-D02: starting point for rung M2 (near-empty host versus the 0.5 host)
Agent: product-shaping-architect   Date: 2026-10-13

## Options considered
- **A. Near-empty host per D02**: within the M0 host family, tail widths at s ≤ 0.25 (33,978 parameters, 11.9 percent of reference; s = 0.125 is 20,426, 7.2 percent). s = 0.25 is the first standard halving whose count falls outside the 25–40 percent band D02 rejected. Finding, not assumption: the front (stem + producer) is 14,928 parameters whichever tail widths are chosen (recomputed from `host.py:57-70`; the formula reproduces 284,634 and 85,274 exactly), so this family floors at 5.2 percent. Nothing in M0 trains any of these.
- **B. The 0.5 frontier host** (85,274 parameters, 30.0 percent of reference). M0 trains six; it sits inside the 25–40 percent band D02 rejected.
- **D. Open until the M2 draft** (author's default), with M0 recording nothing extra.
- Dropped: **C**, reference-width host with a grow-and-prune schedule. That is pruning a full model, the ordinary pruning literature, and contradicts D02 and D14's "jointly with addition in continued development".

## Recommendation
Decide the principle now, the number at the M2 draft: **M2 starts near-empty in the sense of D02; the 0.5 host is not that start and may become it only by John's explicit, recorded relaxation of D02; within the M0 host family (front 14,928 parameters) "near-empty" means tail widths at or below s = 0.25; whether M2 keeps that family and its S1 site is M2's own question (STEERING §5: site discovery is M2's); the exact width and growth schedule are fixed in the M2 draft after M0's frontier is measured.** M0 records five decision-independent items (Dependencies) so no option is foreclosed, and, if John leans near-empty, swaps F(0.25) for F(0.75) in the extended order.

## Strongest case for
D02 is John's intent, not an author default (DECISIONS.md:10), and the alignment review already caught the ladder silently starting M2/M3 from the 30 percent host (alignment-review-v1:26); leaving J-D02 fully open keeps that drift path open for another cycle. Fixing the principle costs no GPU time and changes only what M0 records. Fixing the number now would be unsupported: no evidence exists below s = 0.5, and the Claude TCD's own rule was a measured deficit before freezing a starved host (sultai-concept-demonstrator-design.md:179).

## Strongest case against
Trainability of s ≤ 0.25 under the same recipe is unmeasured; a near-empty start may need far more compute than the 0.5 start for M2's matched-compute comparison, and M2 must also discover sites and remove host channels, none of which M0 builds. The 0.5 host is the only starved host M0 produces, with published hashes and a measured loss, so rejecting it as the M2 start discards M0's one directly transferable starting point. Esper-lite offers no counter-prior: seeds germinate into a full trained host, rent is overage against a 500,000 budget (README.md:243; ROADMAP.md:38), PRUNE exists only before fossilisation (README.md:210-211) and Emrakul, removal of committed structure, was planned and never shipped (ROADMAP.md:370-385). M2 from near-empty is unprecedented in both theories.

## Dependencies on other decisions
What M0 must record regardless of how J-D02 lands:
1. **Seed registry**: Appendix A (plan:438-449) reserves no block for M2 starting hosts; reserve one now under the never-reuse rule.
2. **Burn rule**: 0.5 hosts 6000–6005 become development anchors, never confirmatory, only if extended item 6 runs (plan:216-217); so J6 (ceiling) and the overrun rule decide whether M2 can reuse them even for development.
3. **Constructor**: the new tail-width parameter (plan:72-73) must accept explicit width tuples, not only s ∈ {0.5, 0.75, 1.0, 1.5}, or M2 re-reviews host code. Requirement on the implementation plan.
4. **Hashes**: publish `final_state_sha256` for every frontier host so M2 replays bitwise without loading a checkpoint (consistent with plan:93-95).
5. **Removal boundary**: slicing with mean fold (plan:126-130) is module-only; host-channel removal across conv + BN is M2 new code.

Reusable from M0 as priors/scale only, never as M2 arms: F(s) losses (10-epoch same recipe; M2 compares at matched compute, STEERING.md:106, so its static comparator is retrained), V2/D2 and survival curves (terminal truncation precedent, STEERING.md:109-116), T-cool (how undertrained the recipe is), S-static (the D12 design-in control M2 must also carry). Burned: anchors 3000–3005; 6000–6005 per item 2; 7500–7599 are reference-width, irrelevant unless C.

Couplings: **J4/J6**: if John leans near-empty, F(0.25), six seeds, replaces F(0.75) in the extended order at equal or lower cost (0.5 host booked 25 s, plan:321; ≈ 20 s each, ≈ 150 s booked, unmeasured) so the frontier reaches below the M2 start. **J-host** is separate: the 0.5 host as M0's fallback on a re-tuning-equivalent V1 (plan:304-305) is an M0 question. **J-D2** sets the grow-then-cut prior, not the host. **J9**: Gate 5 is read by M2, so M2's host must support developmental, not terminal, removal.

## Evidence
DECISIONS.md:9-10, :17, :22 (D01, D02, D09, D14); CONCEPT.md:20-25, :155-168; STEERING.md:43-47, :106, :109-116; plan v2 :72-73, :93-95, :100-105, :126-130, :212-217, :304-305, :319-321, :336, :425, :438-449; src/sultai_screen/host.py:57-70 (counts recomputed: front 14,928; s = 0.25 → 33,978; s = 0.125 → 20,426); alignment-review-v1:26, v1-dispositions:17, :51; sultai-concept-demonstrator-design.md:19, :102, :179; esper-lite README.md:73-77, :210-211, :243-245, ROADMAP.md:32-47, :370-385.

## Versus the recorded §12 default
Differs in degree. The default records J-D02 as open and M0 as unaffected. This recommendation keeps the number open but fixes the principle (0.5 is not near-empty without an explicit D02 relaxation; the family's floor is 14,928) and names five M0 recordings plus one conditional extended-arm swap, because "open" without them lets M0 consume seeds and freeze a constructor that M2 then has to re-review.

## What would change this recommendation
John explicitly relaxes D02 to the 30 percent host (then B, with the 6000–6005 burn rule binding and a fresh 0.5 block for M2). The M0 frontier slope between 0.5 and 1.0 is flat within its seed-bootstrap interval, or s ≤ 0.25 hosts fail to train under the unchanged recipe: either says near-empty must be a different family. V2 reads "hurts" with the refit certified, which lowers the grow-and-remove prior and argues for the cheapest M2 first.

## Confidence
Medium: the D02 band, the 30.0 percent figure and the 14,928 front floor are hard facts; trainability and compute at s ≤ 0.25 are unmeasured.
