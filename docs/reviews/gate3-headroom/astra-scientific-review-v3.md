# Independent Astra review, round 2: Gate 3 headroom screen DRAFT v3

Reviewer: fresh independent Astra agent, 2026-10-11. Read-only; no edits, no runs, no tracker writes. Reviewed `/home/john/sultai/docs/plans/2026-10-10-gate3-headroom-screen.md` at SHA-256 `6338c8141b2a4f650a6714a7b3f49aa1be5ba493a43797e6c5ab42f2a36b52c1` (verified) against the three round-one reports in `docs/reviews/gate3-headroom/`, the correction protocol, DECISIONS (D06, D11, D15, D21, D25), HYBRID_CONTRACT, the Gate 2 final certificate, the Claude design, and `src/sultai/formation.py` / `repair.py` / `formation_controls.py`. Derivations marked [derived] are mine and unmeasured.

Verdict up front: **not ready to freeze.** Seven items block (F1–F7 below). Every one is a one-to-three-sentence fix and none changes the study's shape. The round-one fixes are largely real; three of them introduced new defects (Astra B2, Astra N2, Astra B6 artifact naming) and one decided item (D-M) has a scientific consequence the document does not state.

---

## Part 1. Verification of §15 dispositions against the v3 text

Verdict key: **fixed** / **partially fixed** / **not fixed** / **fix introduced new defect**.

| Round-one finding | §15 claim | Verification verdict | Evidence in v3 |
|---|---|---|---|
| Builder Q1 / Astra B2: cheap comparator unspecified; H a free parameter | Adopted | **Fix introduced new defect** | §6 defines full-batch step (2,000 conditioning, (W,b) only, plain SGD, zero init), minibatch step (256), η-set, B1/B2/B2₁₆ at fixed η_B by a pilot rule, B2′ best-of-4 on selection; §9 forbids changing sizes or arm definitions. Real and complete on the axes round one named. New defect: the single η-set {0.3, 0.1, 0.03, 0.01} is shared between B3 (200–800 minibatch steps, where stability caps η) and B2′ (4 full-batch steps, which lives in a different η regime; the synthetic instrument's few-step SGD used lr 4.0, `formation.py` `Protocol.sgd_learning_rate`). The cheap side of the headline contrast is capped at the oracle's regime. See F4. Also the row text "isolates optimisation budget from selection access" overstates: B3 retains 24 candidates on selection vs B2′'s 4 (6× search budget via seeds × schedule). See N3. |
| Builder Q2 / Astra B1: P5 absolute ≈ 0 contradicts P8 | Adopted; owner D-M | **Partially fixed** | V0 clause 4 is the matched-pair relative criterion (§7, "P5 is judged relative to P6 on the same host, not against zero"). P8 is "null calibrator" (D-M). But Astra B1's fix said: if null calibrator, "V1 on P6 should additionally require B3_P6 − B3_P8 ≥ 0.05 so that headroom is repair-specific." v3 reports (median B3 on P6) − F8 "with no verdict role" (§7 V1). The calibrator calibrates nothing; clause 5 cannot fail. See F5. |
| Builder Q3 / Astra N6 / torch N6: A1 affine; tanh cannot reach it | Adopted | **Fixed** | V0 restructured as clauses 1–3; A3 ≥ A2 predeclared under D21 (§6 after Table A); P3 norm κ·RMS(z). Builder Q3's second question (an affine B3 alongside the tanh B3) is answered only by implication ("The tanh family is primary for every gradient arm", §3); acceptable. Note [derived]: at P1 α = 0.1 the attenuated channels sit in tanh's near-linear range, so the tanh family can approximate the inverse there; this makes clause 3 a fair test of the optimiser, which is the right choice of validation case. |
| Builder Q4 / Astra B3 / torch N6: twin is not a noise floor | Adopted | **Fixed** (with one loose end) | §7 "Noise floor and twin": twin bitwise, mismatch is a stop; floor for differences is the per-example paired bootstrap SE on dev; A1 tolerance 1e-4 (clause 2). Loose end: the B3 plateau rule (§6 Table B) cites "the paired bootstrap SE (§7)", which §7 defines over the 5,000 dev examples, so plateau certification is computed on the endpoint; and the training-loss flatness clause is not computable as written. See N1. |
| Builder Q5 / Astra B6 / torch B2: former span; V2 foregone | Adopted; owner D-F | **Fix introduced new defect** (minor) | V2 descriptive (§7); A8 analytic four-template fit added (Table A, matches `formation_controls.analytic_template_fit`); P3′ in-span case (§5); evidence mode `paired`, seeds 1000–1031 named (Table A). New defect: "the materialised weight SHA-256 is recorded at freeze and asserted at run time" is not computable at freeze. `FrozenGenerator.form` materialises a 272-parameter adapter per conditioning set (`formation.py` lines 298–302), so the materialised W depends on each real anchor's evidence. The fixed artifact is the 11×4 meta-regression `FrozenGenerator.weights`. See N4. |
| Builder Q6: P3 norm unspecified | Adopted | **Fixed** | §5 "κ·RMS(z)", pilot rule over {0.25, 0.5, 1.0}, smallest κ with A1 ≥ 0.05 on every dev anchor. RMS(z) population (per anchor, over conditioning) is implied, not stated; decidable. |
| Advisor on v2: P6 "by construction"; clause 3 mixes two questions | Adopted | **Partially fixed** | §5 text: "P6 is repairable by information, not by guarantee" with the width rule {4, 2, 8} and a declared stop. But §1 ("a deficit that is repairable by construction") and the §5 table P6 row ("Repairable by construction, no planted answer") still say "by construction". Clause 3 separation is real (§7 clause 3 text). See N7, N8. |
| Astra B4: V1 point estimate; X from noise | Adopted; owner D-D | **Fixed** | §7 V1: one-sided 95% anchor-bootstrap bounds of the median H′, three outcomes; X justified as a repair margin under D15; Y removed. Coherence verified in Part 2 item 1. One omission: nothing orders the owner's fixing of X before the pilot's gain summaries are opened. See F1. |
| Astra B5: B3 never validated on a known answer | Adopted | **Fixed** (with a pilot gap) | Table A "B3 oracle (P1 α = 0.1 only)"; V0 clause 3; ledger line "64 P1"; A6 retired. Gap: §9 items 4 and 7 never exercise clause 3 on development anchors, so a grid too weak for the known answer is discovered only in confirmation, where it ends the attempt. See N2. |
| Astra B7: V0 failure permits fix-and-reread | Adopted | **Fixed** | §7 "Fail → the attempt ends. Confirmatory anchors are never reused after a V0 failure. Any change ... requires a new preregistration version with a new SHA-256, an explicit disposition and a fresh confirmatory seed range." |
| Astra N1: grid changes outcome-contingent | Adopted | **Fixed** for its own scope | §9 ordering rule: timing (items 1–3) read and grid reductions decided before gain summaries (items 4–6); seeds cut before schedules. Residual: §9 then says the pilot "may not change ... any arm's definition", which a seed cut is. See N5. No ordering inside items 4–6. See N6. |
| Astra N2: pilot cannot size from n = 2 | Adopted | **Partially fixed** | 6 development anchors per family (§9). But no rule converts measured spread into a count: §4 "it may raise counts before freeze from measured per-anchor spread"; §9 item 6 "fills counts by the rule 'raise only'". "Raise only" is a direction, not a rule, and the same pilot item exposes the dev H′ level. See F7. |
| Astra N3: branches buy nothing | Adopted | **Fixed** | §4: one host per anchor, 64/32, no spawn point, no basin check. |
| Astra N4: no cost axis on the cheap arm | Adopted | **Fixed** | k ∈ {1, 4, 16}; B6 steps-to-match over {1, 4, 16, 64}; B3 wall cost reported (§7 V1). B2′(k) for k ≠ 4 is used by B6 but only B2′(4) is defined; trivial. |
| Astra N5: no rule combining families | Adopted | **Fixed** | §7 V1: "There is no rule combining families; P6 is the only primary." |
| Astra N7: intervals condition on one data draw; recipe tuned on endpoint | Adopted | **Fixed** | §7 per-example paired bootstrap as second interval; §3 disclosure. |
| Astra N8: Gate 3 clause imprecise; "corpus justified" | Adopted | **Fixed** | §1 rewritten: "not ruled out; it is not thereby justified"; B4 retrievable fraction (§6). |
| Astra N9: retrieval from same-anchor branches; alignment batch | Adopted | **Fixed** (one gap) | "nearest other anchor", metric named, conditioning batch (Table B B4). Gap: on P6 the module reads z and writes to z̃, and z̃'s 16 channels are a second per-anchor basis; alignment is stated for z only. See N9. |
| Astra N10: admission dropped silently | Adopted | **Fixed** | §2 "Admission (D25) is out of scope ... all harms are reported raw." |
| Astra N11: builder gaps | Adopted item by item | **Fixed** | Each of the ten items is present in §3–§6 as claimed (draw granularity, every-BN rule, step definitions, B5 tanh 16→64→16, A7 artifact, anchor units, uncertified flagged, derived card assignment, no Simic import, fixed sizes). B5's module form (where the tanh sits) is still loose; diagnostic only. |
| Torch B1: gate scripts do not cover a sibling package | Decided D-L | **Fixed as a decision** | §11 names both scripts' current scan (`check_policy.py` rglob of `src/sultai`, `check_quality.py` non-recursive glob and fixed mypy target; confirmed against the scripts), the widening, extended negative controls, ruff first-party, and the re-run assertion on the stdlib instrument. Consequence not stated: §11 heading "(no code until the freeze)" contradicts the pilot running before the freeze. See N10. |
| Torch B3: determinism flag set incomplete | Adopted | **Fixed** | §3 rule 5 lists the full set, recorded in every result file. |
| Torch B4: split derivation under-specified | Adopted | **Fixed** (tag misuse) | §3 writes the encoding out; seed, first eight indices and permutation SHA recorded; dev indices to be committed as source of truth; checked against Simic's bounded-pilot manifest. The check is tagged **[measured]**, which §0 defines as "from a saved artifact"; no artifact of that 2026-10-11 check exists in the repo (`git status` shows only the draft and reviews). See N11. The three-label sub-partition derivation is still not written out. See N12. |
| Torch N1: ledger optimistic; B5 missing; pilot over ceiling | Adopted | **Fixed** | §8 rebuilt on the minibatch definition; B5 reduced; ceilings re-proposed; concurrency stated. Arithmetic checks [derived]: 12 × 200 + 12 × 800 = 12,000 steps per host-case, ≈ 60 s at 5 ms; 192 host-cases ≈ 3.2 GPU-h. Consistent. |
| Torch N2: BN every layer; cached z; P6 residual base | Adopted | **Fixed** | §3 rule 1; S2b base "u = z̃ + (W tanh(z) + b)". |
| Torch N3: determinism offenders; pilot tests | Adopted | **Fixed** | §3 GAP as `mean(dim=(2,3))`; derived `manual_seed`; §9 item 3 lists the five tests. |
| Torch N4: audit hook; five-file loader | Adopted | **Fixed** | §3 rule 2 and Data paragraph. |
| Torch N5: typing idioms; one exemption | Adopted | **Fixed** | §11 idioms; `as_tensor` runtime-checked helper; cuDNN version not recorded. |
| Torch N7: record recipe; wheel string; cached-z interface | Adopted | **Fixed** | §3 recipe; §11 `torch==2.9.1+cu128` with index; cached-z as primary abstraction. |
| Torch N1 branch overcount | Moot | **Moot, correctly** | Branches removed. |

Every round-one finding has a row in §15; no finding is silently dropped. One cross-reference nit: the §15 row for the twin cites "torch N6" correctly, and the former row cites "torch B2" correctly.

---

## Part 2. Fresh findings on v3, ranked by severity

### Blocking

**F1. Owner numbers can legally be fixed after the pilot's dev H′ is seen.**
Claim: nothing in v3 orders the fixing of X, ρ, Hm before pilot items 4–6 are opened.
Evidence: §0 lists "John has replaced every OWNER-DECISION placeholder" and "the pilot has filled the values it is permitted to fill" as co-equal freeze conditions with no ordering; §9's heading requires only "John's go and a GPU-hour figure"; §9 item 6 exposes "per-anchor spread of H′ on development P6 anchors", which necessarily exposes the dev median too; §7's "OWNER-DECISION values are proposals and are not chosen from any outcome" is an assertion, not a process rule. Round one's sound list credited v1 with "margins fixed and the document committed before the pilot"; that property is gone from v3's process text.
Why it matters: D15 forbids choosing the margin from observed outcomes; an X chosen with the dev median in view is exactly that, however honestly intended.
Fix: add to §9: "X, ρ and Hm are fixed and committed (a versioned draft with SHA-256) before the pilot's go. The pilot may not be opened until that commit exists." One sentence.

**F2. The "nonfinite loss" stop rule, inherited by the confirmatory run, is triggered by a single divergent grid candidate.**
Claim: as written, one B3 or B2′ candidate diverging at η = 0.3 ends a 16 GPU-hour attempt.
Evidence: §9 "Pilot stops: timeout, nonfinite loss, ..."; §10 "Stops as in §9 plus the budget ceiling." B3 runs plain SGD at η up to 0.3 for 800 minibatch steps through the frozen downstream, 24 candidates × 192 host-cases = 4,608 fits; B2′ runs 4 full-batch steps at η = 0.3 on 128 host-cases. §11 separately provides "a dedicated T2 failure class for oracle non-convergence", which implies non-convergence is recoverable, but the stop list does not say so. [derived] Divergence of the largest-η candidate somewhere in 4,608 fits is likely, not rare.
Why it matters: the attempt ends, confirmatory anchors are burned (§7 "never reused"), and a new version with fresh anchors is required, for a reason that has nothing to do with the instrument.
Fix: split the rule: "A nonfinite loss in host training, in any no-op evaluation, or in the evaluation of a selected candidate is a stop. A nonfinite loss inside a grid candidate's fit marks that candidate non-finite; it cannot be selected and is counted in the ledger. An anchor whose every candidate in an arm is non-finite is retained, flagged, and reported." Also state that a non-finite or divergent candidate's selection loss is treated as +∞.

**F3. "No headroom" has a programme consequence but no plateau guard.**
Claim: V1 can return "absent" (programme stop, §1) with the oracle unconverged, and no predeclared rule prevents it.
Evidence: §1 "If not, a learned former has nothing to amortise and the programme stops cheaply." Table B: "uncertified anchors are retained and flagged, never dropped." §7 V1 reads the median over all 32 anchors with no reference to certification. Clause 3 validates the grid only on P1 α = 0.1, whose needed repair [derived, W_kk ≈ 9 on four channels] is a different optimisation problem from the P6 bypass.
Why it matters: this is Astra B5's false-negative moved, not removed. A mostly-uncertified P6 set with upper bound < X would be read as "SGD already does this" when it means "the oracle did not finish."
Fix: predeclare one rule: "If more than 25% of P6 anchors are plateau-uncertified, a 'no headroom' outcome is reported as 'indeterminate: oracle not at plateau' and is not a programme stop. 'Headroom present' is unaffected, since an unconverged oracle biases against it." The 25% is an owner number; propose it as such.

**F4. The cheap comparator's learning-rate range is inherited from the oracle's regime.**
Claim: B2′ (4 full-batch steps) selects over the same η-set {0.3, 0.1, 0.03, 0.01} that B3 uses for 200–800 minibatch steps; the cap at 0.3 is set by the oracle's stability needs, not the cheap fit's.
Evidence: §6 "η-set = {0.3, 0.1, 0.03, 0.01}"; Table B B2′ "4 full-batch steps at each η in η-set"; §9 forbids the pilot from changing any arm's definition, so this cannot be repaired after the pilot. For scale only: the synthetic instrument's one/two/four-step SGD control runs at `sgd_learning_rate = 4.0` (`formation.py` line 50), on MSE with its own per-example normalisation (`scale = 2·lr/(n·16)`, line 399), so the number is not comparable in magnitude to a cross-entropy full-batch η; it is cited only to show that few-step fits in this project have been run at rates an order of magnitude above 0.3 and that the cheap regime was treated as its own regime. The finding rests on the derivation that follows. [derived] A full-batch gradient on a 272-parameter module through a deep downstream is O(0.1–1) per entry; four steps at η ≤ 0.3 move W by at most about one unit, while B3 at 4× can move it by tens. If the P6 repair needs W entries of several units, B2′ is handicapped by range, not by budget.
Why it matters: this is the "manufactured headroom" mechanism round one's B2 named, reappearing one level up. B2′ selects on selection, so adding larger rates costs four extra 4-step fits per host-case and cannot hurt it; a divergent rate simply loses selection (given F2's fix).
Fix: define a separate cheap-arm rate set η-set_cheap = {3, 1, 0.3, 0.1, 0.03} (or η-set ∪ {1, 3}) for B1/B2/B2′/B6 and the η_B rule; keep η-set for B3. State that B2′ then has 5–6 candidates versus B3's 24 and that this residual is part of B3's declared generosity. Owner decision on the set, but it must be in the frozen text.

**F5. V1 "headroom present" can fire from generic stitching gain, and D-M's choice leaves the null calibrator with no function.**
Claim: H′(P6) ≥ X is satisfied whenever a 24-fit minibatch grid beats a 4-step full-batch fit on any host, repair or no repair; nothing in V1 separates that from the bypass repair.
Evidence: §7 V1 reads H′ on P6 only; clause 5 "median B3 gain on P8, F8, is reported with its interval" is a clause that cannot fail inside an "all clauses must pass" gate; (median B3 on P6) − F8 "with no verdict role"; §1 claims a positive V1 "establishes real-host repair headroom". The ledger shows cheap arms run on P5 ("128 host-cases" = 32 P5 + 32 P6 + 64 P8), so H′(P5), the same-host paired null for H′(P6) (module after the bottleneck, no bypass information, same host, same grid, same cheap fit), is already computed and read by nothing. P8 is on a different host population and cannot be paired; P5 can.
Why it matters: this is the Gate 2 lesson "endpoint recovery does not establish specificity" in new clothes. If H′(P6) ≈ H′(P5) ≈ 0.03, V1 says "present", the programme proceeds toward a corpus of repairs, and the headroom is in generic fine-tuning that a repair former would not target.
Fix: predeclare a repair-specificity reading that moves no threshold: "ΔH′ = H′(P6) − H′(P5) per bottleneck host (paired). Alongside the V1 outcome, report the one-sided 95% lower bound of the median ΔH′; label the outcome 'repair-specific' if that bound > 0, 'generic fine-tuning headroom' otherwise. A programme go requires 'present' and 'repair-specific'." Then either give clause 5 a function (F8 as the stitching reference in the report) or move it out of V0 into the descriptive report so that V0 contains only failable clauses. D-M's decision stands; its consequence must be written.

**F6. V0 clause 1 is "every" where every other clause is ≥ 90%, and a single anchor fails the attempt.**
Claim: one of 64 intact anchors with A1 gain < 0.05 on P3 (or P1/P2) ends the attempt for a reason unrelated to instrument validity.
Evidence: §7 clause 1 "A1 gain ≥ 0.05 nats on every P1, P2, P3 anchor"; clauses 2–4 use "≥ 90%"; §7 "Fail → the attempt ends." The κ rule guarantees A1 ≥ 0.05 on six development anchors (§5), which does not bound the minimum over 64; A1 is exact, so its gain is purely the host's sensitivity to the planted deficit, which varies by anchor. The Claude design's G0 used "≥ 90% of repairable-labelled test cases".
Why it matters: an immaterial impairment on one anchor makes that anchor uninformative for clauses 2–3; it says nothing about the instrument, yet the cost is a full re-run with fresh anchors.
Fix: "≥ 90% of anchors per family; anchors below 0.05 are listed and excluded from the denominators of clauses 2 and 3 for that family." Exclusion on an A1 criterion is not outcome-contingent on any arm.

**F7. Count-raising has no declared rule and is decided by someone who has seen the dev H′ level.**
Claim: "raise only" is a direction, not a sizing rule; the pilot item that supplies the spread also supplies the median.
Evidence: §4 "it may raise counts before freeze from measured per-anchor spread"; §9 item 6 "fills counts by the rule 'raise only'". No formula. The same six dev anchors give the H′ median.
Why it matters: raising n does not move the point estimate but changes the probability of "indeterminate" versus a verdict, and the person deciding knows which verdict is near. That is outcome-contingent in the direction that matters, under D15.
Fix: declare the rule as a function of spread only: "n_P6 = max(32, the smallest n at which the one-sided 95% bootstrap half-width of the median, computed from the pilot's per-anchor SD of H′ under a normal approximation, is ≤ X/2)", or an equivalent, and require the pilot to report the SD before the median is opened. Alternatively drop sizing and fix 32 with the declared prior.

### Non-blocking, ranked

**N1. Plateau rule: wrong set and a non-computable clause.** Table B: "the 4× best improves on the 1× best by less than 2 × the paired bootstrap SE (§7)"; §7 defines that SE over the 5,000 dev examples, so certification is computed on the endpoint. It chooses nothing and uncertified anchors are retained, so it does not leak into selection, but it is an unnecessary touch of dev; compute it on selection. The second clause, "the 4× training loss over its final 10% of steps is flat within the same SE", compares minibatch-256 training losses (per-step SE ≈ 0.03 [derived]) against a 5,000-example SE (≈ 0.007 [derived]); it will fail by noise on most anchors, making the flag decorative. Fix: "mean training loss over steps 721–760 versus 761–800 differ by less than 2 × the SE of those step losses", or drop the clause. Also define "4× best" and "1× best" as best-on-selection within each schedule.

**N2. Clause 3 is never exercised in the pilot.** §9 items 4 and 7 cover A1 materiality and the P6 − P5 known answer, not B3 ≥ ρ·A3 on P1 α = 0.1. If the grid cannot find the attenuation inverse on development anchors, that is learned only in confirmation, where it ends the attempt. Fix: add to §9 item 4 "B3 versus A3 on P1 α = 0.1 on every development intact anchor; a fraction below ρ on any development anchor is an instrument finding that requires a new draft before freeze (the pilot may not change B3's definition)."

**N3. "Isolates optimisation budget from selection access" overstates.** B3 has 24 candidates on selection, B2′ has 4. The 6× residual (seeds × schedule) is search budget, not step budget. The selection-set winner's curse does not reach dev (sound), but best-of-24 picks a genuinely better candidate more often than best-of-4. Fix the row text: "matches the learning-rate choice; B3 retains a 6× candidate advantage, declared as part of its generosity." Also B3's N(0, 1e-3) init versus B2′'s zero init is a stated asymmetry; immaterial at that scale [derived], say so.

**N4. A7 artifact hash as written is not computable at freeze.** `form()` materialises per conditioning set; record the SHA-256 of `generators["paired"].weights` (11 × 4) from `run_formation()` plus the `run_formation` report hash, and assert that at run time.

**N5. §9 permission sentence is inconsistent with §5 and with itself.** "The pilot may fill κ, η_B, the replay rule and raise counts" omits the bottleneck width (§5, §9 item 4) and the P1/P2 magnitude-set revision (§5). "Grid reductions cut seeds before schedules" is a change to B3's definition, which the next sentence forbids. Fix: list width and the magnitude revision among permitted fills; state that the seed count (3 → 2 → 1) is the one permitted definitional change and is decided from timing only. Also declare the P1/P2 alternative magnitude ladder now (κ has one; α and s_min do not).

**N6. No ordering inside pilot items 4–6.** The width search (item 4, reads B3 on P5/P6) and the H′ spread (item 6, reads B2′ too) can be opened together. Require item 4 to be decided before item 6 is opened, so the width is chosen with B3 only in view, as §5 intends.

**N7. Width order {4, 2, 8} is unjustified.** §5's stated failure mechanism (a downstream trained on bottlenecked features may not use the bypass) predicts a wider bottleneck is less likely, not more, to yield the known answer. If width 4 fails for a different reason (a 4-unit BN-ReLU block training degenerately), say so and justify 8. Not a bias, since the trigger is a known-answer failure, but a reader should not have to guess.

**N8. "By construction" survives in §1 and the §5 table** while §5's text says "by information, not by guarantee." Make the table and §1 match the text.

**N9. B4 on P6 needs write-side alignment.** The retrieved (W, b) lives in the donor anchor's z basis on the read side and z̃ basis on the write side; Table B states Hungarian alignment on z only. Descriptive arm, but "retrievable fraction" is uninterpretable without the write-side permutation. State both alignments and the transport W′ = P_outᵀ W P_in, b′ = P_outᵀ b.

**N10. "No code until the freeze" (§11 heading) contradicts the process.** The pilot runs before the freeze (§0) and needs the package. Presumably "no code until John's go on D-A"; say that. Related: the mypy 60 s timeout is "checked in the pilot" (§8), but the gate must pass for the code to land before the pilot; fold the timeout decision into the D-L widening change.

**N11. Tag misuse.** The split-derivation check is tagged **[measured]** ("from a saved artifact") with no artifact in the repo. Either save the check output under `docs/results/` before freeze or retag it. The 5,000-index int32 artifact is "committed as the source of truth" but is not listed as a freeze precondition in §0/§13; add it.

**N12. Sub-partition derivation and seed ranges are not written out.** §3 gives three labels (`sultai-host`, `sultai-conditioning`, `sultai-selection`) for one 40,000/2,000/3,000 partition of the 45,000 fit indices without saying whether that is one permutation sliced or three draws. Development and confirmatory seed ranges are "separate" but non-numeric (the correction protocol gave numbers). Both are needed for "fresh confirmatory seed range" on restart to mean anything.

**N13. Clause 2 scope and A3 ridge.** "P1–P3 anchors" leaves P3′ ambiguous; A3's ridge value is unstated (A2 inherits 1e-8 from `repair.fit`). Note also that calling the stdlib `repair.fit` literally on 512,000 position pairs is pure-Python normal equations, minutes per fit [derived]; implement in torch with a known-answer test against `repair.fit` on a small fixture.

**N14. The bottleneck deficit is never reported.** No line gives L(bottleneck host) − L(intact host) or B3_P6 gain as a fraction of it (the design's recovered fraction). Without it, "0.05 nats" has no scale. Add as descriptive.

**N15. Software validity and scientific outcome fields.** The correction protocol requires them as separate fields; v3's report has V0 (instrument) and V1/V2 but does not say the report records software-valid / instrument-valid / scientific verdict separately. One sentence.

**N16. Bootstrap method unstated.** "One-sided 95% anchor-bootstrap bounds" does not say percentile. See the transparency note below.

### Answers to the explicit questions

*V0 computability and unrelated-reason pass/fail.* Clauses 1–4 and 6 are computable from declared quantities. Clause 1 can fail for an unrelated reason (F6). Clause 3 can fail because the grid is weak on an ill-conditioned known answer while adequate on P6 (N2); that is a related reason by design but is not pre-screened. Clause 5 cannot fail (F5). Clause 6 depends on a pilot-filled tolerance rule, which is declared.

*V1 coherence.* The lower one-sided bound is never above the upper, so "present" and "absent" are mutually exclusive; the three outcomes partition the space. The percentile bootstrap of a median at n = 32 is discrete on midpoints of adjacent order statistics; the one-sided 95% lower bound sits near the 11th–12th order statistic [derived, from the binomial sign-test equivalence], so "present" is effectively "about 21 or more of 32 anchors have H′ ≥ X" and "absent" is effectively "about 11 or fewer do." That is a perfectly good screen rule; the document should state the equivalence for transparency and say "percentile".

*H′ residual asymmetry.* Remaining asymmetries after B2′: candidate count (24 vs 4), init (N(0, 1e-3) vs zero; immaterial), minibatch vs full-batch (a budget axis, intended), step count (intended), and the η range (F4, not intended). The plateau rule is non-circular (it feeds neither selection nor verdict) but is half non-computable (N1).

*Width search and "repairable by information".* The search is triggered by a B3-only known-answer failure, tries a fixed order, and stops with publication; it is not biased by B2′ or H′ provided N6's ordering is added. "Repairable by information, not by guarantee" is honest; the table and §1 are not yet (N8).

*Pilot permissions.* Nothing the pilot may fill leaks into V1 except through count-raising without a rule (F7) and the un-ordered opening of items 4–6 (N6). The rate η_B affects only the reported H, not H′.

*D-M, D-L, D-F consequences.* D-M: F5. D-L: N10, plus the aggregate source hash changes scope (fine, Gate 2's hash is recorded at its commit). D-F: A7's loss to A8 on P3′ has two undecomposed explanations, the former's regression quality and the input-distribution shift from U(−2, 2) synthetic h to post-ReLU-plus-bias real z′; predeclare that V2 does not separate them.

*Standing decisions and Gate 2.* D06, D11, D21, D25 are respected. D15 is respected in letter (X as a worthwhile margin) and violated in process (F1, F7). Gate 2's three lessons (analytic control, matched-rate control, specificity) are taken up as A8, B2′ and the P5/P6 pair; the specificity lesson stops one step short of V1 (F5). The correction protocol's separate software/scientific fields are not named (N15).

---

## Part 3. Sound, one line each

- Partition design: host on 40,000, conditioning 2,000 and selection 3,000 held out, dev the sole endpoint, test batch sealed by an audit hook with a negative control.
- Table A / Table B separation: no intact-target arm appears in a headroom family; nothing privileged enters V1.
- Matched pair on the same host (P5 vs P6) as the known answer, read relatively; the right correction of v1's absolute clause.
- V0 clause 3 validates the grid V1 reads, on a case (P1 α = 0.1) where the tanh family can approximate the inverse [derived], so it tests the optimiser rather than representability.
- Three-way V1 on a bootstrap bound with mutually exclusive outcomes and "no threshold moves".
- A3 as the D21 folding control with A3 ≥ A2 predeclared, so the affine winner is expected rather than discovered.
- V0 failure ends the attempt; fresh anchors and a new version required.
- Determinism contract: full flag set, derived generators, one process per card, replay on both cards and under co-tenancy, hashes over parameters and buffers.
- Ordering rule: timings and determinism before any gain summary; schedule axis never removed.
- Uncertified plateaus retained and flagged, not dropped.
- κ and the P1/P2 materiality rules depend on A1 alone, which no arm's outcome can influence.
- B4 restricted to other anchors on a conditioning batch; descriptive reading predeclared.
- Admission out of scope and stated; harms raw.
- Every arm named as supervised or gradient-based adaptation; no formation claim (D06).
- Cost ledger rebuilt on the minibatch definition and arithmetically consistent [derived]; concurrency and memory stated.
- Gate 2's "no later gate inherits a synthetic pass" respected throughout §1–§2.
- A8 and P3′ turn V2 from a foregone negative into a descriptive in-span comparison; the right call under D-F.

---

## Verdict

**Not ready to freeze.** Seven blockers: F1 (owner numbers must be committed before the pilot's gain summaries), F2 (nonfinite grid candidates must not stop the attempt), F3 (a plateau guard on "absent"), F4 (a cheap-arm rate set not inherited from the oracle), F5 (repair-specificity via the paired P5 null, and a V0 clause that can fail or no clause), F6 (clause 1 at ≥ 90% with declared exclusion), F7 (a spread-only sizing rule). Each is a short textual change; none alters an arm, a family, a margin or the study's shape. The non-blocking items N1–N16 should be swept in the same version. Per §12, the disposition of F1–F7 and N1–N16 must be recorded in v4 before freeze; whether v4 receives a further independent read is John's call. On my reading, a v4 with F1–F7 applied and the owner numbers fixed in the order F1 requires (committed before the pilot's gain summaries are opened) is ready to freeze.
