# Independent Astra review: Gate 3 headroom screen draft

Reviewer: independent Astra agent, 2026-10-11. Read-only; no edits, no runs,
no tracker writes. Reviewed `docs/plans/2026-10-10-gate3-headroom-screen.md`
(draft, uncommitted, SHA-256
`90f1f4a7eded0243bd517442cd62fa9ce5517eaf6a4cc7160e4211553deb7ef0`) against
the correction protocol, the Nyx transfer proposal, the Claude design,
HYBRID_CONTRACT, DECISIONS, the Gate 2 final certificate, and
`src/sultai/formation.py` / `repair.py`.

Verdict: **not ready to freeze.** Six defects block the screen (B1–B5, B7);
one (B6) blocks V2 only. All are repairable without changing the study's
shape. The partition design and the Table A / Table B separation are sound;
the verdict machinery (V0, V1, V2) is not.

Question index: Q1 leakage → B6, N9, N7, sound list. Q2 label access → B2
(selection-access asymmetry), B5, sound list. Q3 outcome-contingency → B4,
B7, N1, N2, N5. Q4 unit of analysis → B3, N3, N7. Q5 does V1 answer the
question → B1, B2, N4, N8. Q6 standing decisions → B4 (D15), N6 (D21), N10
(D25), N8 (Gate 3), D06 sound. Q7 Gate 2 implications → closing summary.

## Blocking

### B1. V0's P5 clause contradicts V1's P8 clause and drops the source design's relative repairability criterion

Claim: the draft demands the oracle gain ≈ 0 on P5 (post-bottleneck site) as
an instrument check, while simultaneously hoping the same 272-parameter module,
fitted the same way, gains ≥ 0.05 nats on an intact host (P8) as "headroom".

Evidence: draft §7 V0 "P5 oracle gain ≤ twin noise floor + 0.01 nats on ≥ 90%
of anchors"; §5 table P8 "Headroom on an intact weak host"; §7 V1 reads P8
with "B3 gain ≥ 0.05 nats on ≥ Y% of anchors". The Claude design (Appendix A
table, P8 row; G0 row) used P8 as the *null calibrator*: "intact-host oracle
gain ≤ 0.01 nats on average", and defined repairable as "oracle gain beats the
intact-host oracle gain by at least 0.05 nats" (design §"A 16-channel host").
The design also notes "a learned stitch can improve a weaker top with no
impairment at all" (Bansal), which is precisely why P5 ≈ 0 is not implied by
the data-processing argument: a module after the bottleneck cannot add label
information, but it can re-map z̃ so the frozen downstream uses it better,
the same stitching mechanism the draft relies on for P8.

Why it matters: either P8 oracle gain is small, in which case P8 is not a
headroom family and V1 on P8 is decorative; or it is material, in which case
P5's absolute ≈ 0 clause will fail by the same mechanism and the whole screen
is declared `instrument_failure` for a reason unrelated to instrument
validity. Both branches are defects. The design avoided this with a relative
criterion the draft silently replaced with an absolute one.

Fix: restore the matched-pair known answer that the bottleneck family was
built for. On the same bottleneck hosts, require (B3 gain at P6) − (B3 gain at
P5) ≥ 0.05 nats on ≥ 90% of anchors; report P5 absolute gain descriptively.
Decide before freeze whether P8 is a null calibrator (then V1 on P6 should
additionally require B3_P6 − B3_P8 ≥ 0.05 so that headroom is repair-specific)
or a headroom family (then say "fine-tuning headroom on an intact host", not
repair headroom, and drop the P5 ≈ 0 clause). Sentence to add: "P5 is judged
relative to P6 on the same host, not against zero."

### B2. The cheap comparator B2 (and A4/A5/B1) is unspecified, so H is a function of an unstated choice

Claim: the primary contrast H = median(B3 − B2) has one side fully specified
(24-point grid, selection-chosen, plateau-certified) and the other side
specified only as "SGD 4 steps ... matched example budget".

Evidence: draft §6 Table A row A4/A5 and Table B row B1/B2: no learning rate,
no batch size, no initialisation (zero or random), no definition of "step"
(full-batch over 2,000 conditioning examples, or minibatch), no definition of
what the budget is matched to. §9 lists "oracle grid" as pilot-tunable; B2's
recipe is not listed anywhere as fixed or tunable. Compare the synthetic
instrument, which fixes `sgd_learning_rate: float = 4.0` and zero init
(`formation.py` `Protocol`, `_sgd`). Compare the correction protocol, which
added `no_growth_lr_matched` precisely because an unmatched-rate control
flattered the formed arm (correction protocol §"Lifecycle and trajectory").

Why it matters: with lr 0.3 and full-batch steps, four steps of a 272-parameter
module may be most of the oracle; with lr 0.01 and minibatches they are
nothing. H moves by tenths of a nat on this choice alone. A positive V1 with a
weak B2 is manufactured headroom; a negative V1 with a strong B2 is the
honest "SGD already does this" reading the design wanted (design §"Pre-
registered readings", STOP row). The Gate 2 negative was exactly an
unmatched-cheap-control story; the draft repeats the setup.

Also: B3 gets 24 draws on the 3,000 selection labels, B2 gets one. H therefore
includes a selection-access benefit, not only an optimisation-budget benefit.
The draft does not say so.

Fix: define B2 fully before freeze: zero init, full-batch gradient over the
2,000 conditioning examples, lr fixed by a predeclared rule (best 4-step lr on
pilot dev anchors' selection sets, chosen before any confirmatory anchor is
trained), and record it in §13. Add a selection-matched cheap arm B2′ = best
4-step candidate over the B3 lr set, chosen on selection, so that H′ = B3 −
B2′ isolates the optimisation budget. Report both H and H′; V1 reads H′.
Sentence to add: "B2's learning rate, initialisation and step definition are
fixed in this table and are not pilot-tunable."

### B3. The noise floor is vacuous on a frozen host

Claim: "A no-op twin on a resampled evaluation order gives the noise floor"
measures nothing in this screen.

Evidence: draft §7; §3 hard rule 1 (all BN in eval during evaluation); §3
rule 5 (deterministic algorithms). With a frozen host, eval-mode BN and no
stochastic layers, per-example cross-entropy is independent of evaluation
order; only the reduction order of the mean changes, at ~1e-7 nats. The twin
is inherited from the design's arm A0′ (design §"Two experiments", rung 1a
table), where it also was mislabelled; the design's rung 1b twin was a
"bitwise-verified" determinism assertion, which is the only thing a twin can
be here.

Why it matters: two V0 clauses ("A1 restores intact loss within the twin
noise floor"; "P5 oracle gain ≤ twin noise floor + 0.01") and the B3 plateau
certification ("all of them agree within the twin noise floor", design
Appendix A) rest on it. With the floor at ~1e-7, A1's float32 inverse
(D⁻¹·D·z ≠ z bitwise) can fail V0 spuriously; plateau certification becomes
either impossible or meaningless.

Fix: keep the twin as a bitwise determinism assertion (must be identical;
failure is a stop). Replace the floor with branch-free quantities so that N3
can be adopted independently: for every *difference* (plateau agreement
among grid points, A1 restoration), a per-example paired bootstrap SE over
the 5,000 dev examples; for any *absolute* null-gain clause that survives
B1, a selection-matched norm-random candidate (the correction protocol's own
control, §"Formation" item 3), which also answers what gain a null procedure
shows after best-of-24 selection. Give A1 an explicit numeric tolerance
(e.g. 1e-4 nats). Sentence to add: "The twin must reproduce no-op loss
bitwise; the noise floor for differences is the per-example paired bootstrap
SE on dev."

### B4. V1 is a point estimate against a threshold with no interval, and X is justified by noise

Claim: the headline verdict carries no uncertainty, and its margin is derived
from an unrelated instrument's resolution.

Evidence: draft §7 V1: "H = median over anchors of (B3 gain − B2 gain).
Headroom present if H ≥ X"; §7 earlier: "all intervals are over anchors (t
interval and cluster bootstrap)", but no interval enters V1. §7: X "proposed
0.02, just above Simic's measured 0.018 resolution"; §4 the same number. That
resolution is for a scheduled-minus-no-growth contrast across training
branches at n = 48 (design §"Simic's instrument fits"), not for a frozen-host
repair contrast. DECISIONS D15: "Choose a task-relevant equivalence/
noninferiority margin independently of observed outcomes and report
uncertainty; noise alone must not define success."

Why it matters: with 16 anchors the bootstrap interval on a median is wide; a
point median of 0.021 vs X = 0.02 is a coin flip dressed as a verdict. Using
instrument noise as the margin is the thing D15 forbids.

Fix: three-way reading on the interval: headroom present if the lower
one-sided 95% bootstrap bound of H ≥ X; absent if the upper bound < X; else
indeterminate. Justify X as a worthwhile repair margin (an owner decision
under D15) and state it in nats of cross-entropy relative to the host's
loss. Also repair the verdict logic: with Y = 50 and 16 anchors, "median ≥ X"
and "B2 within X of B3 on ≥ 50% of anchors" can both fire at an 8/8 split;
decouple Y from the no-headroom clause. Sentence to add: "V1 is read on the
anchor-bootstrap interval of H, not on the point estimate."

### B5. The oracle actually used for V1 (B3) is never validated on a known answer

Claim: V0 validates A6, a 2-point grid with one seed and one schedule; V1 uses
B3, a 24-point grid with plateau certification. The procedure that produces
the headline number is never shown to find a repair where one is known to
exist.

Evidence: draft §6 Table A row A6 ("reduced grid: 1 seed × lr {0.1, 0.03} × 1×
schedule"); Table B row B3 ("full grid: 3 seeds × lr {0.3, 0.1, 0.03, 0.01}
× schedule {1×, 4×}"); §7 V0 gates on A6 only; §8 ledger runs A6 on P1–P3
and B3 on P5/P6/P8 only.

Why it matters: a negative V1 ("no headroom, stop the programme cheaply") is
safe only if B3 is at plateau and is known to find repairs that exist. The
draft asserts plateau certification but (B3) rests it on the vacuous twin and
never tests the grid on a case with a known inverse. A6 failing V0 for being
too weak (two learning rates, one seed) would also abort the screen for a
reason unrelated to the host.

Fix: run the B3 grid on at least one post-hoc family (P1 α = 0.1, 64 hosts ×
24 fits = 1,536 fits, 1–2 GPU-h at the draft's own 2–4 s per fit) and gate V0
on it:
B3 gain ≥ ρ · A3 gain on ≥ 90% of cases, ρ an owner decision (A3 is the
affine supervised fit, which can represent the known inverse; see N6). Retire
A6 or keep it as a cost diagnostic. Sentence to add: "V0 validates the same
grid that V1 reads."

### B6. V2 is a foregone negative, and the control that decided Gate 2 is missing (blocks V2 only)

Claim: the former's output space cannot represent any of the P1–P3 inverses,
so "does not transfer" is guaranteed by construction, not measured. The
within-family comparator Gate 2 used (analytic four-template fit) is absent.

Evidence: `formation.py` `public_templates()`: the former emits W = c₁·I +
c₂·shift + c₃·alternating-shift and b = c₄·(+1,−1,+1,…), four coefficients
(`materialize`, `FrozenGenerator.form`). HYBRID_CONTRACT Assay A: "its
effective output dimension is four despite a materialized 272-parameter
adapter." Draft §5: P1 attenuates g = 4 channels (a random channel group per
design Appendix A; the needed W is diagonal on 4 specific channels, outside
span{I, shift, alt-shift}); P2 is a Haar-random mixing per design Appendix A
(dense, outside the span); P3 is a per-channel bias at fixed norm (random
direction per design Appendix A, outside span{alternating}). Draft §7 V2
compares A7 to A2, the full 272-parameter tanh ridge. Gate 2 certificate,
Scientific disposition: "Paired evaluation mean admitted MSE is approximately
0.000273462, versus analytic 1.93e-23"; the correction protocol §"Formation"
item 1 defines that analytic four-template control. Draft Table A has no such
arm. The draft also does not name the generator artifact A7 loads ("no
refit"): `docs/results/gate2-confirmation-2026-10-10-01a125af/` holds no
saved weights (its `historical_generator_parameters` field is parameter
counts 28/44/44, inspected); the generator is re-derived in-process from
seeds 1000–1031 by `run_formation`, and the evidence mode (coarse / paired /
paired_probes) is unstated.

Why it matters: V2 as written cannot produce information. A7's input
distribution is also far out of the meta-regression's training range
(post-ReLU z ≥ 0 vs. h ~ U(−2, 2)), which would be a legitimate transfer
failure, but it is masked by the representability failure. The draft says
V2 does not change V1, so this blocks V2 as a verdict, not the screen.

Fix: either drop A7/V2, or (a) demote V2 to a descriptive arm, (b) add the
analytic four-template least-squares fit on real position pairs as A7's
matched-family comparator (the comparison Gate 2 actually ran), (c) add one
in-span known-answer case (P3 with c ∝ (+1,−1,…)) so representability is
controlled, and (d) name the generator artifact: re-derivation command, seed
range 1000–1031, evidence mode `paired`, and the SHA-256 of the materialised
weight matrix recorded in this document and asserted at run time.
Sentence to add: "V2 compares A7 to the analytic four-template fit on the
same pairs; the full ridge A2 is reported as the family ceiling."

### B7. V0 failure permits a fix-and-reread loop on the same confirmatory anchors

Claim: "Fail → instrument_failure; fix host or fits; no scientific reading"
allows tuning fits until V0 passes and then reading V1 on anchors whose
results have already been seen.

Evidence: draft §7 V0 last bullet; §10 "Restart is a new attempt with linked
provenance and no budget reset" does not require fresh anchors or a new
preregistration version. Correction protocol §"Execution and stop rules":
"Corrections to bugs invalidate affected prior evidence and require an
explicit disposition; never silently recycle it"; "Do not ... alter
confirmatory settings to continue."

Fix: a V0 failure on confirmatory anchors ends the attempt; any change to
host, impairment or fit code requires a new preregistration version (new
SHA-256), a new disposition, and a fresh confirmatory seed range. Sentence to
add: "Confirmatory anchors are never reused after a V0 failure."

## Non-blocking, ranked

### N1. The pilot may change the oracle grid after seeing pilot H, and cost-driven grid reduction contradicts plateau certification
Draft §9: "Pilot results may change counts, spawn point, oracle grid and
replay rule"; §8: "if the pilot shows B3 fits above 4 s, the schedule 4× arm
is the first candidate for reduction." Removing the schedule axis makes B3 a
non-plateau and biases V1 toward "no headroom"; expanding the grid after
seeing pilot (B3 − B2) biases the other way. Fix: grid changes are decided
from pilot timings only, before the pilot's gain summaries are read; cut
seeds before schedules; B2's recipe is fixed by the same rule (B2).

### N2. The pilot cannot size the study
Draft §9: 2 bottleneck and 4 intact development anchors; §4: the pilot's
"per-anchor spread of (B3 − B2) on P6 and P8" decides whether anchors are
added. An anchor-level SD from n = 2 is not an estimate. Fix: either more
development anchors (≥ 6 per family) or drop the sizing claim and size from
a declared prior SD.

### N3. Branches buy nothing on a frozen host; 64 independent anchors cost the same and give n = 64
Draft §4: 16 anchors × 4 branches for intact hosts, anchor as the unit. No
arm continues training after spawn, so common-future pairing (the reason
Simic uses branches) is absent; all arms are evaluated on the same frozen
host, so pairing is automatic. Branches are correlated replicates that reduce
n from 64 to 16. The basin check "at spawn epoch 3" (§9 item 3) is also
vacuous as worded: branches are the same checkpoint at spawn; the barrier
must be measured between branches at the end of training. Fix: one branch
per anchor, 64 anchors, unless a declared purpose for branches (e.g. a later
rung 1b, or a within-basin vs across-basin retrieval diagnostic) is written
down. Compatible with B3's branch-free floor. Read together with N2.

### N4. The cheap-vs-oracle contrast has no cost axis
Draft Table B has k ∈ {1, 4}; the design had k ∈ {1, 2, 5, 20, 100}. If
SGD(16) ≈ B3, a positive V1 over 4 steps is closable conventionally at
trivial cost. The design's break-even N* is dropped. Fix: add one
intermediate k and a predeclared "steps-to-match-B3" report, and report the
absolute wall cost of B3 per host; "headroom" realised by a 24-fit grid in
about a minute is not the same programme signal as headroom that needs a
corpus.

### N5. No predeclared rule combines P6, P8 and P10 into the programme decision
Draft §7: "No Bonferroni across families; each family's V1 is a separate
declared question"; §1: "If so, building a generator corpus is justified."
If any family positive triggers "go", the family-wise false-positive rate is
inflated. Fix: name P6 (repairable by construction) as the primary family;
P8/P10 are secondary and reported.

### N6. A1 lives in the affine family; the primary tanh module cannot represent linear inverses
For P1/P2 the exact inverse is u = D⁻¹z′, affine in z′ (Table A row A3), not
of the form z′ + W tanh(z′) + b. A1 therefore validates the host and
impairment plumbing, not the tanh module; A2 and A6 are expected to fall
short of A1 by a representability gap, and A3 should beat A2 on P1–P3. The
draft does not predeclare this; a reviewer would otherwise discover it as a
surprise and the D21 control would look like the winner. P3's "fixed norm"
is unstated, so the 0.05-nat V0 clause may be unreachable even by A1. Fix:
restructure V0 as (i) A1 gain ≥ 0.05 on every case (impairment is material),
(ii) A3 restores within tolerance of A1 (supervised fit in the right family
works), (iii) B3 ≥ ρ·A3 (gradient oracle in the primary family recovers a
declared fraction); state the P3 norm; predeclare A3 > A2 on P1–P3 as
expected under D21.

### N7. Anchor intervals condition on one fixed data draw
All anchors share the same 40,000/2,000/3,000/5,000 examples; the anchor
interval captures init and order variance only, not conditioning-set or
dev-set sampling. Simic's host recipe, copied "as-is" (§3), was tuned against
the same 5,000 dev examples that are this screen's endpoint. Neither is
fatal for a screen, both must be disclosed. Fix: add a per-example paired
bootstrap over dev as a second interval; state that the recipe was selected
on the endpoint split.

### N8. The screen states its Gate 3 clause imprecisely
Correction protocol Gate 3: "report corrected within-family comparisons.
Beyond-template transfer remains unproven." A positive V1 establishes that
real-host headroom exists for a gradient-fitted 272-parameter module; it
does not establish beyond-template transfer of a learned former, and B4 (the
only arm touching whether repairs are predictable across lineages) is a
diagnostic with no declared reading. Fix: state which clause the screen can
close; give B4 a predeclared descriptive reading (retrievable fraction = B4
gain / B3 gain per anchor); replace "building a generator corpus is
justified" with "a corpus study is not ruled out".

### N9. Retrieval B4 is underspecified and may retrieve from same-anchor branches
Draft §6 B4: "nearest other-lineage oracle repair". If "lineage" admits
another branch of the same anchor, B4 retrieves a near-twin's repair and
"repairs are shared across lineages" is trivially true. The distance metric
and the partition of the "fixed unlabelled batch" for Hungarian alignment
are unstated; if that batch comes from dev, it is an unlabelled use of the
endpoint in a fitting step. Fix: "other anchor", metric named (the former's
`_retrieve` uses paired moments), alignment batch from conditioning only.

### N10. No admission step; raw-vs-admitted convention dropped
Every arm's candidate is inserted raw; the harm screen then reports harms an
admission step (D25, HYBRID_CONTRACT "Report raw and admitted outcomes")
would have blocked. Acceptable for a headroom screen if stated. Fix: one
sentence saying admission is out of scope and all harms are raw; or add
admitted outcomes for B1/B2.

### N11. Builder gaps the document must close before freeze
Impairment draw granularity (per anchor, per branch, or per case) for P1
channel group, P2 U/V, P3 direction; BN rule should cover every BN layer
(stem, producer, bottleneck), not only "downstream"; "1× schedule" and
"step" undefined; B5 module nonlinearity; A7 evidence mode and artifact (B6);
V0's "90% of cases" mixes case and anchor units; uncertified-plateau anchors
are retained and flagged, not dropped; anchor-to-card assignment fixed by
derivation (§3 rule 4 vs §8 two-card wall estimate); the confirmatory run
must not import Simic code at run time (D11), the split-hash check is a
one-time offline comparison; "counts" in §9 must exclude conditioning and
selection sizes, which B2 depends on.

## Sound, one line each

- Partition design: host trained on 40,000; module fitted on 2,000 held out
  from host training; selected on 3,000; dev is the sole endpoint; test
  batch sealed with a test. Sound.
- Selection-set winner's curse on B3's best-of-24 does not reach the dev
  endpoint. Sound.
- D06: every arm named as supervised or gradient-based adaptation; no
  formation claim. Sound.
- Margins fixed and the document committed before the pilot; development
  anchors from a separate seed range, never reused. Sound.
- Anchor as the unit, branches not counted as independent, no branch on both
  sides of a split. Sound (but see N3).
- Table A / Table B separation: no intact-target arm appears in a headroom
  family; the "two regimes" framing hides no privileged comparison in V1.
  Sound. V2's privilege problem is representability (B6), not label access.

## Gate 2 implications the draft ignores, summarised

1. Analytic four-template fitting beat the former; the draft keeps the former
   and drops the analytic control (B6).
2. The correction protocol added a rate-matched control because an unmatched
   cheap control flattered the formed arm; the draft's cheap control is
   unspecified (B2).
3. "Endpoint recovery does not establish specificity": the analogue here is
   that P8 fine-tuning gain does not establish repair; the draft reads it as
   headroom without a relative criterion (B1, N6).
4. Gate 2's "no later gate inherits a synthetic pass" is respected (§1, §2).

Tracker note: read-only review; no filigree claim, transition or observation
was made. Parent records this review and its disposition in `docs/reviews/`.
