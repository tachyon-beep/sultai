# Capacity frontier screen on the CIFAR-10 host — DRAFT PREREGISTRATION v3 (programme rung M0)

Status: **draft v3; decisions J0–J9 taken on 2026-10-13; authorises no
execution.** v1 was committed at `5aeee56` and reviewed by three independent
readers; v2 (`20dee97`, `fdef5fc`) adopted every finding. On 2026-10-13
twelve agents each took one of John's decisions and wrote a recommendation
with the case for and against; a synthesizer reconciled them against source
([`docs/reviews/m0-capacity/j-recommendations/SYNTHESIS.md`](../reviews/m0-capacity/j-recommendations/SYNTHESIS.md)).
John then said "ok go with the recommendations, recut the draft to v3". This
is that recut: §12 records each decision as taken, with its source report,
and §2 and §8 absorb the sixteen draft defects the synthesis listed. The
freeze waits on John's acceptance of the
[gate-ladder amendment](2026-10-13-gate-ladder-amendment.md) text and on
J10 (merge). Written by `sultai-claude-nyx` under the
[steering statement](../STEERING.md). Values marked [taken] were fixed by
John's instruction; [proposed] remain the author's; [measured] come from
Gate 3 attempt-4 receipts; [pilot-fill] are filled by the pilot under §9.

## 1. The question, in the programme's units

In Gate 3 attempt 4 a 272-parameter module fitted by ordinary loss gradient
lowered development loss on every **impaired** host it was tried on: by
0.037 to 0.260 nats on the bottleneck hosts whether or not it could read the
pre-bottleneck tensor, and, on the attenuated hosts, to 0.048 to 0.207 nats
below the intact host's own loss
([hold table](../certificates/2026-10-13-gate3-linked-pilot-3-hold.md)).
No fit on an unimpaired host was ever run; arm M-sq below is that first
direct measurement. The scientific review of v1 showed what those numbers
look like: six hosts whose intact losses span 0.164 nats were compressed by
the module onto a 0.035-nat floor, with gain monotone in the host's starting
loss. That is the signature of **calibrating or finishing** an unconverged
constant-rate host, as much as of capacity the host lacks. The host has
284,634 parameters, trains for 10 epochs at a constant rate, and the module
is 0.1 percent of it. So the study is built so that a module cannot read as
"capacity" unless it beats, on the same data, every cheaper route to the
same loss.

**Q1, efficient added capacity.** Fitted on examples the host already
trained on, does a 272-parameter nonlinear module at the site buy more loss
than (i) recalibrating the host's BatchNorm statistics and its logits with
zero or eleven parameters, (ii) a 272-parameter **affine** module that folds
into the next convolution exactly away from the image border (a measured
residual remains on the border ring, §6) and so adds no retained capacity,
(iii) re-tuning 272 or 170 of the host's existing parameters at the site,
(iv) a decayed-rate finish of the host's own training, and (v) the same
module present from step zero? And where does the result sit against the
width frontier of the host family at the same compute?

**Q2, add and remove (Sultai versus the plug-in route).** A module trained
in isolation and blended at α = 1 is Esper-lite's plug-in route in
miniature. Sultai's route overgrows and physically cuts. Does a 64-hidden
module (2,128 parameters) fitted and then **sliced** to 8 hidden units (280
parameters) keep more of its gain than a 280-parameter module built small,
at matched optimiser budget (matched optimiser steps and example passes,
§4)? The built-small arm is also the cut-down architecture retrained from
scratch (the Liu et al. control). This is the truncation precedent for
Sultai's thesis, not the developmental-removal question, which belongs to
rung M2.

A positive V1 says there is site capacity that ordinary SGD finds and no
cheaper route reproduces; that is what a generated builder would later have
to find more cheaply (rung M1). A negative V1 says the reference host's site
is not where capacity is missing, and the next screen moves to the starved
host that this study's frontier sweep already trains. Site S1 is a
privileged, known choice inherited from Gate 3; site discovery is rung M2's
problem. Either verdict is a published finding.

## 2. What carries over, what is new code, what is retired

**Carried over unchanged** (verified against source by the reality
reviews): host architecture and recipe at s = 1.0 and 10 epochs; the data
seal (test batch never read) and the 40,000 / 2,000 / 3,000 / 5,000
sub-partition; the padded fixed-shape evaluator; the frozen-host hash
check; `losses()` on any two- or four-tensor adapter of any hidden width;
the plateau rule's form (its step indices change); the per-example paired
bootstrap; the harm screen (Hm = 0.01 nats); custody receipts, ledgers and
GPU-time accounting; the P1 known answer **mathematics** (A1, A2, A3, B3 at
α = 0.1; B3 ≥ ρ · A3 gain, ρ = 0.8) as instrument validation only.

**New code, each with its own exact-source review before the pilot.**

*Hosts and training.* A tail-width constructor parameter that accepts an
explicit width tuple (not only s ∈ {0.25, 0.5, 0.75, 1.0, 1.5}), keeps the
bottleneck argument, and leaves the s = 1.0 layer order, parameter and
buffer names and RNG consumption bitwise identical (the state hash covers
kind, name, dtype and shape, so any renaming breaks the attempt-4 replay);
the attempt-4 `final_state_sha256` strings pinned in `src/` so V0 compares
a fresh receipt to a pinned string rather than to another fresh receipt;
seed admission beyond 3000–3005 (`main()` refuses other seeds today) and a
training check that reads the width tuple, not only the bottleneck width;
saving optimiser state and a resume path with a decayed-rate schedule
(T-cool); an epoch count other than 10 with a named decay for the extended
20-epoch point; the published `final_state_sha256` of every frontier host.

*Fitting.* The oracle grid's rate set, checkpointed single-run schedule,
hidden-width parameter, warm start and the **affine SGD path** (`fit_sgd`
is nonlinear-only today; only the closed-form ridge has an affine option);
the plateau comparator at 800 versus 3,200 steps (1,600 versus 6,400 for
M-small-B) with the last two 40-step windows; the re-tuning fit paths
(declared parameter subset trained, complement hashed; arms that touch
`front` re-run the site from images every step); BatchNorm recalibration
and logit calibration; a static-module host; physical slicing with the mean
fold, unit ranking and the slicing known-answer test; the exact fold of a
fitted M-aff into `tail[0]` and its residual evaluation (§6); a
**dev-accuracy evaluator** (no argmax code exists under `src/` today) for
the ΔAcc secondary.

*Instrument carve-out* [taken, J-retire]. Retiring the Gate 3 families is a
subtractive code change, not prose: today `materiality_worker` ladders P1
with P2 and P3 and `magnitude_decision` can hold the pilot on a P3 replay
or a P2/P3 exhaustion; `posthoc_worker` runs P1, P2, P3, P3′ and the former
arms A7/A8 in one loop and `check_posthoc` requires all four families and
former agreement before it reads B3 ≥ ρ · A3; `timing_worker` probes B4 and
B5; `costs.project_cost` projects the Gate 3 arm set; six tests in
`tests/test_screen.py` and all of `tests/test_v42_ladder.py` pin retired
families. M0 therefore carries **P1-only** materiality, post-hoc and check
functions, a replacement cost projection, and the test rewrite, with
`former_bridge.py` and `alignment.py` off the new workers' import path.
Nothing retired feeds the P1 numbers.

*Supervision.* New worker actions, identity pins and tests; the M0 pilot
ceiling as a **new constant** (`PILOT_SECONDS` stays, because the Gate 3
ledgers are checked against it); a confirmatory ceiling constant with its
own supervisor instance (nothing enforces a confirmatory ceiling today; the
sizing receipt only reports `within_ceiling`); a one-process-per-card
scheduling constraint; the anchor-bootstrap, sign-test and verdict
aggregation. A stage that depends on new code reads nothing until that
code's review is preserved.

**Retired from this study** [taken, J-retire]: P2, P3, P3′, P5, P6, the
former arms A7/A8, retrieval B4 and the cheap bank B1/B2/B2′/B6. They stay
published Gate 3 evidence. "Retired" means no verdict, report or
confirmatory role in M0 and no presence on the M0 workers' code path; the
Gate 3 workers themselves are not deleted. Reversal triggers, recorded by
the shaping review: P5/P6 return only at a starved-host screen if M0 is
null on both the reference and 0.5 hosts; A7/A8, B4 and the cheap bank
return at rung M1 as its comparators; P2/P3/P3′ return only as a diagnostic
if V0 fails B3 ≥ ρ · A3.

## 3. Hosts, frontier, site and modules

**Reference host** (v4 §3): 284,634 parameters, tail widths (64, 64, 128,
128). Development anchors reuse seeds 3000–3005; the freshly trained hosts
must reproduce attempt 4's published `final_state_sha256` strings, pinned
in `src/` (no Gate 3 checkpoint is loaded).

**Frontier family** [taken, J4]: the same architecture with the four tail
widths scaled by s; the front (stem and producer) is a fixed 14,928
parameters at every s. Parameter counts recomputed from the constructor on
CPU by the reality review.

| s | Tail widths | Parameters | Set |
|---|---|---:|---|
| 0.25 | 16, 16, 32, 32 | 33,978 | extended; the first halving outside the 25–40 percent band D02 rejected, so the frontier reaches below rung M2's near-empty start |
| 0.5 | 32, 32, 64, 64 | 85,274 | core; the starved host of the fallback screen. At 30 percent of the reference it is **not** near-empty in the sense of D02 (J-D02) |
| 0.75 | 48, 48, 96, 96 | 168,826 | extended; a linearity check on the slope |
| 1.0 | 64, 64, 128, 128 | 284,634 | core; the six reference anchors are these points |
| 1.5 | 96, 96, 192, 192 | 613,018 | core; the leverage point above the reference |

Every frontier host trains under the unchanged recipe for 10 epochs
(400,000 example passes). Disclosure from v4 §3: the recipe was tuned by
Simic for the 1.0 host against the same 5,000 dev examples, so this is a
**same-recipe frontier, 10 epochs**, and every reading that cites it says
so. The extended set adds a 20-epoch cooled 1.0 point [proposed]: the
unchanged recipe for 15 epochs, then the rate decayed linearly to zero over
epochs 16–20. It measures a compute doubling at matched architecture; it
never enters the same-recipe slope, and finishing is answered by T-cool,
not by it. Every frontier host publishes its `final_state_sha256`.

**Site and modules.** Site S1 reads z (16 × 16 × 16) after the producer;
nothing sits between S1 and `tail[0]`, a Conv2d(16, 64, 3, padding 1)
whose BatchNorm runs in evaluation mode under frozen execution.

| Name | Form | Retained parameters | Role |
|---|---|---:|---|
| M-sq | u = z + W tanh(z) + b | 272 | the capacity reading; plug-in route |
| M-aff | u = z + W z + b | 272, **folds to 0 off the border** | primary matched-count control. M-sq and M-aff are the same two-tensor adapter differing in one flag, so the count match is by construction. The fold K′ = K(I + W) is exact everywhere (1e-6 on CPU); the bias fold is exact in the interior but over-adds on the one-pixel zero-padded border ring (60 of 256 positions, 23.4 percent), so a fitted M-aff retains a 16-degree-of-freedom border bias the host cannot express. The residual is measured per anchor (§6) and bounded in V0 (§7) |
| M-wide | u = z + V tanh(U z + c) + d, h = 64 | 2,128 | overgrown |
| M-wide→h | M-wide sliced to h ∈ {16, 8, 4, 2} | 33h + 16 = 544, 280, 148, 82 | add-and-remove |
| M-small | the h = 8 form built from scratch | 280 | built small; cut-down architecture retrained |
| M-exp (extended) | W = W₁W₂, W₁ 16×64, W₂ 64×16, contracted exactly to M-sq after fitting | 272 | the true ExpandNets analogue |

At zero weights every module is bitwise identity with the host.

**Physical slicing** [proposed]: rank hidden units by contribution
‖V[:, i]‖₂ · RMS over the fit examples of tanh(U z + c)_i; remove the lowest;
slice U, c, V to the survivors; fold the removed units' means exactly into
the bias, d ← d + Σ_removed V[:, i] · mean(a_i); print the parameter count.
Report the unfolded slice beside it. Controls: a random cut to the same h
with the same fold, and a leave-one-unit-out loss ablation on the fit
examples as the ceiling for the cheap criterion. V0 carries a slicing known
answer (§7). This slicing is module-only; removal of host channels across a
convolution and its BatchNorm is rung M2's new code (J-D02).

## 4. Data regime, fitting and compute booking

**Primary fit data is seen data.** Every module and re-tune arm takes its
gradient steps on **2,000 examples drawn by a fixed seed from the host's own
40,000-example training partition** (new sub-partition label
`m0-seen-fit-v1`, derivation stated in the implementation plan). Candidates
are chosen on the unchanged held-out 3,000 selection examples; the 5,000
development examples are the endpoint. So no arm sees data the host did not
train on, the frontier, static and finishing hosts are data-matched to the
module arms, and the one advantage the module arms keep over host-level
points is candidate selection on 3,000 held-out examples, declared here.
The held-out 2,000 conditioning examples serve two arms only: the P1
instrument (mathematics unchanged from Gate 3) and M-sq-fresh (§5), whose
job is to measure the fresh-data effect directly.

**Published no-op.** For each anchor, N0 is the host after BatchNorm
recalibration on the fit examples (R-bn, no gradient, one pass) if its
selection loss is lower than the raw host's, else the raw host; the choice
is made on selection, never on dev. Every arm is fitted on N0 and every gain
is relative to N0, so no module can collect the recalibration gain as its
own. Raw-host loss, R-bn loss, and the one- and eleven-parameter logit
calibrations R-temp and R-cal (temperature; temperature plus class bias,
fitted on the fit examples) are reported beside.

**Oracle grid** [proposed]: minibatch SGD (batch 256) on the arm's
parameters, init N(0, 1e-3) for modules, one seed, rates {1, 0.3, 0.1} for
nonlinear module arms (attempt 4 selected the top rate 0.3 in 28 of 30
grids [measured]) and {1, 0.3, 0.1, 0.03, 0.01} for re-tune arms **and for
M-aff** [taken, J-route], because M-aff's features are the unbounded
post-ReLU z rather than tanh(z), so the module grid may be off-scale for
it and a badly fitted control must not be the reason V1 reads anything.
One 3,200-step run per rate with selection checkpoints at 200 and 800
steps (bitwise the same as separate runs under a seed-derived stream). Best
on selection. The selected rate is reported with an **edge-of-grid flag**.
Plateau is certified by the v4 Table B rule applied to the 800- versus
3,200-step checkpoints and the last two 40-step training windows; for
M-small-B the comparator is 1,600 versus 6,400 steps. Attempt 4 certified
2 of 30 oracle grids at 800 steps (anchor 3003, both families, width 4)
[measured]; the pilot may extend the schedule axis, raise only. R-all, run
to 800 steps only, is never plateau-certified and enters no certification
fraction.

**Warm starts and matched budget** [taken, J-D2]. M-wide→h refit uses the
sliced tensors as the start and the full module grid; its step-0 selection
loss (the as-sliced loss) and its selected rate and step with the
edge-of-grid flag are reported beside D2, so a null D2 can be read as
"refit erased the slice" or "slice equals built-small". M-small-B is
M-small at twice the schedule ({400, 1,600, 6,400}). **"Matched budget"
means matched optimiser steps and example passes**: the overgrow route
spends 3 × 3,200 steps on M-wide plus 3 × 3,200 on the refit, 19,200 steps
and 4,915,200 example passes per anchor; M-small-B spends 3 × 6,400 =
19,200 steps and 4,915,200 passes. Measured wall per step is flat in module
width (4.68 ms for the 272-parameter module against 4.64 ms for h = 64
[measured]), because the cost is the host forward and backward, so matched
steps is matched wall and matched FLOPs (the Scratch-B convention).
Parameter-steps are **not** matched and the ratio is disclosed in the
ledger: 23.1 M for the overgrow route against 5.4 M for M-small-B, 4.3 ×;
7.6 × for the strict pair.

**Booking** (v4 §6 units, per arm and anchor): example passes, optimiser
parameter-steps, wall seconds, device, physically retained parameters.
R-aff and R-all re-run the front per step and are booked in both
parameter-steps and wall; nothing is netted. **Every kernel in §8 is a
one-process-per-card number**: attempt 4 charged co-tenant hosts at about
102 s against 44.5 s solo, and the supervisor charges wall per process, so
the pilot and the confirmatory schedule at most one process per card and
every projection uses one-per-card kernels.

## 5. Arms

All arms on each development anchor of the reference host. Gains G are in
nats of mean dev cross-entropy relative to N0; dev accuracy is reported
beside every loss as a labelled secondary (J0).

**Core set** (bold arms enter a verdict):

| Arm | Definition | Retained added | Separates |
|---|---|---:|---|
| N0 | published no-op (§4) and its bitwise twin | 0 | zero and noise floor |
| R-temp, R-cal | logit temperature; temperature + class bias, fit examples | 1, 11 | calibration |
| **M-sq** | square tanh module, module grid | 272 | the reading |
| **M-aff** | affine module, five-rate grid; folded into `tail[0]` after fitting and the residual measured | 272 → 0 by folding, off the border | nonlinearity versus re-tuning |
| R-272s | re-tune the 3×3 centre taps of **output channels 0–15** of `tail[0]` over its 16 input channels (256 entries) plus those channels' 16 biases, 272 in all; re-tune grid; cached site. (The v2 definition "16×16 centre taps and 16 output biases" did not exist: `tail[0]` has 1,024 centre taps and 64 biases) | 0 | structured equal-count re-tune in another basis |
| R-aff | re-tune producer BN affine (32), first downstream BN affine (128), classifier bias (10) = 170, re-tune grid; re-runs the front | 0 | the classic cheap fine-tune route |
| M-sq-fresh | M-sq fitted on the held-out 2,000 conditioning examples instead | 272 | fresh-data effect |
| **T-cool** | resume the host from its saved optimiser state for 5 epochs on the training partition with the rate decayed linearly to 0; no module | 0 | finishing |
| **M-sq \| T-cool** | M-sq fitted on the T-cool host | 272 | does the module survive finishing |
| **M-wide** | h = 64 module, module grid | 2,128 | overgrown |
| **M-wide→8** | sliced to 8 by contribution rank and by random rank; as sliced with fold; and refit with the full module grid | 280 | add-and-remove |
| M-wide→{16, 4, 2} | as sliced with fold; one refit at the M-wide rate, 800 steps | 544, 148, 82 | survival curve |
| **M-small** | h = 8 built from scratch, module grid | 280 | built small (plug-in) |
| **M-small-B** | M-small at twice the schedule | 280 | built small at the overgrow route's budget |
| S-static | host trained from scratch under the **anchor's own seed** with M-sq present from step zero at W = 0, b = 0 (host init and data stream bitwise identical to the anchor; W under the recipe's weight decay); then the module grid refitted on the frozen static host | 272 | design-in versus insertion |
| F(s) | frontier hosts, s ∈ {0.5, 1.0, 1.5}, 6 seeds each | by width | scale |
| P1 instrument | A1, A2, A3, B3 on P1 at α = 0.1, held-out conditioning | n/a | the fitter works |

**Extended set** [taken, J4, J-host, J-D02; ceiling B under J6], run in
this priority order only while measured kernels leave room under the
ceiling (§8), each descriptive with no verdict role:

1. **The 0.5-host pair, enlarged**: N0 and calibrations, M-sq, M-aff,
   T-cool and M-sq | T-cool on the six 0.5 frontier hosts, reporting D1 and
   D3 bounds on the starved host. One host cannot separate "nothing is
   missing at S1 on this host" from "a module at S1 never adds capacity";
   D3 read on both hosts can (a finishing harvest predicts D3 below X on
   both; capacity predicts D3 at or above X after T-cool on at least the
   starved host), and it names the population rung M1 would train on. The
   six 0.5 hosts then become development anchors never reused for a
   starved-host confirmatory set. *Provenance*: the leverage review added
   T-cool and M-sq | T-cool to the v2 pair; John was told these were new
   arms and did not discount them.
2. R-all, re-tune every host parameter, one seed, rates as re-tune arms,
   checkpoints every 40 steps to 800, BN recalibrated after fitting,
   re-runs the front.
3. M-exp.
4. Lottery cut, the surviving 8 units' initial weights retrained with the
   module grid.
5. The 20-epoch cooled F(1.0) (§3).
6. F(0.25), six seeds, so the frontier reaches below rung M2's near-empty
   start. *Provenance*: added by the J-D02 shaping review; same flag as
   item 1.
7. F(0.75), the linearity check.

## 6. Quantities

Per anchor, with per-example paired-bootstrap SE over the 5,000 dev
examples (v4 §7), for loss and, as a secondary, for accuracy:

- **D1 = G(M-sq) − G(M-aff)**, the nonlinear module against the foldable
  affine module, equal count, same data, selection on the same examples.
- **Fold residual** r_aff = L(host with `tail[0]` replaced by the folded
  weights) − L(host + fitted M-aff) on dev, per anchor; the measured size of
  the border bias M-aff retains. It is reported beside D1 and bounded in V0.
- ΔAcc = acc(M-sq) − acc(N0), the accuracy secondary, with a one-sided
  bound; it conditions no verdict (J0).
- D1′ = G(M-sq) − max(G(R-272s), G(R-aff)), against every named re-tune.
- **D3 = L(T-cool host, after its own N0 rule) − L(M-sq \| T-cool)**, the
  module's gain over the finished host, not over the anchor's N0; and the
  finishing gain G(T-cool) itself, relative to the anchor's N0. D3 is a
  single-arm gain, not a two-arm difference, and is read at its achieved
  half-width.
- D4 = G(M-sq-fresh) − G(M-sq), the fresh-data effect.
- D5 = L(host + M-sq) − L(S-static + refit), negative when insertion beats
  design-in; also without the refit. Paired by data stream, not by host, so
  its noise is the host's seed-to-seed spread; the pilot records SD(D5)
  before any median (§9).
- **D2 = G(M-wide→8, contribution rank, refit) − G(M-small-B)** at matched
  budget (§4); D2s = G(M-wide→8, as sliced with fold) − G(M-small), the
  strict reading; the random-rank versions of both; the refit's step-0
  selection loss and selected rate beside D2; survival curves
  G(M-wide→h)/G(M-wide) for every h, sliced and refit; the leave-one-out
  ceiling.
- Calibration: G(R-temp), G(R-cal); re-tune fraction G(M-sq)/G(R-all) when
  R-all runs.
- **Frontier**: slope in nats per doubling of parameters from the **three
  core widths only** (0.5, 1.0, 1.5) with a seed-bootstrap interval (the
  reality review puts the slope SE near 0.012 nats per doubling at the
  measured host spread); the module's gain G(M-sq) expressed as a fraction
  of one doubling with its interval. If F(0.25) or F(0.75) run, a second,
  descriptive fit over every 10-epoch width is reported beside, labelled as
  such; the 20-epoch point never enters either. No extrapolation beyond
  613,018; no ratio to 272; label "same-recipe frontier, 10 epochs".

Across anchors: median with one-sided 95 percent percentile
anchor-bootstrap bounds (v4 §7), and an exact sign-test bound beside it at
confirmatory n. In the pilot only the per-anchor SDs are read before any
median (§9).

## 7. Verdicts (margins [taken], fixed before the pilot)

**V0 instrument.** Replay bitwise on both cards against the pinned
attempt-4 hashes; A1 restores within 1e-4; B3 ≥ ρ · A3 gain on every
development anchor; zero module bitwise identity on every family; every
sliced module's printed count equals 33h + 16; the slicing known answer (a
64-unit module with 56 columns of V exactly zero must, after the criterion
cut to 8 with fold, reproduce its output within 1e-6, a random cut must
not, and a negative control must fail); the complement hash unchanged
under every re-tune arm; S-static's host tensors bitwise equal to the
anchor's at step 0; the **fold residual** |r_aff| ≤ X/4 = 0.005 nats on
every development anchor [taken, J-route] (a larger residual means the
"folds to 0" premise does not hold on this host and V1 is not opened).

**V1 efficient added capacity**, primary, reference host, in loss [taken,
J0: mean dev cross-entropy is the sole primary unit; see §12]:

- **present** if the lower bound of the median D1 ≥ X [taken, J1: X = 0.02
  nats, the smallest margin resolvable at n = 32 for a between-arm
  correlation of 0.9 or better, and the equivalence bound below; by outside
  prior about 0.5 to 0.7 accuracy points] **and** M-aff is plateau-certified
  on ≥ 75 percent of anchors [proposed];
- **re-tuning equivalent** if the upper bound of the median D1 < X and
  M-sq is certified on ≥ 75 percent of anchors. Said plainly: a true
  premium below about 0.016 nats under the measured correlation is
  published as equivalent to re-tuning, and John has accepted that a
  premium of that size from 272 parameters does not open rung M1;
- otherwise indeterminate, reported with the certification fractions.

Labels attached to any V1 reading, moving no threshold: **"loss gain
without accuracy movement"** if the one-sided lower bound of the median
ΔAcc is ≤ 0 (J0; the former "calibration, not capacity" condition, now a
label); secondary readings, each a declared one-sided 95 percent
anchor-bootstrap bound: **finishing**, "survives finishing" if the lower
bound of the median D3 ≥ X and "harvests undertraining" if the upper bound
of the median D3 < X, both equivalence calls at X read at D3's achieved
half-width; "beats every named re-tune" if the lower bound of D1′ > 0;
"fresh-data effect" if the lower bound of D4 > 0; **static** is
descriptive at the pilot and reads "design-in beats insertion" only at
confirmatory n, only if S-static is in the confirmatory set (§10), and
only if the lower bound of the median D5 > M_s [taken, J3: M_s = 0.02 nats;
S-static is admitted to the confirmatory set only if the pilot's recorded
SD(D5) ≤ 0.0275 nats, the value at which n* ≤ 32; a pilot SD near that
threshold is John's call, since a six-anchor SD carries about ±40 percent
sampling error]; the frontier fraction is descriptive.

**V2 add and remove**, co-primary, reference host:

- **overgrow-then-cut helps** if the lower bound of the median D2 ≥ X₂
  [taken, J2: X₂ = 0.01 nats, about three half-widths at the measured
  correlation of the closest attempt-4 analogue] and M-small-B is certified
  on ≥ 75 percent of anchors;
- **hurts** if the upper bound of the median D2 < 0 and the refit is
  certified on ≥ 75 percent of anchors;
- otherwise indeterminate.

D2s, the random-rank contrasts, the survival curves and the extended arms
are reported beside with bounds and no verdict role.

**Programme reading.** Rung M1 (generated builder) opens only on V1
present. V1 re-tuning-equivalent or harvests-undertraining sends the next
screen to the starved 0.5 host [taken, J-host], whose extended pair in
this study is the first evidence that screen would read. V2 sets the prior
for rung M2, which opens independently of M1 and starts near-empty in the
sense of D02 (J-D02, §12).

## 8. Cost ledger [measured kernels; process-seconds, one process per card, which is what the ceiling counts]

Kernels [measured, attempt 4]: 1.0 host 44.5 s charged (35.8 s training;
bottleneck hosts 41–42 s, so training is launch-bound and a narrower host
is not proportionally cheaper); one 24-candidate two-schedule grid 50 s;
per-step cost linear at 3.2 s per 800 steps; dev evaluation 0.04 s;
replay, timing and co-tenancy stages 883 s. Derived: a one-seed,
three-rate, checkpointed 3,200-step grid about 40 s, booked 60; a
five-rate grid with cached site booked 100 for M-aff and 80 for the
re-tune arms; arms that re-run the front booked at 3 × (unmeasured; 4 ×
is the defensible worst case; the pilot measures it first).

| Core item | Count | Booked |
|---|---|---:|
| Reference hosts (replay) | 6 × 45 s | 270 s |
| T-cool resumes | 6 × 25 s | 150 s |
| S-static hosts | 6 × 50 s | 300 s |
| Frontier hosts s = 0.5 and 1.5 | 6 × 35 s + 6 × 85 s | 720 s |
| Replay, timing, co-tenancy | as attempt 4 (about 600 s of it re-proves an unchanged host trainer; a V0-scope lever for John, not taken here) | 900 s |
| P1 instrument | 6 × 56 s (inherits the Gate 3 post-hoc wall; the P1-only carve-out costs less) | 340 s |
| Module-grid cases: M-sq, M-sq-fresh, M-sq \| T-cool, M-wide, M-small, S-static refit | 6 × 6 × 60 s | 2,160 s |
| M-aff, five-rate grid, fold and residual | 6 × 100 s | 600 s |
| M-small-B | 6 × 120 s | 720 s |
| M-wide→8 refit grids, two rankings | 6 × 2 × 60 s | 720 s |
| R-272s | 6 × 80 s | 480 s |
| R-aff (re-runs front) | 6 × 250 s | 1,500 s |
| Cuts, folds, leave-one-out, calibrations, evaluations | 6 × 30 s | 180 s |
| **Core pilot total** | | **about 9,050 s** |

Extended set, in priority order: (1) the enlarged 0.5-host pair, 6 ×
(T-cool 25 + M-sq 60 + M-aff 100 + M-sq | T-cool 60 + calibrations 30) =
1,650 s at the 1.0-host bookings (grid arms run only the tail, so the true
cost is lower; booked at the 1.0 figures); (2) R-all about 900 s (one
seed, five rates, 4,000 steps, at the scientific review's 3 to 5 × per-step
estimate for a full forward and backward, plus recalibration and
evaluations; booked high on purpose); (3) M-exp 360 s; (4) lottery 360 s;
(5) 20-epoch cooled F(1.0) 570 s; (6) F(0.25) 150 s; (7) F(0.75) 210 s;
about 4,200 s in all. Core plus extended is about 13,250 s at nominal.

**Overrun rule**, read strictly: extended arms run in that order only while
the measured charge projects to finish under the ceiling, and the pilot
stops admitting at the **first** extended arm that does not project to
fit; core arms are never dropped; if the core alone projects over the
ceiling the pilot stops before V1 is opened and John chooses. The stop is
hard and nothing resumes, so a stop costs the whole core again.

**Ceiling** [taken, J6: option B]: 4 GPU-hours, 14,400 process-seconds, as
a **new M0 constant**; `PILOT_SECONDS` (3 GPU-hours) is left untouched
because the Gate 3 ledgers are checked against it. The program-design
review's arithmetic: under 3 hours the core alone survives only a 1.23 ×
kernel overrun and core plus R-all a 1.11 ×, and a late overrun found in
step 5 burns the whole charge; under 4 hours the core survives 1.64 × and
at a 1.3 × stress the pilot still delivers the core, the 0.5-host pair and
part of the rest. The 4,593 s left in the Gate 3 cap belong to a closed
study and are not carried.

**Confirmatory** (§10): about 1,010 s per anchor for the confirmatory arm
set (host 45, T-cool 25, N0 and calibrations 30, five nonlinear module
grids 300, M-aff 100, two M-wide→8 refits 120, M-small-B 120, R-272s 80,
R-aff 250, summed from the lines above), plus a **fixed V0 term** of 300 to
1,500 s depending on whether the development hosts and P1 instrument are
re-run; n = 32 costs about 9.4 GPU-hours with the larger fixed term, and
survives about a 1.28 × overrun under the 12-hour ceiling. Adding S-static
costs 110 s more per anchor and leaves only a 1.2 × margin, so S-static
enters the confirmatory set only under a 14-hour ceiling (§10). The
projection is made from the pilot's measured one-per-card maxima in the
existing `costs.py` style, not from these bookings.

## 9. Stage 1, pilot (requires this document frozen with its SHA-256 pinned, every new-code review preserved, the certificate, and John's go)

Development anchors 3000–3005. Outputs to `docs/results/m0-pilot-<date>-<id>/`.

1. Host costs; replay of attempt 4's pinned state hashes; both-card
   determinism; one grid replay; co-tenancy replay.
2. P1 known answer; zero-module identities; slicing known answer; M-aff
   fold residual on every anchor.
3. Frontier, T-cool and S-static hosts; frontier slope.
4. Measured kernel of the first front-re-running arm **and one module grid
   on the new fitter** (4,400 s of core runs on that fitter and the P1
   instrument uses the old one, so it is not a proxy); projected charge from
   one-per-card maxima; extended-set admission by the overrun rule.
5. All core cases; plateau fractions per arm; the **per-anchor SDs of D1,
   D2, D3 and D5** recorded before any median is opened; S-static's
   confirmatory admission decided from SD(D5) (§7).
6. Confirmatory count: n = max(32, n*) where n* gives a half-width ≤ X/2
   for the median D1 under a normal approximation with the v4 inflation
   (the 32 floor and the inflation are the existing code); sized on D1
   only; D2, D3 and D5 are reported at their achieved half-widths. The
   projected charge includes the fixed V0 term. Stop before freeze if the
   count exceeds the ceiling, with John's choice.
7. Development medians, for the record only.

**What the pilot may fill**: the kernels, the extended-set admission, the
schedule axis (raise only), the confirmatory count, the replay rule, the
S-static admission by the pre-stated SD(D5) rule.
**What it may not change**: X, X₂, M_s, ρ, Hm, the fold tolerance, the 75
percent certification rule, the fit-data and selection sizes, the slicing
criterion, the arm definitions, the no-op rule or the verdict logic.

## 10. Stage 2, confirmatory (separate go; margins frozen; fresh seeds)

n reference-host anchors from the confirmatory block with the
**confirmatory arm set** [taken, J7]: N0 and calibrations, M-sq, M-aff
(with fold residual), R-272s, R-aff, T-cool, M-sq | T-cool, M-wide,
M-wide→8 (both rankings, sliced and refit), M-small, M-small-B. Frontier,
M-sq-fresh and every extended arm are pilot-only descriptives. S-static
joins only if the pilot's SD(D5) rule admits it **and** John raises the
confirmatory ceiling to 14 GPU-hours; under the 12-hour ceiling it stays
out. The ceiling, **12 GPU-hours as a new hard-stop constant with its own
supervisor instance** [taken, J7; nothing enforces a confirmatory ceiling
today], has no automatic extension. V0 then V1 and V2 read once. A V0
failure ends the attempt and burns its seeds. Any change to host, fit,
slicing, re-tune or calibration code after the freeze is a new version
with a new hash, a disposition and a fresh seed block.

## 11. Seeds and identity

| Lineage | Seeds [proposed] |
|---|---|
| Frontier hosts, pilot | s = 0.5: 6000–6005; s = 1.5: 6012–6017; s = 0.75 (extended): 6006–6011; 20-epoch cooled s = 1.0 (extended): 6018–6023; s = 0.25 (extended): 6024–6029 |
| S-static, pilot | the anchor's own seed (no new seeds) |
| Confirmatory reference anchors | 7500 onward, contiguous, ≤ 7599 |
| Restart after a failed confirmatory attempt | 7700 onward, ≤ 7799 |
| Rung M2 starting hosts (reserved now, J-D02; never used by M0) | 6100–6199 |

Appendix A is the seed registry; the blocks above collide with nothing in
it. The 0.5 hosts 6000–6005 become development anchors only if extended
item 1 runs; whether rung M2 may reuse them even for development follows
from that.

**Identity.** `PROTOCOL_VERSION = "m0-v1"`; `PROTOCOL_SHA` = SHA-256 of
`"m0-v1\n" + sha(this file) + "\n" + sha(docs/STEERING.md) + "\n" +
sha(docs/plans/2026-10-13-gate-ladder-amendment.md) + "\n"` (the v4.1
composition rule, one active identity from separately hashed documents);
`protocol_identity()` carries `protocol_version`, `protocol_sha256`,
`preregistration_sha256` (this file), `steering_sha256` and
`amendment_sha256` (the gate-ladder amendment, not the v4 amendments);
`source_identity()` pins these three documents instead of the v4 chain;
the identity tests are rewritten; the Gate 3 chain is not inherited and no
Gate 3 checkpoint is loaded.

## 12. Decisions taken (John, 2026-10-13: "ok go with the recommendations, recut the draft to v3")

Each row names the decision as taken and the report it came from
(`docs/reviews/m0-capacity/j-recommendations/`). Rows marked *author's
resolution* record a point the reports split on, resolved by the author
under John's own words; he can reverse any of them in one line before the
freeze.

| ID | Taken | Source | Note |
|---|---|---|---|
| J0 | Mean dev cross-entropy in nats is the sole primary unit for V1 and V2; ΔAcc is a reported secondary with the label "loss gain without accuracy movement"; it conditions no verdict | J0 critic; statistician and validator | *Author's resolution of a split.* The statistician's case for keeping the guard rested on accuracy being John's stated currency; John then said accuracy "was just a value we used in esper, its not 'the value we have to use'". The critic's case (no accuracy code exists, the guard was unsized, M1–M3 and V2 are loss-only) stands on its own. If John wants an accuracy claim in print, the guard returns to V1 and is sized under §9 step 6 |
| J1 | X = 0.02 nats, relabelled: the smallest margin resolvable at n = 32 for a between-arm correlation ≥ 0.9, and the equivalence bound; D3 keeps X and is read at its achieved half-width | statistician; validator | The validator confirmed every number and noted that "exactly n* = 32 at ρ = 0.9" is a five-degree-of-freedom coincidence; the pilot's SD(D1) is the unknown |
| J2 | X₂ = 0.01 nats, the "helps" bar; "hurts" at 0 | statistician; validator | |
| J3 | M_s = 0.02 nats; S-static enters the confirmatory set only if the pilot's SD(D5) ≤ 0.0275 and the ceiling is 14 h | statistician; J7 | |
| J-route | M-aff defines D1, with the fold residual measured and bounded at X/4, M-aff on the five-rate grid, and R-272s redefined to output channels 0–15 of `tail[0]` | solution-design reviewer | The fold is exact off the border only; v2's "folds exactly" is corrected in §1, §3, §5 |
| J-D2 | The matched-budget pair defines V2; matched means optimiser steps and example passes; the 4.3 × parameter-step ratio is disclosed; M-small-B certifies at 1,600 versus 6,400 | tech-selection critic | |
| J-host | Reference host primary; the 0.5-host extended pair enlarged with T-cool and M-sq \| T-cool and run first in the extended order | leverage analyst | The two added arms were not in v2; flagged as the reviewer's addition |
| J4 | Core widths {0.5, 1.0, 1.5}; the 20-epoch cooled point ahead of F(0.75); its schedule named in §3; the slope from the three core widths only | plan-review reality | F(0.25) sits between them at the J-D02 review's request |
| J6 | Option B: 4 GPU-hours (14,400 s) as a new M0 constant; `PILOT_SECONDS` untouched; the overrun rule reads stop-at-first-miss | program-design architect | The only outright rejection of a v2 default |
| J7 | The §10 arm set; n = max(32, n*); 12 GPU-hours as a new hard-stop constant with its own supervisor instance; S-static out unless 14 h; a fixed V0 term in the projection | program-design architect; validator | |
| J-retire | Retired as §2; mechanism is a code carve-out (P1-only workers, checks, cost projection, tests), not prose; reversal triggers recorded | shaping architect; plan-review reality | The two reports disagreed on mechanism; source settled it for the reality reading |
| J-D02 | Principle fixed now: rung M2 starts near-empty in the sense of D02; the 0.5 host is not that start and becomes it only by John's explicit, recorded relaxation of D02; near-empty in this family means s ≤ 0.25; the number is fixed at the M2 draft. M0 records: a reserved M2 seed block, the burn rule for 6000–6005, an explicit width-tuple constructor, published frontier hashes, and that slicing is module-only | shaping architect | F(0.25) added to the extended set; flagged as the reviewer's addition |
| J9 | The gate reading in STEERING §6 becomes a separate, dated, SHA-pinned, prospective amendment to the frozen correction protocol, bound into the M0 identity; the protocol's bytes stay untouched | J9 critic; Codex (independent model, same substance) | **The amendment is drafted; its seven gate rows await John's acceptance at the freeze.** He accepted the form, not yet the text |
| J10 | Merge of the steering package to `main` | — | Pending John. Nothing in this package is durable for the repository until then |

Seed blocks and the 0.5-host anchor count are the author's and live in the
implementation plan.

## 13. Acceptance

v3 checked once more in a targeted pass against the synthesis table and
the recut checklist, the report preserved and dispositioned; John accepts
the gate-ladder amendment text; the file's SHA-256 pinned in the package;
then the implementation plan with exact-source reviews for every item in
§2; then the pilot go.

## Appendix A. Seed registry (never reuse)

| Block | Owner |
|---|---|
| 1000–1031 | former training lineages (correction protocol, v4 A7) |
| 2000–2007, 3000–3007 | `sultai.formation` dev and test lineages |
| 3000–3099 | Gate 3 and M0 development anchors (3000–3005 used) |
| 4000–4007, 5000–5007, 8000s, 9000s, 14000s, 15000s, 24000s, 25000s | correction protocol cohorts and streams |
| 4000–4363, 5000–5799 | Gate 3 confirmatory blocks and restarts, retired unused |
| 7001–7004 | `sultai.lifecycle` streams |
| 6000–6029 | M0 frontier hosts (6000–6005 s = 0.5; 6006–6011 s = 0.75; 6012–6017 s = 1.5; 6018–6023 20-epoch; 6024–6029 s = 0.25) |
| 6100–6199 | rung M2 starting hosts, reserved (J-D02) |
| 7500–7599, 7700–7799 | M0 confirmatory anchors and restart |
