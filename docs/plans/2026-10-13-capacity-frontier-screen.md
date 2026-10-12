# Capacity frontier screen on the CIFAR-10 host — DRAFT PREREGISTRATION v2 (programme rung M0)

Status: **draft v2 for John's decisions; authorises no execution.** v1 was
committed at `5aeee56` and reviewed by three independent readers
(alignment, scientific, reality); their reports and the disposition of every
finding are preserved under
[`docs/reviews/m0-capacity/`](../reviews/m0-capacity/v1-dispositions-2026-10-13.md).
Written 2026-10-13 by `sultai-claude-nyx` under the
[steering statement](../STEERING.md). Values marked [John] are his to fix
before the pilot; [proposed] are the author's defaults; [measured] come from
Gate 3 attempt-4 receipts.

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
exactly into the next convolution and so adds no retained capacity, (iii)
re-tuning 272 or 170 of the host's existing parameters at the site, (iv) a
decayed-rate finish of the host's own training, and (v) the same module
present from step zero? And where does the result sit against the width
frontier of the host family at the same compute?

**Q2, add and remove (Sultai versus the plug-in route).** A module trained
in isolation and blended at α = 1 is Esper-lite's plug-in route in
miniature. Sultai's route overgrows and physically cuts. Does a 64-hidden
module (2,128 parameters) fitted and then **sliced** to 8 hidden units (280
parameters) keep more of its gain than a 280-parameter module built small,
at matched optimiser budget? The built-small arm is also the cut-down
architecture retrained from scratch (the Liu et al. control). This is the
truncation precedent for Sultai's thesis, not the developmental-removal
question, which belongs to rung M2.

A positive V1 says there is site capacity that ordinary SGD finds and no
cheaper route reproduces; that is what a generated builder would later have
to find more cheaply (rung M1). A negative V1 says the reference host's site
is not where capacity is missing, and the next screen moves to the starved
host that this study's frontier sweep already trains. Site S1 is a
privileged, known choice inherited from Gate 3; site discovery is rung M2's
problem. Either verdict is a published finding.

## 2. What carries over, what is new code, what is retired

**Carried over unchanged** (verified against source by the reality review):
host architecture and recipe at s = 1.0 and 10 epochs; the data seal (test
batch never read) and the 40,000 / 2,000 / 3,000 / 5,000 sub-partition; the
padded fixed-shape evaluator; the frozen-host hash check; `losses()` on any
two- or four-tensor adapter of any hidden width; the plateau rule's form (its step indices change);
the per-example paired bootstrap; the harm screen (Hm = 0.01 nats); custody
receipts, ledgers and GPU-time accounting; the P1 known answer (A1, A2, A3,
B3 at α = 0.1; B3 ≥ ρ · A3 gain, ρ = 0.8) as instrument validation only.

**New code, each with its own exact-source review before the pilot**: a
tail-width constructor parameter that leaves the s = 1.0 layer order and RNG
consumption bitwise identical; saving optimiser state and a resume path with
a decayed-rate schedule; the oracle grid's rate set, checkpointed single-run
schedule, hidden-width parameter, warm start and affine option; the
plateau comparator at 800 versus 3,200 steps with the last two 40-step
windows; an epoch count other than 10 for the extended 20-epoch point; the
re-tuning fit paths (declared parameter subset trained, complement hashed;
arms that touch `front` re-run the site from images every step); BatchNorm
recalibration and logit calibration; a static-module host; physical
slicing with the mean fold, unit ranking and the slicing known-answer test;
new worker actions, seed admission, identity pins and tests; the
anchor-bootstrap, sign-test and verdict aggregation. A stage that depends on
new code reads nothing until that code's review is preserved.

**Retired from this study** [John, J-retire]: P2, P3, P3′, P5, P6, the former
arms A7/A8, retrieval B4 and the cheap bank B1/B2/B2′/B6. They stay
published Gate 3 evidence.

## 3. Hosts, frontier, site and modules

**Reference host** (v4 §3): 284,634 parameters, tail widths (64, 64, 128,
128). Development anchors reuse seeds 3000–3005; the freshly trained hosts
must reproduce attempt 4's published `final_state_sha256` strings (the
fresh receipt is compared to the string; no Gate 3 checkpoint is loaded).

**Frontier family** [proposed]: the same architecture with the four tail
widths scaled by s. Parameter counts recomputed from the constructor.

| s | Tail widths | Parameters | In the core set |
|---|---|---:|---|
| 0.5 | 32, 32, 64, 64 | 85,274 | yes; corresponds to the Claude design's starved host |
| 0.75 | 48, 48, 96, 96 | 168,826 | extended only |
| 1.0 | 64, 64, 128, 128 | 284,634 | yes; the six reference anchors are these points |
| 1.5 | 96, 96, 192, 192 | 613,018 | yes |

Every frontier host trains under the unchanged recipe for 10 epochs
(400,000 example passes). Disclosure from v4 §3: the recipe was tuned by
Simic for the 1.0 host against the same 5,000 dev examples, so this is a
**same-recipe frontier, 10 epochs**, and every reading that cites it says
so. The extended set adds a 20-epoch cooled 1.0 point.

**Site and modules.** Site S1 reads z (16 × 16 × 16) after the producer.

| Name | Form | Retained parameters | Role |
|---|---|---:|---|
| M-sq | u = z + W tanh(z) + b | 272 | the capacity reading; plug-in route |
| M-aff | u = z + W z + b | 272, **folds to 0** | primary matched-count control: a per-position affine ahead of a 3×3 conv folds exactly into that conv in the interior (approximately at the zero-padded border), so it adds no retained capacity (D21) |
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
answer (§7).

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
instrument (unchanged from Gate 3) and M-sq-fresh (§5), whose job is to
measure the fresh-data effect directly.

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
module arms (attempt 4 selected the top rate 0.3 in 28 of 30 grids
[measured]) and {1, 0.3, 0.1, 0.03, 0.01} for re-tune arms, one 3,200-step
run per rate with selection checkpoints at 200 and 800 steps (bitwise the
same as separate runs under a seed-derived stream). Best on selection. The
selected rate is reported with an **edge-of-grid flag**. Plateau is
certified by the v4 Table B rule applied to the 800- versus 3,200-step
checkpoints and the last two 40-step training windows. Attempt 4 certified
2 of 30 oracle grids at 800 steps (anchor 3003, both families, width 4)
[measured]; the pilot may extend the schedule axis, raise only. R-all,
run to 800 steps only, is never plateau-certified and enters no
certification fraction.

**Warm starts.** M-wide→h refit uses the sliced tensors as the start and the
full module grid. M-small-B is M-small at twice the schedule
({400, 1,600, 6,400}), the overgrow route's total optimiser steps.

**Booking** (v4 §6 units, per arm and anchor): example passes, optimiser
parameter-steps, wall seconds, device, physically retained parameters. R-aff
and R-all re-run the front per step and are booked in both parameter-steps
and wall; nothing is netted.

## 5. Arms

All arms on each development anchor of the reference host. Gains G are in
nats of mean dev cross-entropy relative to N0; dev accuracy is reported
beside every loss.

**Core set** (bold arms enter a verdict):

| Arm | Definition | Retained added | Separates |
|---|---|---:|---|
| N0 | published no-op (§4) and its bitwise twin | 0 | zero and noise floor |
| R-temp, R-cal | logit temperature; temperature + class bias, fit examples | 1, 11 | calibration |
| **M-sq** | square tanh module, module grid | 272 | the reading |
| **M-aff** | affine module, module grid | 272 → 0 by folding | nonlinearity versus re-tuning |
| R-272s | re-tune the 16×16 centre taps and 16 output biases of the first downstream conv (272 entries), re-tune grid; cached site | 0 | structured equal-count re-tune |
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

**Extended set**, run in this priority order only while measured kernels
leave room under the ceiling (§8), each descriptive: (1) R-all, re-tune
every host parameter, one seed, rates as re-tune arms, checkpoints every 40
steps to 800, BN recalibrated after fitting, re-runs the front; (2) M-exp;
(3) lottery cut, the surviving 8 units' initial weights retrained with the
module grid; (4) F(0.75); (5) a 20-epoch cooled F(1.0); (6) the Q1 pair
M-sq and M-aff on the six 0.5 frontier hosts, which then become development
anchors never reused for a starved-host confirmatory set.

## 6. Quantities

Per anchor, with per-example paired-bootstrap SE over the 5,000 dev
examples (v4 §7), for loss and for accuracy:

- **D1 = G(M-sq) − G(M-aff)**, the nonlinear module against the foldable
  affine module, equal count, same data, same grid.
- **ΔAcc = acc(M-sq) − acc(N0)**, the accuracy co-read.
- D1′ = G(M-sq) − max(G(R-272s), G(R-aff)), against every named re-tune.
- **D3 = L(T-cool host, after its own N0 rule) − L(M-sq \| T-cool)**, the
  module's gain over the finished host, not over the anchor's N0; and the
  finishing gain G(T-cool) itself, relative to the anchor's N0.
- D4 = G(M-sq-fresh) − G(M-sq), the fresh-data effect.
- D5 = L(host + M-sq) − L(S-static + refit), negative when insertion beats
  design-in; also without the refit. Paired by data stream, not by host, so
  its noise is the host's seed-to-seed spread.
- **D2 = G(M-wide→8, contribution rank, refit) − G(M-small-B)** at matched
  budget; D2s = G(M-wide→8, as sliced with fold) − G(M-small), the strict
  reading; the random-rank versions of both; survival curves
  G(M-wide→h)/G(M-wide) for every h, sliced and refit; the leave-one-out
  ceiling.
- Calibration: G(R-temp), G(R-cal); re-tune fraction G(M-sq)/G(R-all) when
  R-all runs.
- **Frontier**: slope in nats per doubling of parameters from the three
  core widths with a seed-bootstrap interval; the module's gain G(M-sq)
  expressed as a fraction of one doubling with its interval. No
  extrapolation beyond 613,018; no ratio to 272; label "same-recipe
  frontier, 10 epochs".

Across anchors: median with one-sided 95 percent percentile
anchor-bootstrap bounds (v4 §7), and an exact sign-test bound beside it at
confirmatory n. In the pilot only the per-anchor SDs are read before any
median (§9).

## 7. Verdicts (margins [John], fixed before the pilot)

**V0 instrument.** Replay bitwise on both cards against the published
attempt-4 hashes; A1 restores within 1e-4; B3 ≥ ρ · A3 gain on every
development anchor; zero module bitwise identity on every family; every
sliced module's printed count equals 33h + 16; the slicing known answer (a
64-unit module with 56 columns of V exactly zero must, after the criterion
cut to 8 with fold, reproduce its output within 1e-6, a random cut must
not, and a negative control must fail); the complement hash unchanged
under every re-tune arm; S-static's host tensors bitwise equal to the
anchor's at step 0.

**V1 efficient added capacity**, primary, reference host:

- **present** if the lower bound of the median D1 ≥ X [proposed X = 0.02
  nats; inherited from the Gate 3 margin and, against a per-example paired
  SE near 0.004 nats, a detectability floor rather than a capacity
  threshold, because 272 parameters of width buy of order 1e-4 nats on the
  frontier], **and** the lower bound of the median ΔAcc > 0, **and** M-aff is
  plateau-certified on ≥ 75 percent of anchors [proposed];
- **calibration, not capacity** if the D1 condition holds but the ΔAcc
  bound is ≤ 0;
- **re-tuning equivalent** if the upper bound of the median D1 < X and
  M-sq is certified on ≥ 75 percent of anchors;
- otherwise indeterminate, reported with the certification fractions.

Secondary readings, each a declared one-sided 95 percent anchor-bootstrap
bound, moving no threshold: **finishing**, labelled "survives finishing" if
the lower bound of the median D3 ≥ X and "harvests undertraining" if the
upper bound of the median D3 < X; "beats every named re-tune" if the lower
bound of D1′ > 0; "fresh-data effect" if the lower bound of D4 > 0;
**static** is descriptive at the pilot and reads "design-in beats
insertion" only at confirmatory n, and only if S-static is in the
confirmatory set (J7), if the lower bound of the median D5 > M_s [proposed
M_s = 0.02; underpowered against seed-to-seed spread, said so]; the
frontier fraction is descriptive.

**V2 add and remove**, co-primary, reference host:

- **overgrow-then-cut helps** if the lower bound of the median D2 ≥ X₂
  [proposed X₂ = 0.01 nats] and M-small-B is certified on ≥ 75 percent of
  anchors;
- **hurts** if the upper bound of the median D2 < 0 and the refit is
  certified on ≥ 75 percent of anchors;
- otherwise indeterminate.

D2s, the random-rank contrasts, the survival curves and the extended arms
are reported beside with bounds and no verdict role.

**Programme reading.** Rung M1 (generated builder) opens only on V1
present. V1 re-tuning-equivalent, calibration or harvests-undertraining
sends the next screen to the starved 0.5 host [John, J-host]. V2 sets the
prior for rung M2, which opens independently of M1.

## 8. Cost ledger [measured kernels; process-seconds, which is what the ceiling counts]

Kernels [measured, attempt 4]: 1.0 host 44.5 s charged; one 24-candidate
two-schedule grid 50 s; per-step cost linear at 3.2 s per 800 steps; dev
evaluation 0.04 s; replay, timing and co-tenancy stages 883 s. Derived: a
one-seed, three-rate, checkpointed 3,200-step grid about 40 s, booked 60;
the five-rate re-tune grid with cached site booked 80; arms that re-run the
front booked at 3 × (unmeasured; the pilot measures it first).

| Core item | Count | Booked |
|---|---|---:|
| Reference hosts (replay) | 6 × 45 s | 270 s |
| T-cool resumes | 6 × 25 s | 150 s |
| S-static hosts | 6 × 50 s | 300 s |
| Frontier hosts s = 0.5 and 1.5 | 6 × 25 s + 6 × 95 s | 720 s |
| Replay, timing, co-tenancy | as attempt 4 | 900 s |
| P1 instrument | 6 × 56 s | 340 s |
| Module-grid cases: M-sq, M-aff, M-sq-fresh, M-sq \| T-cool, M-wide, M-small, S-static refit | 6 × 7 × 60 s | 2,520 s |
| M-small-B | 6 × 120 s | 720 s |
| M-wide→8 refit grids, two rankings | 6 × 2 × 60 s | 720 s |
| R-272s | 6 × 80 s | 480 s |
| R-aff (re-runs front) | 6 × 250 s | 1,500 s |
| Cuts, folds, leave-one-out, calibrations, evaluations | 6 × 30 s | 180 s |
| **Core pilot total** | | **about 8,800 s** |

Extended set, in priority order: R-all about 900 s (one seed, five rates,
4,000 steps, at the scientific review's 3 to 5 × per-step estimate for a
full forward and backward, plus recalibration and evaluations; booked high
on purpose); M-exp 360 s; lottery
360 s; F(0.75) 210 s; 20-epoch cooled F(1.0) 570 s; the 0.5-host Q1 pair
720 s; about 3,100 s in all. **Overrun rule**: extended arms run in that
order only while the measured charge projects to finish under the ceiling;
core arms are never dropped; if the core alone projects over the ceiling
the pilot stops before V1 is opened and John chooses.

**Ceiling** [John, J6]: option A, a fresh 3 GPU-hour pilot ceiling
(10,800 s, the existing `PILOT_SECONDS`), core set with whatever extended
arms fit; option B, 4 GPU-hours (14,400 s) admitting the whole extended set
[code change]. The 4,593 s left in the Gate 3
cap belong to a closed study and are not carried by default.

**Confirmatory** (§10): about 965 s per anchor for the confirmatory arm set;
n = 32 costs about 8.6 GPU-hours; n = 43 about 11.5. Proposed ceiling 12
GPU-hours [John, J7; code change from the 16-hour constant], no automatic
extension. Adding S-static to the confirmatory set costs about 110 s more
per anchor.

## 9. Stage 1, pilot (requires this document frozen with its SHA-256 pinned, every new-code review preserved, the certificate, and John's go)

Development anchors 3000–3005. Outputs to `docs/results/m0-pilot-<date>-<id>/`.

1. Host costs; replay of attempt 4's published state hashes; both-card
   determinism; one grid replay; co-tenancy replay.
2. P1 known answer; zero-module identities; slicing known answer.
3. Frontier, T-cool and S-static hosts; frontier slope.
4. Measured kernel of the first front-re-running arm; projected charge;
   extended-set admission by the overrun rule.
5. All core cases; plateau fractions per arm; the **per-anchor SDs of D1,
   D2 and D3** recorded before any median is opened.
6. Confirmatory count: n = max(32, n*) where n* gives a half-width ≤ X/2
   for the median D1 under a normal approximation with the v4 inflation
   (the 32 floor and the inflation are the existing code); sized on D1
   only; D2 and D3 are reported at their achieved half-widths. Stop before
   freeze if the count exceeds the ceiling, with John's choice.
7. Development medians, for the record only.

**What the pilot may fill**: the kernels, the extended-set admission, the
schedule axis (raise only), the confirmatory count, the replay rule.
**What it may not change**: X, X₂, M_s, ρ, Hm, the 75 percent certification
rule, the fit-data and selection sizes, the slicing criterion, the arm
definitions, the no-op rule or the verdict logic.

## 10. Stage 2, confirmatory (separate go; margins frozen; fresh seeds)

n reference-host anchors from the confirmatory block with the
**confirmatory arm set**: N0 and calibrations, M-sq, M-aff, R-272s, R-aff,
T-cool, M-sq | T-cool, M-wide, M-wide→8 (both rankings, sliced and refit),
M-small, M-small-B. Frontier, S-static, M-sq-fresh and every extended arm
are pilot-only descriptives unless John adds them [John, J7]. V0 then V1
and V2 read once. A V0 failure ends the attempt and burns its seeds. Any
change to host, fit, slicing, re-tune or calibration code after the freeze
is a new version with a new hash, a disposition and a fresh seed block.

## 11. Seeds and identity

| Lineage | Seeds [proposed] |
|---|---|
| Frontier hosts, pilot | s = 0.5: 6000–6005; s = 1.5: 6012–6017; s = 0.75 (extended): 6006–6011; 20-epoch cooled s = 1.0 (extended): 6018–6023 |
| S-static, pilot | the anchor's own seed (no new seeds) |
| Confirmatory reference anchors | 7500 onward, contiguous, ≤ 7599 |
| Restart after a failed confirmatory attempt | 7700 onward, ≤ 7799 |

Appendix A is the seed registry; the blocks above collide with nothing in
it.

**Identity.** `PROTOCOL_VERSION = "m0-v1"`; `PROTOCOL_SHA` = SHA-256 of
`"m0-v1\n" + sha(this file) + "\n" + sha(docs/STEERING.md) + "\n"`;
`protocol_identity()` carries `protocol_version`, `protocol_sha256`,
`preregistration_sha256` (this file) and `steering_sha256`, dropping
`amendment_sha256`; `source_identity()` pins this file and the steering
statement instead of the v4 chain; the identity tests are rewritten; the
Gate 3 chain is not inherited and no Gate 3 checkpoint is loaded.

## 12. Decisions John must make before the pilot

| ID | Decision | Proposed default |
|---|---|---|
| J0 | Primary unit: mean dev cross-entropy in nats with the accuracy co-read as a guard (the author's formalisation of "accuracy at the most efficient parameter cost"; loss is what every arm optimises and what the paired bootstrap measures) | as stated |
| J1 | X, the capacity margin for D1 and D3 | 0.02 nats, inherited, labelled a detectability floor |
| J2 | X₂, the add-and-remove margin for D2 | 0.01 nats |
| J3 | M_s, the static margin (confirmatory only) | 0.02 nats |
| J-route | Which control defines D1 | M-aff; R-272s and R-aff as D1′ |
| J-D2 | Which pair defines V2 | matched budget (refit grid versus M-small-B); as-sliced is the strict secondary |
| J-host | Primary host for M0 | reference host; the 0.5 host as the extended Q1 pair and as the fallback screen |
| J4 | Frontier widths and the 20-epoch cooled point | {0.5, 1.0, 1.5} core; 0.75 and 20-epoch extended |
| J6 | Pilot ceiling: A, 3 GPU-hours core-first; B, 4 GPU-hours | A |
| J7 | Confirmatory arm set, floor and ceiling | as §10, n ≥ 32, 12 GPU-hours |
| J-retire | Retiring the published Gate 3 families and arms from this study | as §2 |
| J-D02 | Starting point for rung M2 (near-empty per D02, or the 0.5 host) | decided at the M2 draft, recorded now as open |
| J9 | Gate reinterpretation, steering statement §6 | as proposed |
| J10 | Merge of the steering package to `main` | — |

Seed blocks and the 0.5-host anchor count are the author's and live in the
implementation plan.

## 13. Acceptance

v2 reviewed once more in a targeted pass against the disposition table;
John's decisions recorded; the file's SHA-256 pinned in the package; then
the implementation plan with exact-source reviews; then the pilot go.

## Appendix A. Seed registry (never reuse)

| Block | Owner |
|---|---|
| 1000–1031 | former training lineages (correction protocol, v4 A7) |
| 2000–2007, 3000–3007 | `sultai.formation` dev and test lineages |
| 3000–3099 | Gate 3 and M0 development anchors (3000–3005 used) |
| 4000–4007, 5000–5007, 8000s, 9000s, 14000s, 15000s, 24000s, 25000s | correction protocol cohorts and streams |
| 4000–4363, 5000–5799 | Gate 3 confirmatory blocks and restarts, retired unused |
| 7001–7004 | `sultai.lifecycle` streams |
| 6000–6023 | M0 frontier hosts |
| 7500–7599, 7700–7799 | M0 confirmatory anchors and restart |
