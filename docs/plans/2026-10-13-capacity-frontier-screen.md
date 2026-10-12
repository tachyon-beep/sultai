# Capacity frontier screen on the CIFAR-10 host — DRAFT PREREGISTRATION v1 (programme rung M0)

Status: **draft for independent review and John's decisions.** Authorises no
execution. Written 2026-10-13 by `sultai-claude-nyx` under the
[steering statement](../STEERING.md), after John redirected the programme
from repair to efficient capacity. Every value marked [John] is his to fix
before the pilot opens; every value marked [proposed] is the author's
default; every value marked [measured] comes from Gate 3 attempt-4 receipts.

## 1. The question, in the programme's units

On a real CIFAR-10 host at a 16-channel site, a 272-parameter module fitted
by ordinary loss gradient lowered development loss on every host it was
tried on in Gate 3 attempt 4, by 0.04 to 0.24 nats, whether or not it could
see any discarded information
([hold](../certificates/2026-10-13-gate3-linked-pilot-3-hold.md)). The host
has 284,634 parameters, so the module is 0.1 percent of it. That number is
uninterpretable on its own, because the host trains for 10 epochs and is
undertrained, and because the module was fitted on 2,000 examples the host
never saw. Two questions turn it into a measurement of John's objective.

**Q1, efficient added capacity.** Does the new 272-parameter module buy more
loss than re-tuning an equal count of the host's *existing* parameters on the
same examples with the same optimiser? And how does its gain compare with
re-tuning the whole host on those examples, with more host training on the
training partition, with the same module present from step zero, and with
the from-scratch width frontier of the host family?

**Q2, add and remove.** Does a deliberately overgrown module (64 hidden
units, 2,128 parameters), fitted and then **physically cut** to 8 hidden
units (280 parameters), keep more of its gain than a module built at 8
hidden units from the start? This is the smallest measurable instance of
Sultai's add-and-remove thesis and the ExpandNets precedent
([concept](../CONCEPT.md) and the
[diffusion report §2.2](../research/claude-handoff/simic-diffusion-grown-modules.md)).
The cut-down module retrained from scratch is the Liu et al. counter-control
and is the same arm as "built small".

A positive Q1 says there is site capacity that ordinary SGD finds and that is
not available by re-tuning; that is what a generated builder would later have
to find more cheaply (rung M1). A negative Q1 says this host's site is not
where capacity is missing, and the next study moves to the starved host that
this screen's frontier sweep already trains. Either is a published finding.

## 2. What carries over from Gate 3 unchanged

Host architecture, training recipe, data seal (the CIFAR-10 test batch is
never read), the fit/dev permutation and the 40,000 / 2,000 / 3,000 / 5,000
sub-partition, the fixed-shape padded evaluator (v4.1 §3), the determinism
flag set and replay checks, the oracle fitter and its selection-only
candidate choice, plateau certification, the paired-bootstrap SE, the harm
screen at Hm = 0.01 nats, custody receipts, GPU-time accounting, source and
protocol identity, the quality gates, and the P1 known answer as instrument
validation (B3 ≥ ρ · A3 gain with ρ = 0.8, on every development anchor).
Section numbers below cite `plans/2026-10-10-gate3-headroom-screen.md` (v4)
for definitions not restated.

The P2, P3, P3′, P5 and P6 families, the former arms A7/A8, the retrieval arm
B4 and the cheap bank B1/B2/B2′/B6 are **retired from this study**. They
remain published Gate 3 evidence.

## 3. Hosts, frontier, site and modules

**Reference host** (v4 §3): 284,634 parameters, tail widths (64, 64, 128,
128). Development anchors reuse seeds 3000–3005 so the freshly trained hosts
must replay attempt 4's `final_state_sha256` bitwise; a mismatch is a hold.

**Frontier family** [proposed]: the same architecture with the four tail
widths scaled by s and rounded, classifier input following the last width.

| s | Tail widths | Parameters |
|---|---|---:|
| 0.5 | 32, 32, 64, 64 | 85,274 |
| 0.75 | 48, 48, 96, 96 | 168,826 |
| 1.0 | 64, 64, 128, 128 | 284,634 (reference) |
| 1.5 | 96, 96, 192, 192 | 613,018 |

Each width trains under the unchanged recipe at the same 10 epochs (matched
example passes, 400,000). Disclosure carried from v4 §3: the recipe was tuned
by Simic for the 1.0 host against the same 5,000 dev examples, so this is a
**same-recipe frontier**, weaker at its ends; the verdict wording in §7 says
so. A second epoch budget for the frontier (20 epochs) is optional [John].
The 0.5 host is the starved host of the Claude design (P10) and is also the
secondary host for the Q1 contrast (§5).

**Site and primary module** (v4 §3): site S1 reads z (16 × 16 × 16) after
the producer; module u = z + (W tanh(z) + b), 272 parameters, α = 1, W = 0,
b = 0 bitwise identity. Call it **M-sq** (square, 272).

**Hidden-unit family** (already implemented as the v4 B5 wide adapter):
u = z + V tanh(U z + c) + d with h hidden units; parameters 32h + h + 16.

| h | Parameters |
|---|---:|
| 64 | 2,128 (overgrown, **M-wide**) |
| 16 | 544 |
| 8 | 280 (**M-small**, built small; the cut target) |
| 4 | 148 |
| 2 | 82 |

**Physical truncation** [proposed]: rank hidden units by contribution
‖V[:, i]‖₂ · RMS over conditioning positions of tanh(U z + c)_i; remove the
lowest-ranked; slice U, c, V to the surviving rows and columns so the stored
tensors are smaller and the printed parameter count falls. Evaluate the cut
module (i) as sliced and (ii) after a short refit (§4). A random cut at the
same h is the control for the ranking criterion.

## 4. Fitting, data regime and compute booking

All module fits and all re-tuning controls use the **same 2,000 conditioning
examples** for gradient steps, the **3,000 selection examples** for candidate
choice, and the **5,000 development examples** as the endpoint, through the
frozen downstream with BatchNorm in eval mode (v4 §3). Nothing in this study
trains on dev or reads the test batch.

**Oracle grid** [proposed, changed from v4 to address plateau]: minibatch
SGD on the module parameters, init N(0, 1e-3), 2 seeds × rates {0.3, 0.1,
0.03} × schedule {200, 800, 3,200} steps, 18 candidates, best on selection.
Plateau certified by the v4 Table B rule applied to the 800 versus 3,200
comparator and the last two 40-step windows. Attempt 4 certified 1 of 24
fits at 800 steps [measured], which is why the axis grows. Uncertified fits
are retained and flagged; the §7 verdicts carry the uncertified fraction.

**Refit after truncation** [proposed]: 800 minibatch steps at the rate the
oracle chose for M-wide, one seed; the cut module's own selection loss is
reported before and after.

**Booking** (v4 §6 units, per arm and anchor): example passes, optimiser
parameter-steps, wall seconds, device, and physically retained parameters.
Static and frontier hosts book their full training. Nothing is netted.

## 5. Arms

All arms on each development anchor of the reference host; the Q1 pair also
on the 0.5 host [proposed; count by John]. Gains are in nats of mean dev
cross-entropy versus the host's own no-op; dev accuracy is reported beside.

| Arm | Definition | Retained parameters added | What it tells apart |
|---|---|---:|---|
| A0 | Host, no module, and the bitwise no-op twin (v4 §9 item 3) | 0 | Zero and noise floor |
| **M-sq** | 272-parameter square module, oracle grid | 272 | The capacity reading |
| **R-272** | Re-tune a fixed 272-entry subset of the host's own parameters [proposed: entries of the first downstream conv (16→64, 9,216 weights) chosen by a fixed-seed draw per anchor], same examples, same grid; every other host parameter frozen and hashed | 0 | Re-tuning of an equal count at the same site |
| R-aff | Re-tune the affine parameters around the site: producer BN (32), first downstream BN (128), classifier bias (10) = 170 entries, same grid | 0 | Descriptive second re-tune control |
| **R-all** | Re-tune every host parameter on the conditioning examples, BN statistics frozen, same grid | 0 | Ceiling of re-tuning on the same data |
| **T-more** | Continue the host's own recipe on the 40,000-example training partition for 200,000 more example passes (5 epochs), then evaluate with no module | 0 | "The host is just undertrained" |
| **S-static** | Host trained from scratch with M-sq present at α = 1 from step zero, module parameters in the recipe's optimiser, same seed and data stream | 272 | Capacity designed in versus inserted |
| **F(s)** | Frontier hosts at s ∈ {0.5, 0.75, 1.0, 1.5}, 6 seeds each, no module | by width | Scale: nats per parameter from width |
| **M-wide** | 64-hidden module, oracle grid | 2,128 | Overgrown fit |
| **M-wide→h** | M-wide physically cut to h ∈ {16, 8, 4, 2}, by contribution rank and by random rank, as sliced and after refit | 32h + h + 16 | Add-and-remove |
| **M-small** | 8-hidden module built from scratch, oracle grid | 280 | Built small; also the cut-down architecture retrained |
| P1 known answer | v4 Table A arms A1, A2, A3 and B3 on P1 at α = 0.1, 6 anchors | n/a | Instrument validation only, ρ = 0.8 |

Arms in bold enter a verdict. R-272 and R-all need a **new reviewed fit
path**: the existing evaluator rejects any host whose parameter hash changes,
so re-tuning requires a path that trains a declared parameter subset and
hashes the complement. S-static needs the module inside the host's training
graph. T-more needs a resume-from-checkpoint training path. Everything else
is a composition of existing code.

## 6. Quantities

Per anchor, with per-example paired-bootstrap SE over the 5,000 dev examples
(v4 §7):

- G(arm) = L(A0) − L(arm) for every module and re-tune arm.
- **D1 = G(M-sq) − G(R-272)**: new parameters versus re-tuned parameters,
  equal count, same data, same optimiser.
- **D2 = G(M-wide→8, after refit) − G(M-small)**: cut-down versus built
  small at 280 retained parameters.
- Survival curve: G(M-wide→h, as sliced) / G(M-wide) for each h, and the
  same after refit; contribution rank versus random rank.
- Static contrast: L(S-static) − L(host + M-sq), paired on the seed.
- Re-tune fraction: G(M-sq) / G(R-all).
- Undertraining: G(T-more), and L(F(1.0) at 20 epochs) if the second epoch
  budget is taken.
- **Frontier-equivalent parameters N_eq**: fit L against log₂ N over the
  frontier means; N_eq is the parameter count at which the frontier reaches
  L(host + M-sq); report N_eq − 284,634 and the ratio to 272. This is the
  scale statement "a 272-parameter module bought as much loss as adding
  N_eq − N parameters of width", and it is descriptive.

Across anchors: median with one-sided 95 percent percentile
anchor-bootstrap bounds, as v4 §7.

## 7. Verdicts (margins [John], fixed before the pilot)

**V0 instrument.** Replay bitwise on both cards; A1 restores within 1e-4;
B3 ≥ ρ · A3 gain on every development anchor; zero module bitwise identity;
every cut module's printed parameter count equals the formula; the complement
hash under R-272, R-aff and R-all unchanged; at least 75 percent [proposed]
of verdict-arm fits plateau-certified, otherwise V1 and V2 are reported as
"indeterminate: oracle not at plateau" and are not a programme stop.

**V1 efficient added capacity**, primary, on the reference host:

- **present** if the lower bound of the median D1 ≥ X [proposed X = 0.02
  nats, the Gate 3 margin];
- **re-tuning equivalent** if the upper bound of the median D1 < X;
- otherwise indeterminate.

Read alongside, moving no threshold: the re-tune fraction (a value near 1
with V1 present means 272 new parameters buy what full re-tuning buys); the
static contrast with its bound (static better than inserted by more than
margin M_s [proposed 0.02] is published as "design-in beats insertion", a
route finding, not a capacity negative); G(T-more) (if it exceeds G(M-sq),
the host is undertrained and the reading is labelled "capacity on an
undertrained host"); and N_eq.

**V2 add and remove**, co-primary, on the reference host:

- **overgrow-then-cut helps** if the lower bound of the median D2 ≥ X₂
  [proposed 0.01 nats];
- **hurts** if the upper bound of the median D2 < 0;
- otherwise indeterminate.

Reported alongside: the survival curves and the contribution-versus-random
contrast; cut without refit at h = 8 against M-small is the strict reading
(nothing re-learned after removal).

**Programme reading.** Rung M1 (generated builder) opens only on V1 present.
V1 re-tuning-equivalent sends the next study to the starved 0.5 host, where
the frontier sweep in this screen already measures the deficit. V2 is
published either way and sets the prior for rung M2.

**Frontier wording.** Every verdict that cites N_eq carries "same-recipe
frontier, 10 epochs" in its label.

## 8. Cost ledger [measured kernels from attempt 4; estimates for the rest]

Kernels: 1.0 host 37 to 43 s; a 24-candidate, 2-schedule oracle case about
50 s; dev evaluation 0.04 s; wide-grid fit about 1 s per 200 steps. The new
grid has 18 candidates with a 3,200-step arm, roughly 2.5 × the old case.

| Item | Count | Estimate |
|---|---|---:|
| Reference hosts, 6 anchors (replay attempt 4) | 6 × 40 s | 240 s |
| Frontier hosts, 4 widths × 6 seeds, 10 epochs | 24 × 25–90 s | 1,300 s |
| S-static hosts | 6 × 45 s | 270 s |
| T-more, 5 epochs per anchor | 6 × 20 s | 120 s |
| Oracle cases on the reference host: M-sq, R-272, R-aff, R-all, M-wide, M-small | 36 × 125 s | 4,500 s |
| Cuts and refits, 4 × 2 rankings × 6 anchors | 48 × 5 s | 240 s |
| Q1 pair on the 0.5 host | 12 × 100 s | 1,200 s |
| P1 known answer | 6 × 55 s | 330 s |
| Replays, twins, evaluations | | 300 s |
| **Pilot total, single stream** | | **about 8,500 s** |

Two cards halve the wall. R-all is the expensive re-tune (284,634
parameters per step); if its kernel exceeds 4 × the module kernel the pilot
reduces its grid to one seed and says so. Optional 20-epoch frontier adds
about 2,600 s.

**Ceiling** [John]: a fresh pilot ceiling of 3 GPU-hours (10,800 s) is
proposed, as in v4 §8. The 4,593 s left in the Gate 3 pilot cap belong to a
closed study; spending them here is John's call, not a default.
Confirmatory ceiling, after the pilot sizes the count: 8 GPU-hours
[proposed], no automatic extension.

## 9. Stage 1, pilot (requires this document committed with its SHA-256, the reviews preserved, and John's go)

Development anchors 3000–3005 on the reference host; frontier and static
seeds from the block in §11. Outputs to `docs/results/m0-pilot-<date>-<id>/`.

1. Host costs, replay of attempt 4's state hashes, both-card determinism,
   one oracle replay, co-tenancy replay (v4 §9 items 1 to 3).
2. P1 known answer and the zero-module identity (instrument).
3. Frontier hosts and the per-width loss, then the slope in nats per log₂
   parameter; S-static and T-more hosts.
4. Oracle cases, plateau fractions, per-arm kernels; the **per-anchor SD of
   D1 and D2** recorded before any median is opened.
5. Confirmatory count by the v4 §4 rule: n = max(16, n*) [proposed] where
   n* gives half-width ≤ X/2 for D1 under a normal approximation; the
   larger of the D1 and D2 counts; stop before freeze if the count exceeds
   the ceiling, with John's choice.
6. Development medians, for the record only.

**What the pilot may fill**: the grid's seed count (2 → 1, from timing
only), the R-all grid reduction, the confirmatory count, the kernels, the
replay rule. **What it may not change**: X, X₂, M_s, ρ, Hm, the plateau
fraction, conditioning and selection sizes, the schedule axis, the
truncation criterion, the arm definitions or the verdict logic.

## 10. Stage 2, confirmatory (separate go; margins frozen; fresh seeds)

n reference-host anchors from the confirmatory block; all bold arms; the
frontier is **not** retrained (its 24 pilot hosts are the scale reference
and are reported as such); S-static on every confirmatory anchor; T-more on
every anchor. V0 then V1 and V2 read once. A V0 failure ends the attempt and
burns its seeds. Any change to host, fit, cut or re-tune code after the
freeze is a new version with a new hash, a disposition and a fresh seed
block (v4 §7 rule).

## 11. Seeds and identity

Blocks already allocated and never reused: 1000–1031 (former training),
3000–3099 (Gate 3 development), 4000–4363 and 5000–5799 (Gate 3 confirmatory
and restarts, v4 to v4.2, all retired unused except 3000–3005 hosts), 4000s,
5000s, 8000s, 9000s, 14000s, 15000s, 24000s and 25000s (correction
protocol). This study [proposed]:

| Lineage | Seeds |
|---|---|
| Frontier hosts, pilot | 6000–6023 (s = 0.5: 6000–6005; 0.75: 6006–6011; 1.0: 6012–6017; 1.5: 6018–6023) |
| S-static hosts, pilot | 6100–6105 (paired by index with anchors 3000–3005: same data stream, module added) |
| Confirmatory reference anchors | 7000 onward, contiguous, ≤ 7099 |
| Confirmatory S-static | 7100 onward, paired by index |
| Restart after a failed confirmatory attempt | 7200– and 7300– |

Note on S-static pairing: the static host must consume the identical data
and augmentation stream as its anchor; the seed label differs only in the
"host-init" derivation for the module tensors. The implementation plan
states the derivation and the test that proves the streams are identical.

Identity: `protocol_version = "m0-v1"`, `protocol_sha256` = SHA-256 of this
file; receipts carry both plus the steering statement's SHA-256. The Gate 3
chain (v4, v4.1, v4.2) is not inherited.

## 12. Decisions John must make before the pilot

| ID | Decision | Proposed default |
|---|---|---|
| J1 | X, the capacity margin for D1 | 0.02 nats |
| J2 | X₂, the add-and-remove margin for D2 | 0.01 nats |
| J3 | M_s, the static-versus-inserted margin | 0.02 nats |
| J4 | Frontier widths and whether to add the 20-epoch budget | {0.5, 0.75, 1.0, 1.5}, 10 epochs only |
| J5 | Number of 0.5-host anchors for the Q1 pair | 6 |
| J6 | Pilot ceiling, and whether the Gate 3 remainder is spent here | fresh 3 GPU-hours |
| J7 | Confirmatory floor and ceiling | n ≥ 16, 8 GPU-hours |
| J8 | Seed blocks in §11 | as proposed |
| J9 | Gate reinterpretation in the steering statement §6 | as proposed |
| J10 | Merge of the steering package to `main` | — |

## 13. Review and acceptance of this draft

Independent review of this document against John's verbatim objective, a
scientific review for confounds and missing controls, and a reality review
against the repository's code and attempt-4 timings, each preserved under
`docs/reviews/m0-capacity/` with dispositions; then John's decisions J1 to
J9; then an implementation plan with its own exact-source reviews; then the
pilot go. The pilot may not open until this file's SHA-256 is pinned in the
package and the certificate names it.
