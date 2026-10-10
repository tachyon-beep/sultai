# Gate 3 headroom screen on a real CIFAR-10 host — DRAFT PREREGISTRATION v4

Status: **draft v4; freeze candidate with all owner numbers fixed; not frozen,
not executed.** Remaining before freeze: commit, gate-script widening, pilot.
Version history: v1 (SHA-256 `90f1f4a7…ef9540bb893d`) reviewed by a builder
check, an Astra-role scientific review and a torch feasibility review on
2026-10-11; v2/v3 dispositioned those and recorded John's decisions; v3
(SHA-256 `6338c814…36b52c1`) received a second independent Astra review on
2026-10-11 whose seven blockers and sixteen non-blocking findings are
dispositioned here (§15). All reports are preserved verbatim in
`docs/reviews/gate3-headroom/`.

**Freeze preconditions**, in order: (1) John fixes the two numbers introduced
by round two (§13 D-N, D-O) — done 2026-10-11; (2) this document is committed
as the versioned draft with its SHA-256 recorded, **before the pilot's go**; (3) the 5,000
dev-index artifact and the split-check output, already written under
`docs/results/gate3-split-2026-10-11/`, are committed; (4) the gate-script widening (§11) lands as its own reviewed
commit — done 2026-10-11, evidence `docs/results/gate3-gate-widening-2026-10-11/quality.json`; (5) the pilot fills only the values §9 permits, in the order §9
requires; (6) the filled document is committed with its SHA-256 before any
confirmatory run, as the 2026-10-09 correction protocol was.

Author: Claude (Nyx session "Sultai - Claude"), 2026-10-10/11, from John's
direction. Status tags: **[measured]** from a saved artifact,
**[reviewer-measured]** measured by a reviewer or by the author in-session and
not yet saved as a project artifact, **[estimated]** not yet measured,
**[proposed]** adopted from a source and open, **[decided]** fixed by John
with the date, **[pilot-fills]** a value the pilot sets by a rule declared
here, **[derived]** a reviewer's unmeasured derivation.

## 1. Why this screen and not the alternatives

Gate 2 closed on 2026-10-10 with the synthetic instrument software-valid and
scientifically negative: analytic four-template fitting beat the learned
former, harmful noisy-healthy admissions persisted, and formed trajectories
lost to matched-rate controls (`docs/certificates/2026-10-10-gate2-nyx-final.md`).
The correction protocol's Gate 3 reads "useful repair headroom: report
corrected within-family comparisons. Beyond-template transfer remains
unproven; no later gate inherits a synthetic pass."

John's sizing criterion (2026-10-10): the smallest study that gives a clear
signal, where "too small to really tell" and "too long to find out" are both
fail states.

| Candidate | Verdict | Reason |
|---|---|---|
| Tiny-MLP CPU screen (`2026-10-09-nyx-transfer-proposal.md`) | Rejected: too small to tell | A second synthetic family with a dense teacher; repair at the output site of a 32-unit MLP is close to linear, so analytic fitting wins by construction |
| Full Claude TCD (`docs/research/claude-handoff/sultai-concept-demonstrator-design.md`) | Rejected for now: too long to find out | 25–45 GPU-hours plus pilot, with ~750 hosts and ~12,000 oracle fits building a generator corpus before anyone knows whether there is headroom to generate into; designed inside Simic's conventions, which this repo may not modify |
| **This screen**: the Claude design's host and instrument gate, run as a headroom-only study | **Selected [decided 2026-10-11, D-A]** | Real CIFAR-10 host, known-answer impairments for the instrument, one trained-through family for headroom, every arm conventional, one gated pilot, both local cards |

**The question.** On a real host at a 16-channel site, with a deficit that is
repairable by information (the module can read what the bottleneck discarded)
but has no planted answer, does a generous gradient-fitted 272-parameter
module gain materially more than a cheap, selection-matched, few-step gradient
fit, **and is that gain specific to the bypass rather than generic
fine-tuning**? If not, a learned former has nothing to amortise and the
programme stops cheaply on a real host. If so, a corpus study is **not ruled
out**; it is not thereby justified.

**What a positive result does and does not close.** A positive, repair-specific
V1 establishes real-host repair headroom for a gradient-fitted module in the
primary family. It does not establish beyond-template transfer of a learned
former, which is the Gate 3 clause that remains open; the retrieval
diagnostic B4 is the only arm that touches cross-lineage predictability and it
has a descriptive reading only.

## 2. Authority, scope and what this is not

- Authorized so far: writing this draft and its reviews. The pilot (§9) and
  the confirmatory screen (§10) each require John's explicit go.
- Hardware: John stated on 2026-10-10 that both RTX 4060 Ti cards (2 × 16 GB)
  are available to Sultai when it is Sultai's turn. Simic rung 4 occupied both
  cards at the time of writing; Sultai does not start until John says the cards
  are free (D-K). No run touches Simic, Esper or ELSPETH. Simic's CIFAR bytes
  and split provenance are read once, offline; **no Sultai run imports Simic
  code** (D11).
- This is a headroom screen. It establishes no learned formation, no
  developmental advantage, no removal benefit, no half-parameter claim and no
  calibrated safety. D06 stays open; every arm here is supervised or
  gradient-based adaptation and is named as such. Admission (D25) is out of
  scope: every candidate is inserted raw and all harms are reported raw.
- The CIFAR-10 test batch is never read. Opening it is a separate John decision.
- The report records three separate fields, per the correction protocol:
  `software_valid` (ran to completion with provenance intact),
  `instrument_valid` (V0 passed) and the scientific verdict (V1 outcome with
  its specificity label, V2 descriptive). A valid negative is publishable.

## 3. Host, site, module and data

**Host** [proposed, Claude design Appendix A]: stem conv3×3 3→32, BN, ReLU;
conv3×3 32→32, BN, ReLU; max-pool to 16×16. Producer conv3×3 32→16, BN, ReLU
gives the site tensor **z** (16 channels × 16 × 16). Downstream conv3×3 16→64,
BN, ReLU; conv3×3 64→64, BN, ReLU; pool to 8×8; conv3×3 64→128, BN, ReLU;
conv3×3 128→128, BN, ReLU; pool to 4×4; global average pool written as
`mean(dim=(2, 3))`; linear 128→10. About 0.284M parameters including BN
affines [reviewer-measured by count; pilot prints it].

**Bottleneck host** [proposed]: as above with a trained-through 1×1 16→w, BN,
ReLU, 1×1 w→16, BN, ReLU block after z, output **z̃**, with width w = 4 first
and the pilot rule in §5 for revision.

**Module and residual base** [proposed]: W 16×16, b 16, 272 parameters, applied
per spatial position with α = 1.

- Site S1 (intact hosts): u = z′ + (W tanh(z′) + b), where z′ is the impaired
  tensor. W = 0, b = 0 is bitwise identity, so no-op equals the unmodified host.
- Site S2a (P5, after the bottleneck): u = z̃ + (W tanh(z̃) + b).
- Site S2b (P6, bypass): reads z, writes residually after the bottleneck:
  u = z̃ + (W tanh(z) + b). At W = 0, b = 0 this is z̃, so the no-op is the
  unmodified bottleneck host, not a bypass.
- The affine variant u = base + (W x + b) is the D21 folding control; it is
  fitted as arm A3 and is the only module family that can represent the
  linear inverses in P1/P2 exactly. The tanh family is primary for every
  gradient arm; no affine oracle is run.

**Training recipe** [measured from Simic `kernel_demo.Config` and
`bounded_data.RunSpec`, read-only; decided D-B]: SGD lr 0.05 constant,
momentum 0.9, Nesterov, weight decay 5e-4 excluding parameters with ndim ≤ 1,
batch 32, 10 epochs on the 40,000-example host-training partition;
normalisation mean (0.4914, 0.4822, 0.4465), std (0.2470, 0.2435, 0.2616);
reflect-pad-4 random crop and p = 0.5 horizontal flip with every draw made up
front on CPU; the uint8 dataset resident on the GPU. Disclosure: this recipe
was tuned by Simic against the same 5,000 dev examples that are this screen's
endpoint. Per-step host-side `float()` syncs from Simic's training loop are
not copied.

**Data** [proposed, reproduces Simic]: CIFAR-10 training set, 50,000 examples,
from a byte-verified copy of `/home/john/simic/runs/data/cifar-10-batches-py`.
The five `data_batch_*` files and `batches.meta` are verified individually by
SHA-256; the archive is never hashed because that would read test-batch
bytes. The loader is a five-file pickle reader behind a T3 boundary; no
torchvision.

Fit/dev permutation, written out exactly:

```
h = sha256()
h.update((20261004).to_bytes(8, "big"))
label = b"cifar-fit-dev-v1"
h.update(len(label).to_bytes(4, "big")); h.update(label)
seed = int.from_bytes(h.digest()[:8], "big")          # 6798730733443669431
perm = torch.randperm(50000, generator=torch.Generator().manual_seed(seed))
fit, dev = perm[:45000], perm[45000:50000]
```

Expected first eight indices `[17416, 42096, 29166, 33200, 42989, 19545, 24596, 13704]`;
SHA-256 of the full permutation as int64 little-endian bytes, no prefix,
`df1d87871f9afe3b15d7c5a434c31c5f68ff99564b4512410cc64fb1d042a24e`.
[measured 2026-10-11: `docs/results/gate3-split-2026-10-11/split-check.json`
records the reproduction on torch 2.9.1+cu128 (CPU generator) and its exact
prefix agreement with the 1,024 saved `fit_indices` and 256 saved
`dev_indices` in `/home/john/simic/runs/bounded-pilot-2026-10-08/manifest.json`,
read read-only; the torch reviewer reproduced the same values independently.]
The 5,000 dev indices are the artifact
`docs/results/gate3-split-2026-10-11/dev-indices-int32-le.bin` (int32
little-endian, SHA-256
`27173d10eb5edba1d0fd331e7f85ea633753cfae11400c4e7ef81810c94eeeba`) and are
the source of truth; the derivation is a cross-check only, because PyTorch
gives no cross-version `randperm` guarantee.

Sub-partition of the 45,000 fit examples [proposed]: one further permutation
of the fit index array by the same derivation with seed label
`"sultai-fit-partition-v1"` (base seed 20261004), sliced as 40,000 host
training, then 2,000 conditioning, then 3,000 selection. The 5,000 dev
examples are the sole endpoint. Conditioning and selection sizes are **not**
pilot-tunable, because the cheap comparator's definition depends on them.

**Seed ranges** [proposed]: development anchors 3000–3099; confirmatory intact
anchors 4000–4063; confirmatory bottleneck anchors 5000 onward, contiguous,
as many as n_P6 requires (at most 5199). A restart after a failed attempt
takes the next thousand block (4100–, 5200–) and records the linkage. Impairment, oracle and bootstrap generators are derived from the
anchor seed and a label; none is shared across anchors.

**Hard rules, each with a unit test and a negative control that shows the
test failing**:

1. **Every** BatchNorm layer in the host (stem, producer, bottleneck,
   downstream) is in eval mode whenever z is extracted, a module is fitted or
   an evaluation runs. z, z̃ for conditioning, selection and dev are computed
   once per host under `eval()` and `no_grad()` and cached. All host parameters
   have `requires_grad` false. Tests: a forward hook on every BN layer asserts
   `not training`; a SHA-256 over `named_parameters()` and `named_buffers()`
   is identical before and after every fit and evaluation; after every fit,
   every host parameter's gradient is `None`.
2. The CIFAR-10 test batch is never opened. A `sys.addaudithook` raising on
   any `open` event whose path contains `test_batch` is installed at process
   start in sealed-run mode; the five `data_batch_*` opens are counted and
   recorded in every result file.
3. Dev index artifact SHA-256 and permutation SHA-256 are checked before any
   run proceeds.
4. Every random draw (host init, data order, augmentation, impairment
   operators, oracle seeds, bootstrap) comes from a CPU `torch.Generator`
   seeded by the sha256 derivation above from the anchor seed and labels; host
   construction happens under an explicit derived `manual_seed`. One anchor
   runs entirely in one process pinned to one card by `CUDA_VISIBLE_DEVICES`,
   asserting `device_count() == 1`; anchor-to-card assignment is derived from
   the anchor index. No run spans both cards.
5. Determinism flag set, applied as the first statements of every process and
   recorded in every result file: `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before
   any CUDA initialisation; `torch.use_deterministic_algorithms(True)`;
   `cudnn.deterministic = True`; `cudnn.benchmark = False`; TF32 off for both
   cuDNN and matmul; no `torch.compile`, no AMP, no channels_last; contiguous
   NCHW throughout; `torch.set_num_threads(2)`. Replay rule: bitwise equality
   of the per-epoch parameter-and-buffer hash and the per-step loss stream
   between two processes on the same card, and on the other card
   [pilot-fills: if bitwise replay fails, the preregistration records a
   tolerance rule and the reason, decided before freeze].
6. Source identity: the committed source aggregate hash, this document's
   SHA-256, `torch.__version__`, `torch.version.cuda` and the installed wheel
   string are written into every result file.

## 4. Lineages and splits

An **anchor** is one host trained from its own initialisation with its own
data order and augmentation. Because no arm continues host training, every
arm is evaluated on the same frozen host and pairing across arms is automatic;
branches would be correlated replicates that reduce the effective sample. So
**one host per anchor, no branches, no spawn point, no basin check**. The
anchor is the unit of analysis.

| Family | Host | Anchors | Role |
|---|---|---|---|
| Intact | §3 host | 64, fixed | Post-hoc impairments P1, P2, P3, P3′; null calibrator P8 |
| Bottleneck | §3 bottleneck host | 32 minimum, sized by the §9 rule | P5 matched control and P6 bypass headroom on the same host |
| Starved | downstream widths 32, 32, 64, 64 | 0 | P10 deferred [decided 2026-10-11, D-C] |

The intact count is fixed because its families serve V0 and descriptive
readings only. The bottleneck count is **n_P6 = max(32, n\*)**, where n\* is the
smallest n at which the one-sided 95% percentile-bootstrap half-width of the
median H′, computed from the pilot's per-anchor SD of H′ under a normal
approximation, is ≤ X/2 (half-width ≈ 1.645 × 1.25 × σ/√n, so σ = 0.02
gives n\* ≈ 17 and 32 stands; σ = 0.03 gives ≈ 38; σ = 0.05 gives ≈ 106
[derived]). The pilot reports that SD **before** any H′ median is opened
(§9). **If n\* exceeds what the confirmatory ceiling (§8) supports at the
pilot's measured per-anchor cost, the pilot reports both numbers and the
screen stops before freeze**; John then chooses between raising the ceiling
and accepting n = 32 with the resulting half-width stated in the frozen text.
The pilot makes neither choice. Counts are never lowered and never changed
after any confirmatory anchor trains. Development anchors are drawn from their own seed
range and are never reused in confirmation.

## 5. Impairment families and what each one is for

| ID | Impairment | Regime | Site | Ground truth | Purpose |
|---|---|---|---|---:|---|
| P1 | Attenuate a random group of g = 4 channels by α ∈ {0.1, 0.3} | Post-hoc | S1 | Exact inverse | Instrument: known answer; B3 validated here |
| P2 | Ill-conditioned mixing T = U diag(s) Vᵀ, Haar U, V, log-spaced s to s_min = 0.1 | Post-hoc | S1 | Exact inverse | Instrument: full-rank known answer |
| P3 | Per-channel bias shift c, Haar-random direction, norm κ·RMS(z) | Post-hoc | S1 | Exact inverse | Instrument: bias-only answer |
| P3′ | As P3 with direction ∝ (+1, −1, +1, …) | Post-hoc | S1 | Exact inverse, inside the former's template span | Representability control for A7 |
| P5 | Bottleneck host, module at S2a | Trained-through | S2a | Locally unrepairable | Matched-pair control and paired null for P6 on the same host |
| P6 | Same host, module at S2b | Trained-through | S2b | Repairable by information, no planted answer | **Primary headroom family** |
| P8 | Intact host, no impairment | n/a | S1 | None | Null calibrator: stitching reference F8 in the report; also reported as secondary "fine-tuning headroom" |
| P10 | Starved host | Trained-through | S1 | Capacity deficit | Deferred |

Dropped from the Claude design for this screen: P4 (probe tier), P7
(wrong-site), P9 (non-capacity nulls). Each answers a later question.

**Draw granularity**: P1's channel group, P2's U and V, P3's direction are
drawn once per anchor from derived seeds. RMS(z) is computed per anchor over
the conditioning set.

**Magnitude ladders and the materiality rule** [pilot-fills]: on development
intact anchors the pilot computes A1 gain for P1, P2 and P3. For each family
the magnitude is the first rung of its ladder at which A1 gain ≥ 0.05 nats on
every development anchor: P3 κ ∈ {0.25, 0.5, 1.0}; P1 α-pair ∈ {{0.1, 0.3},
{0.03, 0.1}}; P2 s_min ∈ {0.1, 0.03}. A1 is exact, so these choices depend on
nothing any arm produces. Magnitudes are revised **before freeze** only, and
never after a confirmatory result.

**P6 is repairable by information, not by guarantee.** The bypass gives the
module the pre-bottleneck z, but a downstream trained on bottlenecked
features may not use the extra information when it is offered. The bottleneck
width is therefore a pilot-revisable magnitude: the pilot tries w = 4 first;
if (B3 gain at P6) − (B3 gain at P5) < 0.05 nats on any development host, it
tries w = 2, because a narrower bottleneck discards more and makes the known
answer more likely, not less; w = 8 is tried only if the w = 4 block trained
degenerately (collapsed units or non-finite BN statistics), which is a
different failure. If no width in {4, 2, 8} yields the known answer on every
development host, the primary family is undefined and the screen **stops
before freeze** with that finding published. The width decision reads B3 on
P5/P6 only, and is made before any cheap-arm or H′ summary is opened (§9).

**Two regimes, two arm tables.** On post-hoc families the intact site tensor
is a regression target, so supervised arms and the existing former can run.
On trained-through families and P8 no intact target exists; the only repair
route is loss-gradient fitting through the frozen host. Arms that need an
intact target are **absent** from the headroom families by construction.

## 6. Arms

All arms receive the same 2,000 cached conditioning examples and the same
3,000 cached selection examples; candidate choice uses selection loss only.
Every arm's cost is booked in example passes and optimiser parameter-steps.
A candidate whose fit produces a non-finite loss is marked non-finite, has
selection loss +∞, cannot be selected, and is counted in the ledger; it is
not a stop (§9).

**Definitions used below.**

- A **full-batch step** is one gradient of the mean cross-entropy over all
  2,000 conditioning examples with respect to (W, b) only, through the frozen
  host, plain SGD, no momentum, no weight decay, zero init.
- A **minibatch step** is the same over a minibatch of 256 conditioning
  examples (8 steps per epoch).
- **η-set (oracle)** = {0.3, 0.1, 0.03, 0.01}, capped by the stability of
  800 minibatch steps.
- **η-set-cheap** = {3, 1, 0.3, 0.1, 0.03} [decided 2026-10-11, D-O], the few-step
  regime's own range; it is not inherited from the oracle's stability cap.

**Table A — post-hoc families P1, P2, P3, P3′ (instrument and former transfer)**

| Arm | Fit | Information |
|---|---|---|
| A0 no-op | none | none |
| A1 exact inverse | closed form from the planted operator; an affine operator, not a module fit | privileged, ground truth |
| A2 nonlinear-272 ridge | ridge (1e-8) of intact z on tanh(z′) over all conditioning positions (2,000 × 256 pairs), as `repair.fit`, implemented in torch with a known-answer test against `repair.fit` on a small fixture | supervised, intact target |
| A3 affine-272 ridge | ridge (1e-8) of intact z on z′ over the same pairs; the D21 folding control and the family that can represent A1 | supervised, intact target |
| A7 existing former | `FrozenGenerator.form` on position pairs; the generator is re-derived in-process by `run_formation` with seeds 1000–1031, evidence mode `paired`; the SHA-256 of the 11×4 meta-regression `weights` and of the `run_formation` report are recorded at freeze and asserted at run time; no refit | supervised evidence, four-template span, trained on the synthetic family |
| A8 analytic four-template fit | least squares for the four public template coefficients on the same position pairs (`formation_controls.analytic_template_fit`); the control that beat the former at Gate 2 | supervised, same span as A7 |
| B3 oracle (P1 α-low only) | as Table B | first-order, generous |

A7 position pairs: 1,024 class-balanced conditioning images × 4 stratified
positions = 4,096 pairs, built directly as tuples. A7 and A8 are **descriptive**
(§7 V2); A2 is reported as the 272-family ceiling for both. Expected under
D21 and predeclared: A3 ≥ A2 on P1/P2 because the inverse is affine and tanh
saturates on post-ReLU activations. Predeclared limit of V2: an A7 shortfall
against A8 on P3′ has two undecomposed explanations, the former's regression
quality and the input-distribution shift from the synthetic U(−2, 2) evidence
to real post-ReLU-plus-bias activations; V2 does not separate them.

**Table B — headroom family P6, matched control P5, null calibrator P8**

| Arm | Fit | Information |
|---|---|---|
| B0 no-op | none; equals the unmodified host bitwise | none |
| B1, B2, B2₁₆ cheap SGD, fixed rate | k ∈ {1, 4, 16} full-batch steps at a **fixed** rate η_B [pilot-fills by rule: the η in η-set-cheap with the best mean 4-step selection loss over development P6 anchors, chosen before any confirmatory anchor trains]; not selection-chosen per anchor | first-order, one candidate |
| B2′(k) selection-matched cheap SGD | k full-batch steps at each η in η-set-cheap, best on selection; defined for k ∈ {1, 4, 16, 64}; **B2′ = B2′(4)**. Matches the learning-rate choice; B3 retains a 6× candidate advantage (seeds × schedule) and a minibatch budget, declared as part of its generosity; B3's N(0, 1e-3) init versus zero init is immaterial at that scale [derived] | first-order, five candidates |
| B3 oracle | minibatch SGD, init N(0, 1e-3), 3 seeds (init and order) × η-set × schedule {1× = 200 steps, 4× = 800 steps}, 24 candidates, best on selection. **Plateau certified** when (i) the best-on-selection 4× candidate improves on the best-on-selection 1× candidate by less than 2 × the per-example paired bootstrap SE on the **selection** set, and (ii) the 4× candidate's mean training loss over steps 721–760 and 761–800 differ by less than 2 × the SE of those step losses. Uncertified anchors are retained and flagged, never dropped | first-order, generous, 24 candidates |
| B4 retrieval | the B3 repair of the nearest **other anchor** by Euclidean distance on aligned per-channel mean and covariance of z over a fixed 1,024-image **conditioning** batch. Hungarian channel alignment on activation correlations over the same batch, on **both** sides: P_in on the read tensor (z) and P_out on the write tensor (z̃ for P6, z̃ for P5, z for P8); transported as W′ = P_outᵀ W P_in, b′ = P_outᵀ b | cross-anchor memory |
| B5 wide adapter | B3 with a 16→64→16 module, tanh after the first layer, reduced grid 1 seed × η ∈ {0.1, 0.03} × 1×, on 8 P6 anchors only | capacity diagnostic |
| B6 steps-to-match | smallest k ∈ {1, 4, 16, 64} at which B2′(k) is within X of B3, per anchor | diagnostic |

B4's predeclared descriptive reading: retrievable fraction = B4 gain / B3 gain
per anchor. B5 and B6 are diagnostics with no verdict.

## 7. Endpoints, noise, verdicts

**Endpoint**: mean cross-entropy in nats on the 5,000 dev examples with the
module inserted and the host frozen. Gain = L(no-op) − L(arm). Per-anchor
values are the mean over that anchor's cases. Headline intervals are
**percentile** cluster bootstraps over anchors with 10,000 resamples; a
per-example paired bootstrap over dev is reported as a second interval that
captures endpoint sampling. For transparency: the one-sided 95% percentile
bound of a median over 32 anchors sits near the 11th–12th order statistic
[derived], so "present" below is equivalent to roughly 21 or more of 32
anchors having H′ ≥ X, and "absent" to roughly 11 or fewer.

**Noise floor and twin**: the no-op twin on a resampled evaluation order is a
**bitwise determinism assertion**, not a noise floor; a mismatch is a stop.
The floor for any *difference* between two arms on one host is the
per-example paired bootstrap SE over the 5,000 dev examples (over the 3,000
selection examples where the quantity is computed on selection).

**Descriptive scale reported with every verdict**: the bottleneck deficit
L(bottleneck host) − L(intact host) per bottleneck anchor and its interval;
B3 gain on P6 as a fraction of that deficit; F8, the median B3 gain on P8
with its interval, as the stitching reference; (median B3 gain on P6) − F8.

All margins were fixed by John on 2026-10-11 before any pilot or confirmatory
outcome existed, and the order in §9 keeps it that way.

**V0 Instrument gate** (all clauses must pass before V1 is read; anchor is
the unit; "cases" are per-anchor means; every clause can fail):

1. Impairment materiality: A1 gain ≥ 0.05 nats on ≥ 90% of anchors in each of
   P1, P2, P3, P3′. Anchors below 0.05 in a family are listed and excluded
   from the denominators of clauses 2 and 3 for that family. A1 is exact, so
   this exclusion depends on no arm's outcome.
2. Supervised family works: A3 loss within 0.01 nats of A1 loss on ≥ 90% of
   the P1, P2, P3, P3′ anchors that passed clause 1. A1 itself is checked
   against the intact host loss with an explicit tolerance of 1e-4 nats.
3. **The oracle used by V1 finds repairs that exist**: on P1 at its lower α,
   B3 gain ≥ ρ · A3 gain on ≥ 90% of anchors that passed clause 1
   [decided 2026-10-11: ρ = 0.8]. Two questions are kept apart: B3 versus A2
   asks whether the optimiser found the best member of its own tanh family;
   B3 versus A3 asks whether the family plus optimiser reach the affine
   ceiling that can represent the answer. Both are reported; the gate is on
   ρ against A3 as the stricter bar.
4. **Matched-pair known answer**: on bottleneck anchors, (B3 gain at P6) −
   (B3 gain at P5) ≥ 0.05 nats on ≥ 90% of anchors. P5 is judged relative to
   P6 on the same host, not against zero; P5 absolute gain is reported.
5. Twin bitwise on every anchor; one development-anchor replay passes under
   the §3 rule; no non-finite loss in host training, in any no-op evaluation
   or in the evaluation of any selected candidate.

**Fail → the attempt ends.** Confirmatory anchors are never reused after a V0
failure. Any change to host, impairment or fit code requires a new
preregistration version with a new SHA-256, an explicit disposition and a
fresh confirmatory seed range.

**V1 Headroom**, primary family P6 only:

- H′ = per-anchor (B3 gain − B2′ gain) on P6. Statistic: median over the n_P6
  anchors; one-sided 95% percentile anchor-bootstrap bounds.
- **Headroom present** if the lower bound of the median H′ ≥ **X**
  [decided 2026-10-11: X = 0.02 nats, a worthwhile repair margin of roughly
  2–3% of the expected host loss, fixed before any outcome, per D15].
- **No headroom** if the upper bound of the median H′ < X, **provided** at
  most **U**% of P6 anchors are plateau-uncertified [decided 2026-10-11: U = 25, D-N].
  If more than U% are uncertified, this outcome is reported as
  "indeterminate: oracle not at plateau" and is not a programme stop.
  "Present" is unaffected by certification, because an unconverged oracle
  biases against it.
- Otherwise **indeterminate**, reported as such; no threshold moves.
- **Repair specificity**, read alongside and moving no threshold:
  ΔH′ = H′(P6) − H′(P5) per bottleneck host, paired on the same host, same
  grid, same cheap fit, P5 lacking the bypass information. Report the
  one-sided 95% percentile lower bound of the median ΔH′. Label the outcome
  **repair-specific** if that bound > 0, otherwise **generic fine-tuning
  headroom**. A programme go toward any corpus work requires **present and
  repair-specific**; "present and generic" is published as a negative for
  the repair hypothesis.
- Reported alongside, with no verdict role: H = B3 − B2 (fixed-rate), B6
  steps-to-match, B3 wall cost per anchor, the descriptive scale quantities
  above.
- P8 is reported with identical statistics under the label "fine-tuning
  headroom on an intact host"; it is not a programme trigger. There is no
  rule combining families; P6 is the only primary.

**V2 Former transfer**, descriptive only, on P1–P3 and P3′:

- Report A7 gain, A8 gain and A2 gain per anchor with intervals. The
  informative comparison is A7 versus A8 on P3′, where the answer lies inside
  the former's span; on P1–P3 the span cannot represent the inverse and the
  comparison is reported as such. No verdict, no tolerance.

**Harm screen**, all families: every arm's raw candidate that degrades an
anchor by more than **Hm** nats versus no-op [decided 2026-10-11: Hm = 0.01
nats] is listed per anchor; no anchor is dropped. Admission is out of scope.

## 8. Cost ledger [estimated, single-stream, torch-review basis; pilot measures every line]

| Item | Count | Unit estimate | Total |
|---|---:|---|---|
| Intact host training | 64 × 12,500 steps | 50–130 s/host, launch-bound at batch 32 | 0.9–2.3 GPU-h |
| Bottleneck host training | 32+ | same | 0.45–1.2 GPU-h |
| B3 grid, minibatch definition | (32 P5 + 32 P6 + 64 P8 + 64 P1) = 192 host-cases × 24 fits | ~5 ms/step: 1× ≈ 1 s, 4× ≈ 4 s; ≈ 60 s per host-case | ~3.2 GPU-h (up to ~8 if steps run slower) |
| B1/B2/B2′(k)/B6 full-batch arms | 128 host-cases × 5 rates × ≤ 64 steps | 50–100 ms/step | ~1.2 GPU-h |
| B5 reduced | 8 anchors × 2 fits | ~1 s | negligible |
| A2, A3, A7, A8, B4 | ~800 fits | < 1 s in torch | ~0.3 GPU-h |
| Evaluations | ~6,000 × 5,000 examples | ms | ~0.3 GPU-h |
| **Total** | | | **~6–13 GPU-h single-stream** |

Three or four processes per card, each pinned, recover most of the
launch-bound time; wall on two cards ≈ 2–4 h. Peak memory per process is
under 1.5 GB; 16 GB per card is not a constraint.

**Budget ceilings [decided 2026-10-11]**: pilot ≤ 3 GPU-hours; confirmatory
≤ 16 GPU-hours across both cards, no automatic extension; wall-clock stop at
10 hours. Each stage still requires its own explicit go from John to start.

## 9. Stage 1 — pilot (requires the committed versioned draft and John's go)

**Precondition**: X, ρ, Hm, U and η-set-cheap are fixed in this document and
the document is committed with its SHA-256 recorded **before** the pilot's go.
The pilot may not be opened until that commit exists.

Development anchors only: 6 intact, 6 bottleneck, from seed range 3000–3099.
Outputs to `docs/results/gate3-pilot-<date>-<id>/`:

1. Per-host training time, peak memory, parameter count.
2. Per-fit time for B3 1× and 4×, and per full-batch step.
3. Determinism: one forward/backward of the host on a 2-image batch under the
   full flag set; two full anchor trainings in separate processes on the same
   card compared by per-epoch hash and per-step loss stream; the same on the
   other card; one B3 fit replay; replay under the confirmatory co-tenancy.
4. Materiality and known answers, **B3 and A-arms only**: A1 gain on P1, P2,
   P3 along their ladders; fills the magnitudes. B3 versus A3 on P1 at its
   lower α on every development intact anchor; a fraction below ρ on any
   development anchor is an instrument finding that requires a new draft
   before freeze (the pilot may not change B3's definition). P6 − P5 known
   answer at w = 4, then w = 2, then w = 8 under the §5 rule; fills the width
   or stops the screen.
5. η_B by the declared rule on development P6 anchors.
6. Per-anchor SD of H′ on development P6 anchors (n = 6), reported and the
   count rule in §4 applied **before** any H′ median is opened; then the
   development medians, for the record only.
7. Known-answer unit tests and their negative controls: A1 restores within
   1e-4; BN eval assertion and hash; test batch sealed; dev index SHA; W = 0
   identity; a non-finite grid candidate is marked and not selected.

**Ordering rule**: items 1–3 are read and any seed-count reduction decided
**before** item 4 is opened; item 4 is decided before item 5 is opened; in
item 6 the SD and the count are recorded before the median is opened.

**What the pilot may fill**: the magnitudes (κ, the P1 α-pair, P2 s_min), the
bottleneck width, η_B, the replay rule, the bottleneck anchor count by the §4
rule (raise only), and the B3 seed count (3 → 2 → 1, from timing only; the one
permitted definitional change). **What it may not change**: X, ρ, Hm, U,
η-set, η-set-cheap, conditioning and selection sizes, the schedule axis, the
verdict logic, or any other arm's definition.

**Stop rules** (pilot and confirmatory): timeout; provenance mismatch; test
batch requested; BN train mode detected; wrong card used; replay mismatch;
non-finite loss in host training, in any no-op evaluation or in the
evaluation of a selected candidate. A non-finite loss inside a grid
candidate's fit is **not** a stop: the candidate is marked, cannot be selected
and is counted; an anchor whose every candidate in an arm is non-finite is
retained, flagged and reported.

## 10. Stage 2 — confirmatory screen (requires freeze; John's conditional go recorded as D-P)

John's go of 2026-10-11 is conditional on the pilot completing cleanly as
defined in §13 D-P; any deviance is a hold and returns to John. Runs once on
confirmatory anchors, one attempt, with an exact replay of one
anchor per family afterwards. Publication is the complete report regardless
of verdict. Stops as in §9 plus the budget ceiling. Restart is a new attempt
with linked provenance, a new preregistration version and fresh anchors from
the next seed block; no budget reset.

## 11. Implementation outline (no screen code until the pilot's go; gate widening first)

- **Package placement [decided 2026-10-11, D-L]**: the screen lives at
  `src/sultai_screen/`. The repo's typing and policy gates currently scan only
  `src/sultai` (`check_policy.py` by `rglob`, `check_quality.py` by
  non-recursive glob and a fixed mypy target). Both scripts are widened to a
  declared list of source roots, their negative controls are extended to
  mutate a file in the new package, the new package is added to ruff's
  first-party list, and the mypy subprocess timeout is set in the same change
  with a measured cold-run time on a torch-importing probe. The widening is
  committed and reviewed as its own change before any screen code lands. The
  stdlib instrument's own gate results are unchanged by the widening and that
  is asserted by re-running it. **Done 2026-10-11**: `SOURCE_ROOTS` in
  `scripts/check_policy.py` is the single declared list, both gates scan it
  recursively, `src/sultai_screen/__init__.py` is a docstring-only placeholder,
  three new negative controls mutate the placeholder (type, lint, planted
  `Any`), the per-check ceiling is 300 s, and the twelve instrument module
  hashes equal the frozen 2026-10-09 evidence.
- Optional dependency group `screen = ["torch==2.9.1+cu128", "numpy==2.3.5"]`
  with the PyTorch cu128 index recorded [measured versions on Nyx].
- Modules: `data.py` (sealed five-file loader, audit hook, split derivation
  and artifact check, hashes), `host.py` (host, bottleneck variant, cached-z
  interface as the primary abstraction, site definitions S1/S2a/S2b),
  `impair.py` (P1–P3′ operators and exact inverses), `fit.py` (torch ridge
  with the `repair.fit` known-answer test, full-batch and minibatch SGD,
  oracle grid and plateau rule, non-finite marking, retrieval and two-sided
  alignment), `screen.py` (run plan, ledger, report fields, provenance),
  `former_bridge.py` (position-pair extraction, A7 artifact assertion, A8).
- Strict typing idioms the package uses so that no `Any`, cast or ignore is
  needed: `torch.autograd.grad(loss, [W, b])` instead of `backward()`;
  module outputs narrowed through one runtime-checked helper
  `as_tensor(obj: object) -> Tensor` using `isinstance`, so there is no silent
  annotated-local narrowing; hashing via `named_parameters()` and
  `named_buffers()`; `.numpy().tobytes()` never `.tolist()`; pickle and
  `torch.load(weights_only=True)` typed `object` through `@t3_boundary` with
  paired tests; `np.ndarray[tuple[int, ...], np.dtype[np.uint8]]` annotations;
  a dedicated T2 failure class for oracle non-convergence. The cuDNN version is
  not recorded through its untyped call; torch and CUDA version strings are.
- Hungarian alignment at n = 16 is implemented in-package; no scipy.
- Runs from a committed source with the aggregate hash in every output.

## 12. Reviews

Round 1, on v1: `builder-check-v1.md`, `astra-scientific-review-v1.md`,
`torch-feasibility-review-v1.md`. Round 2, on v3:
`astra-scientific-review-v3.md`. All in `docs/reviews/gate3-headroom/`,
dispositioned in §15. Round 2's own verdict: a v4 with F1–F7 applied and the
owner numbers fixed in the order F1 requires is ready to freeze; whether v4
receives a further independent read is John's call.

## 13. Owner decisions, collected

| # | Decision | Value | Status |
|---|---|---|---|
| D-A | Accept this screen over the alternatives in §1 | yes | decided 2026-10-11 |
| D-B | Host recipe: Simic defaults as-is, with the endpoint-tuning disclosure | yes | decided 2026-10-11 |
| D-C | P10 starved family | defer | decided 2026-10-11 |
| D-D | X headroom margin, as a worthwhile repair margin under D15 | 0.02 nats | decided 2026-10-11 |
| D-E | ρ oracle-recovers-fraction for V0 clause 3 | 0.8 | decided 2026-10-11 |
| D-F | Keep A7/A8/V2 as descriptive arms | keep | decided 2026-10-11 |
| D-G | Hm harm tolerance | 0.01 nats | decided 2026-10-11 |
| D-H | Pilot GPU-hour ceiling | 3 | decided 2026-10-11 |
| D-I | Confirmatory GPU-hour and wall ceilings | 16 / 10 h, no automatic extension | decided 2026-10-11 |
| D-J | Tracker shape | new issue `sultai-ef0da96d8a` | decided 2026-10-11 |
| D-K | Cards free of Simic rung 4 | — | pending |
| D-L | Package placement and gate-script widening | widen the gate scripts as a reviewed change | decided 2026-10-11 |
| D-M | Role of P8 | null calibrator, secondary report kept | decided 2026-10-11 |
| D-N | U, maximum fraction of plateau-uncertified P6 anchors for a "no headroom" reading | 25% | decided 2026-10-11 (John: take the proposals) |
| D-O | η-set-cheap, the few-step arms' learning-rate set | {3, 1, 0.3, 0.1, 0.03} | decided 2026-10-11 (John: take the proposals) |
| D-P | Confirmatory go | **Conditional go, 2026-10-11**: confirmed subject to the pilot completing cleanly; hold on any deviance. "Cleanly" means every §9 stop rule unfired, every pilot item 1–7 delivered, the width found in {4, 2, 8}, clause-3 fraction ≥ ρ on every development anchor, and n\* within the §8 ceiling. Any deviance returns to John before freeze | §10 |

## 14. What this document does not change

`docs/RESUME.md`, `README.md`, the gate scripts and the stdlib instrument are
unchanged by this draft. The three exact research originals remain a separate
provenance blocker and are not an input to this screen.

## 15. Change log with review dispositions

### v1 → v2 (round 1: builder check, Astra, torch)

| Finding | Disposition |
|---|---|
| Builder Q1, Astra B2: cheap comparator unspecified; H a free parameter | Adopted. Step definitions; B1/B2 at fixed η_B by rule; B2′ added; V1 reads H′; not pilot-tunable. Round 2 found the shared η-set a new defect: see F4 below |
| Builder Q2, Astra B1: P5 absolute ≈ 0 contradicts P8 headroom | Adopted. P5 relative to P6 on the same host; P8 null calibrator (D-M). Round 2 found the calibrator functionless: see F5 below |
| Builder Q3, Astra N6, torch N6: A1 affine; tanh cannot reach it | Adopted. V0 restructured; A3 ≥ A2 predeclared under D21 |
| Builder Q4, Astra B3, torch N6: twin is a determinism check | Adopted. Twin bitwise; paired bootstrap SE floor; A1 tolerance 1e-4 |
| Builder Q5, Astra B6, torch B2: former span; V2 foregone | Adopted. V2 descriptive; A8 and P3′ added; artifact named. Round 2: artifact hash not computable at freeze, see N4 below |
| Builder Q6: P3 norm unspecified | Adopted. κ·RMS(z) with a declared ladder |
| Advisor on v2: P6 "by construction"; clause 3 mixes two questions | Adopted. Width rule with declared stop; clause 3 separated. Round 2: wording survived in §1 and the table, see N8 below |
| Astra B4: V1 point estimate; X from noise | Adopted. Interval reading; X as a repair margin (D-D) |
| Astra B5: B3 never validated on a known answer | Adopted. B3 on P1; V0 clause 3; A6 retired |
| Astra B7: fix-and-reread | Adopted. Attempt ends; new version and anchors |
| Astra N1–N11, torch B1, B3, B4, N1–N5, N7 | Adopted as recorded in v2; round 2 verified each against the text (see its Part 1 table) |

### v3 → v4 (round 2: Astra on v3)

| Finding | Disposition |
|---|---|
| F1: owner numbers could be fixed after the pilot's dev H′ is seen | Adopted. Freeze precondition (2) and the §9 precondition: numbers fixed and the document committed before the pilot's go; pilot item 6 records SD and count before the median is opened |
| F2: one non-finite grid candidate ends a 16 GPU-h attempt | Adopted. Stop rule split (§9); non-finite candidates marked, +∞ on selection, counted, never a stop; all-non-finite anchors retained and flagged |
| F3: "no headroom" with the oracle unconverged | Adopted. U% plateau guard; "indeterminate: oracle not at plateau" is not a programme stop. U is owner D-N |
| F4: cheap-arm rate range inherited from the oracle's stability cap | Adopted. Separate η-set-cheap for B1/B2/B2′/B6 and the η_B rule; B3 keeps η-set; 5 versus 24 candidates declared as B3's generosity. The set is owner D-O |
| F5: "present" can fire from generic stitching; clause 5 cannot fail | Adopted. ΔH′ = H′(P6) − H′(P5) paired on the same host; repair-specific label; programme go requires present and repair-specific; F8 moved out of V0 into the descriptive scale; every V0 clause can now fail |
| F6: clause 1 "every" anchor | Adopted. ≥ 90% per family; sub-0.05 anchors listed and excluded from clauses 2–3 denominators; exclusion depends on A1 only |
| F7: count-raising without a rule | Adopted. n_P6 = max(32, n\*) from the pilot SD under a normal approximation, half-width ≤ X/2; SD recorded before the median; intact count fixed at 64 |
| N1: plateau rule on dev; non-computable flatness clause | Adopted. Selection-set SE; steps 721–760 versus 761–800 against the step-loss SE; best-within-schedule defined as best-on-selection |
| N2: clause 3 never exercised in the pilot | Adopted. §9 item 4 |
| N3: "isolates" overstated; init asymmetry | Adopted. Row text rewritten; init asymmetry declared immaterial [derived] |
| N4: A7 artifact hash not computable at freeze | Adopted. SHA-256 of the 11×4 meta-regression weights and the `run_formation` report |
| N5: permitted fills inconsistent; seed cut is a definitional change | Adopted. Explicit permitted and forbidden lists; seed count the one permitted definitional change, from timing only; P1 and P2 ladders declared |
| N6: no ordering inside pilot items 4–6 | Adopted. Item 4 before 5 before 6; SD and count before median |
| N7: width order unjustified | Adopted. w = 2 next because narrower discards more; w = 8 only for a degenerate block |
| N8: "by construction" survived | Adopted. §1 and §5 table now say "by information" |
| N9: B4 write-side alignment | Adopted. P_in and P_out with the transport formula |
| N10: "no code until the freeze" contradicts the pilot | Adopted. §11 heading; gate widening first with the mypy timeout folded in |
| N11: [measured] tag misuse; artifact not a precondition | Adopted. Artifact and check output written to `docs/results/gate3-split-2026-10-11/` on 2026-10-11; tag now measured; committing them is freeze precondition (3) |
| N12: sub-partition derivation and seed ranges unstated | Adopted. One further permutation with its label; numeric seed ranges and restart blocks |
| N13: clause 2 scope; A3 ridge; pure-Python ridge cost | Adopted. P3′ included; ridge 1e-8; torch ridge with a known-answer test against `repair.fit` |
| N14: bottleneck deficit never reported | Adopted. Descriptive scale paragraph in §7 |
| N15: software and scientific fields | Adopted. §2 last bullet |
| N16: bootstrap method unstated | Adopted. Percentile, with the order-statistic equivalence stated |
| D-F consequence: V2 cannot separate regression quality from distribution shift | Adopted. Predeclared after Table A |
| Advisor on v4: sizing rule can exceed the ceiling; bottleneck seed range too small; ridge pair count unstated | Adopted. Declared stop-before-freeze with John's choice; contiguous range to 5199; 2,000 × 256 pairs |
