# M0 pilot V0 determinism and replay review

Reviewer: determinism reviewer (Claude), 2026-10-14. Scope: the V0 instrument half of
`docs/results/m0-pilot-2026-10-14-0d4179a/` against the preregistration (§4, §7 V0, §8, §9 stage 1)
and amendments 3 and 4. Read-only; no GPU; CIFAR batch files not opened. Code read only to fix
field meanings (`src/sultai_screen/host.py`, `m0/worker.py`, `m0/pilot.py`, `m0/static.py`,
`m0/calibrate.py`, `m0/provenance.py`, `m0/identity.py`, `data.py`).

Equivalence predicate reviewed: bit-exact identity of `state_hash` (sha256 over named parameters
and buffers with kind, dtype and shape, `host.py:103-109`), of the float32 loss-stream bytes and of
the serialized checkpoint bytes; the evaluation and fold-residual checks are the only
tolerance-based comparisons (1e-6).

## Summary

The V0 receipts establish what they claim: the seed-3000 anchor is bitwise identical on both cards
(12,500 steps, 10 epoch hashes, loss stream, 2-image gradient probe, serialized checkpoint bytes)
and bitwise identical to attempt 4 on all six anchors; resume, static step 0 and evaluation checks
hold on both cards; identity and custody are consistent across all 174 children. No finding blocks
the pilot's V0 verdict. The most important gap: the BatchNorm-recalibrated N0 host that every later
arm loads (`noop_sha256`, all six anchors chose R-bn) was computed exactly once, on one card, and is
never replayed anywhere; the environment identity omits driver, cuDNN and GPU model, so the
attempt-4 lineage rests on card UUIDs and torch/CUDA strings alone.

Findings: 0 blocking, 2 major, 4 minor, 3 observations.

## Findings

**F1. Major. The no-op (R-bn) recalibration path has no replay evidence; its hash is single-sourced.**
Evidence: `pilot.py:404-406` compares `REPLAY_KEYS` (`epoch_state_sha256`, `loss_stream_sha256`,
`final_state_sha256`, `forward_backward`) between the two replay children; `pilot.py:424-426`
compares only `straight_epoch_state_sha256` and `checkpoint_rate` between the acceptance children.
`recalibrate_bn` (`calibrate.py:25-45`: BN train-mode pass over `seen_fit`, chunks of 128,
`momentum=None`) runs once per anchor in `noop-<seed>` (all on card 0 for 3000; e.g.
`noop-3000.process.json gpu: 0`), once inside each `tcool-<seed>` and each `static-<seed>`. Every
later arm loads `noop.pt` and checks `state_hash == noop_sha256` (`calibrate.py:128-139`), which is a
load check, not a recomputation. Receipts: `noop-3000/result.json payload.noop_sha256 =
recalibrated_sha256 = c44363f1…`; `03-hosts.json static_noop.*.recalibrated_static_sha256` and
`tcool_noop.*.noop_sha256` are copies of the same single computations. All six anchors, all six
T-cool and all six static hosts chose `R-bn`, so the baseline of every D-quantity is a once-computed
state. Reasoning: the recalibration is order-sensitive by the project's own test (commit c49a4de),
and the chunk order is fixed by construction (`seen_fit` is a `randperm` under `SEEN_FIT_SEED`,
`m0/data.py:14-26`), but nothing in the receipts pins that order (no hash of the fit indices) or
shows the pass reproduces across cards or runs. Confidence: high. Action: in the confirmatory's V0,
run one extra `noop` child for anchor 3000 on card 1 (about 3 s wall, `noop-3000 wall_seconds 2.7`)
and compare `recalibrated_sha256`; publish a sha256 of the `seen_fit` index tensor in the noop
receipt. Code change, so gate B.

**F2. Major. Environment identity omits driver version, cuDNN version, GPU model and Python version.**
Evidence: `provenance.py:17-37` pins and records only `torch_wheel`, `torch`, `cuda`, `numpy_wheel`,
`numpy`; `00-opening.json cards.gpu_query` holds index, UUID, utilization and memory only
(`pilot.py:175`); no child identity or process record names the driver or
`torch.backends.cudnn.version()`; Python 3.12.3 is inferred from `/usr/bin/python3` on this host, not
recorded. Reasoning: the attempt-4 lineage is same-hardware reproducibility (both runs on
`GPU-6198007d…` and `GPU-d5b931ca…`, per the pilot's `00-opening.json` and attempt 4's
`admission.json`), which is a legitimate claim, but a reader of the published receipts cannot name
the SKU, driver or cuDNN behind "bitwise on both cards". The pinned attempt-4 hashes act as a test
vector that would catch a driver or cuDNN change affecting the host training kernels, but not one
affecting only kernels the host path never exercises (einsum/tanh in `StaticHost.forward`, the
adapter fits, CPU LBFGS). Confidence: high. Action: record driver, cuDNN version, GPU name and Python
version in `environment_identity()` and the opening record before the confirmatory. Gate B.

**F3. Minor. `pilot-result.json` does not carry `certificate_sha256` or the custody path.**
Evidence: its keys (listed in the review's checks) contain protocol identity, commit and source hash
but no certificate field; `00-opening.json` and `attempt-custody.json` both carry
`eeee08ac…`. Reasoning: the verdict record is linkable by `implementation_commit`
and `protocol_sha256` but is not self-standing as published evidence. Confidence: high.
Action: add `certificate_sha256` and `custody` to the final record.

**F4. Minor. T-cool receipts record the no-op choice without the two losses that decided it.**
Evidence: `tcool-3000/result.json payload` has `noop_choice`, `noop_sha256`, `raw_sha256`,
`recalibrated_sha256` and no `*_selection_loss` (`worker.py:413-424 _noop_rule` returns none);
`noop-*` and `static-*` do publish both losses. Reasoning: the `<` decision (`calibrate.py:117`,
`static.py:138`) is auditable for N0 and S-static (margins 0.046 to 0.165 nats, and 0.042 to 0.140 nats) but asserted
for T-cool. Confidence: high. Action: return `raw_selection_loss` and `rbn_selection_loss` from
`_noop_rule`.

**F5. Minor. Process records carry no exit status or captured environment.**
Evidence: every `*.process.json` has exactly `aggregate_budget_remaining_seconds`, `command`, `gpu`,
`pid`, `started_unix`. Exit codes live only in the ledgers (`ledger-087.json`: 174 entries, all
`returncode 0`). Environment variables that could alter kernels (`NVIDIA_TF32_OVERRIDE`,
`PYTORCH_CUDA_ALLOC_CONF`, `OMP_NUM_THREADS`) are not captured; the determinism knobs themselves are
read back in-process (`host.py:38-49`) and identical across all 174 children. Confidence: high.
Action: add `returncode` and a whitelist of CUDA/torch-relevant environment variables to the
process record.

**F6. Minor. The CPU slicing known answer records no CPU determinism context.**
Evidence: `02-instrument.json slicing.device = "cpu (supervisor, ruling R18 b)"`,
`rank_cut_max_difference 0.0`, `random_cut_max_difference 0.3209`,
`wrong_cut_max_difference 0.3209`, `known_answer true`; no thread count or BLAS identity for the
supervisor. Reasoning: this is a known-answer test with a 1e-6 tolerance, not a replay claim, so it
is bounded; it says nothing about the GPU slicing path used by the M-wide→8 arms. Confidence:
medium (bounded by tolerance; the GPU path is covered by `deterministic_algorithms=True`, not by a
receipt). Action: none required; note the limit in the evidence narrative.

**F7. Observation. The attempt-4 lineage crosses a code change and a protocol.**
Evidence: attempt 4 ran `sultai_screen.screen train` at commit `027a258`, protocol `gate3-v4.2`,
`host.py` sha `fd8a263e…` (`replay-card0/result.json identity`); the pilot ran
`sultai_screen.m0.worker replay` at `0d4179a`, `host.py` `f4f57103…`. Final, per-epoch, loss-stream,
gradient/output and serialized `host.pt` hashes (`0ed053b2…`) are identical for 3000 and the
per-epoch, loss-stream, checkpoint and probe hashes are identical for 3001 to 3005
(`gate3-pilot-2026-10-13-linked-3/intact-<seed>/result.json` versus `replay-<seed>/result.json`).
The pinned constants `ATTEMPT4_FINAL_STATE` (`worker.py:127-134`) equal the six `intact-<seed>`
finals. Reasoning: strong evidence the training path survived the refactor; the independent source
of the pin is the attempt-4 directory, not the constant. Confidence: high.

**F8. Observation. `both_cards_bitwise: true` is a literal, and acceptance (iii) is an 8-image forward.**
Evidence: `pilot.py:429` writes `True` after a loop that raises on any `REPLAY_KEYS` difference;
`worker.py:1265-1272` compares `static(images)` with `plain(images)` on `data.train[:8]`. Reasoning:
the comparison is real (I re-did it from the two receipts), but the field is not a recorded
comparison; the stronger step-0 evidence is `initial_state_sha256` equality across all six anchors in
`03-hosts.json static_step0`, which I verified against the child receipts. The static host's
data-stream identity with the anchor (§5 line 295) is by construction (`generator(seed,
f"order:{epoch}")`, `host.py:208-210`), not receipted. Confidence: high.

**F9. Observation. The certificate names commit c49a4de; the run records 0d4179a.**
Evidence: `docs/certificates/2026-10-14-m0-pilot-admission-3.json implementation_commit =
c49a4de…`; `00-opening.json implementation_commit = 0d4179a…`; `check_certificate`
(`pilot.py:206-213`) binds by `implementation_source_sha256` equality (`a21584a3…`, matches every
child) and `git merge-base --is-ancestor`; `git diff --stat c49a4de 0d4179a -- src tests
pyproject.toml scripts` is empty. Reasoning: sound by design (merge commit, identical tree); worth
one sentence in the evidence narrative. Confidence: high.

## Question A: do the receipts establish what they claim?

**Both-card bitwise (seed 3000): yes.** `replay-3000-gpu0/result.json` (card 0, pid 1615705) and
`replay-3000-gpu1/result.json` (card 1, pid 1615706) agree on `initial_state_sha256 41d64c71…`, all
ten `epoch_state_sha256`, `final_state_sha256 f607728b…`, `loss_stream_sha256 3752ec79…`,
`forward_backward.gradient_sha256 eb05176a…`, `output_sha256 e9d44bc0…`, `checkpoint_sha256
0ed053b2…` and `optimizer_sha256 159ae50b…`. The checkpoint and optimizer hashes are file hashes of
`host.pt`/`optimizer.pt`; the manifest's `raw_unpublished` entries and the files on disk under
`runs/` carry the same values (checked for both cards). Identical serialized bytes from two
processes on two cards is stronger than a state hash alone.

**Attempt-4 lineage: yes, on the same two cards.** Traced end to end: `worker.py:128` pin
`f607728b…` = `01-replay-timing.json attempt4_final_state_sha256.replay-3000-gpu0` =
`gate3-pilot-2026-10-13-linked-3/replay-card0`, `replay-card1-a`, `replay-card1-b` and
`intact-3000` `final_state_sha256`, with identical epoch hashes, loss stream, probe hashes and
`checkpoint_sha256 0ed053b2…`. Anchors 3001 to 3005 match attempt 4's `intact-<seed>` on every
replay key (F7). Card UUIDs are identical between the two attempts (F2).

**Resume bitwise: yes, on both cards.** `acceptance-{0,1}/result.json`: a fresh 12-epoch straight run
whose epochs 0 to 9 equal the replay child's hashes (a third and fourth in-process replay of the
anchor), `resumed_prefix_state_sha256 f607728b…` loaded from `replay-3000-gpu<n>/host.pt` plus
`optimizer.pt`, two resumed epochs `a2caab3c…`, `d334c4ae…` equal to straight epochs 10 and 11;
`resume_bitwise true` on both cards with identical 12-entry straight lists. The snapshot closure is
parameters, buffers and SGD momentum (`worker.py:1250-1258`); there is no RNG state to carry because
order, crop and flip streams are re-derived per epoch from `(seed, label)` and the rate is a function
of step (`host.py:207-224`).

**Static step 0: yes, all six anchors.** `static-<seed>/result.json payload.initial_state_sha256`
equals `replay-<seed>/result.json payload.initial_state_sha256` for 3000 (`41d64c71…`), 3001
(`bb0f3e3d…`), 3002, 3003, 3004 and 3005; `03-hosts.json static_step0` records the same pairs and
`static_step0_matches true`. Acceptance (iii) `static_step0_bitwise true` on both cards (F8).

**Evaluation consistent:** `loss_max_difference 0.0` against tolerance 1e-6, accuracy equal, both
cards.

**Hashes asserted but single-sourced:** `noop_sha256`/`recalibrated_sha256` in every `noop-*`,
`tcool-*` and `static-*` receipt (F1); `tcool_final_state_sha256` (one 5-epoch resume per anchor,
one card); `static_sha256` (one training per anchor, one card); the module-fit tensors behind
`checkpoint_bitwise` (compared in-process on each card, never across cards; only `checkpoint_rate
0.1` is compared across cards).

Confidence: high. Risk if relied on as stated: low.

## Question B: leaks the receipts do not cover

| Leak | Status | Evidence | Matters for confirmatory replay claims? |
|---|---|---|---|
| Cross-card kernel choice | Ruled out for the host training path on these two cards; bounded elsewhere | 12,500 steps and 12 acceptance epochs bitwise on cards 0 and 1; other arms (noop, tcool, static, frontier, module fits) each ran once on one card | Yes for N0 (F1); the SKU is not named (F2) |
| cuBLAS/cuDNN flags | Ruled out for the flags; open for library versions | `identity.determinism` identical in all 174 children: `CUBLAS_WORKSPACE_CONFIG :4096:8`, `deterministic_algorithms true`, `cudnn_deterministic true`, `cudnn_benchmark false`, TF32 off, `threads 2`, read back from torch (`host.py:38-49`); cuDNN/driver unrecorded | Yes (F2); the pinned hashes detect drift on the host path only |
| Data-loading order | Ruled out | No DataLoader; whole dataset on device; per-epoch `randperm` under a CPU generator seeded by `derive(seed, "order:<epoch>")`; batch files sha-pinned at load (`data.py:20-27,127-129`) and audit-sealed (`data_opens` = 1 each, `test_batch` refused); `loss_stream_sha256` equal across cards and attempt 4 | No |
| BN recalibration chunk order | Deterministic by construction, never replayed | `calibrate.py:25-45`, `seen_fit` fixed permutation, order-sensitivity test c49a4de; no receipt pins the index order or a second computation | Yes (F1) |
| CPU-vs-GPU paths | Bounded, not a replay claim | Slicing known answer on CPU with 0.0 against 1e-6 (F6); R-temp/R-cal LBFGS fits run on CPU float64 (`calibrate.py:66-97`), single computation, threads pinned at 2; frontier bootstrap seeded (`bootstrap_seed 3000`, 10,000 samples) | Low; the CPU fits are reported quantities, not replay claims |
| No-op rule on recomputed features | Bounded by margin | Strict `<` on two floats recomputed from each host; observed margins 0.046 to 0.165 nats (N0) and 0.042 to 0.140 (static); T-cool margins unpublished (F4) | Low unless a confirmatory anchor lands near a tie; publish T-cool losses |
| Environment in opening/process records | Open | F2, F5; `nice -n 19` and CPU contention affect timing only | Yes for published evidence (F2) |
| One process per card | Ruled out for this run | 174 children, all `physical_gpu` equal to the process `gpu`, distinct pids, no same-card overlap of `[started_unix, started_unix + wall_seconds]` (wall from `result.json`, so process lifetime is slightly longer than the checked interval); `compute_apps` empty at opening | No |

Confidence: high on what is ruled out, medium on the bounds. Risk: medium; F1 and F2 are the open
items.

## Question C: custody and identity chain

Intact. All 174 children (every stage: replay, acceptance, timing, instrument, noop, modules, maff,
retune, rall, lottery, mexp, static, tcool, frontier, starved, cooled-20, evaluate, fold-check)
carry byte-identical identity blocks apart from `physical_gpu`: commit `0d4179a990b3…`, protocol
`m0-v4.2`, `protocol_sha256 a2c1e55d…`, preregistration `c6659945…`, amendments `af12d474…`,
`b4740b14…`, `6dcfd9f2…`, `d8dd571b…`, steering `08a9e1aa…`, `source_sha256 a21584a3…`,
environment torch `2.9.1+cu128`, CUDA 12.8, numpy 2.3.5. `00-opening.json`, `attempt-custody.json`
and `pilot-result.json` agree on every shared field; the certificate sha `eeee08ac…` in opening and
custody equals `sha256sum docs/certificates/2026-10-14-m0-pilot-admission-3.json`. I recomputed
`protocol_sha256` from `identity.py:29-31` and the six document hashes (all six documents on disk
hash to their recorded values) and got `a2c1e55d…`. The four key source files hash identically at
`0d4179a` and at HEAD. `.weft/m0-pilot-attempt.json` (gitignored, on disk) names this output root,
pid 1615228, `started_unix 1791958328.246`, 3 ms before the first child; `runs/` holds exactly one
M0 root. The manifest's 449 published files all match their recorded sha256 and no unlisted file
exists. Gaps: F3 (certificate absent from the verdict record), F9 (certificate commit differs from
the run commit, bound by source hash and ancestry).

Confidence: high. Risk: low.

## Question D: what must be fixed before a published confirmatory

Nothing blocks the pilot's V0 verdict as stated. Before a confirmatory whose receipts are published
as evidence, fix F1 (one cross-card replay of the R-bn recalibration and a pinned fit-index hash)
and F2 (driver, cuDNN, GPU model, Python in the identity). F3 to F5 are cheap receipt additions
worth bundling into the same gate-B change. These are recommendations; local evidence supports
them, and external acceptance is John's at gate B.

Confidence: high. Risk: low; F1 and F2 are recommendations pending gate B.

## What I did not check

- Did not run any child, any GPU work, or any test; no hash was recomputed from weights (only from
  files and receipts). Did not open CIFAR batch files.
- Did not audit the instrument half beyond V0 (P1 materiality, M-aff fold on anchors other than the
  fold-check, frontier slope, admission arithmetic, stages 4 to 7).
- Did not verify the `seen_fit` permutation, `cached()`/`EVAL_CHUNK` evaluation chunking, or the
  adapter fitter (`fit.py`) for RNG ownership; `checkpoint_bitwise true` on both cards is taken as
  the in-process evidence it is.
- Did not confirm the two cards are the same SKU or the driver/cuDNN versions in force (not
  recorded anywhere; F2).
- Did not inspect `tests/test_m0_host.py` or `tests/test_v42_ladder.py` for the recalibration-order
  fixture beyond the c49a4de commit message.
- Did not read ledgers 002 to 086 individually; the final ledger covers all 174 jobs with
  `returncode 0`, and the ledgers are cumulative.
