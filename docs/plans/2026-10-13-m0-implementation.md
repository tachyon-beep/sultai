# M0 capacity frontier screen: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the code the frozen M0 preregistration names in §2 so that its pilot (§9) and confirmatory (§10) can run on Nyx under their own identity, ceilings and receipts, with the Gate 3 code path left behaviourally untouched.

**Architecture:** A new subpackage `src/sultai_screen/m0/` owns everything M0-specific (identity, seeds, arms, instrument, statistics, verdicts, workers, pilot, confirmatory, costs). The shared modules `host.py`, `fit.py`, `data.py` and the `Supervisor` in `pilot.py` gain additive keyword parameters whose defaults reproduce today's behaviour bitwise; a golden CPU test is pinned before each shared function is touched. Gate 3 workers, tests, constants and custody receipts are not deleted or edited; `former_bridge` and `alignment` are never imported by M0 code.

**Tech Stack:** Python 3.12 on Nyx (`/usr/bin/python3`), torch 2.9.1+cu128, numpy 2.3.5, `unittest`, mypy 1.19.1 strict, ruff 0.15.4, the Sultai trust policy (`scripts/check_policy.py`).

**Spec:** `docs/plans/2026-10-13-capacity-frontier-screen.md` v4, frozen 2026-10-13 (SHA-256 `c6659945…`), with `docs/STEERING.md` (`08a9e1aa…`), `docs/plans/2026-10-13-gate-ladder-amendment.md` (`af12d474…`) and `docs/plans/2026-10-13-gate-ladder-amendment-2.md` (`b4740b14…`); composed identity `m0-v4` = `3a76c6dbe31ca3f80ec301ed0523b1f382148f8114ee02a164ae4ae9557f6e5d` ([freeze certificate 2](../certificates/2026-10-13-m0-freeze-2.md); v3 and its `m0-v1` identity remain on record in certificate 1). This plan does not edit any of the three pinned files. Where the draft says "stated in the implementation plan" (seen-fit derivation, per-lineage seed blocks, the 0.5-host anchor count, the decay granularity of T-cool and the 20-epoch point), the statement is here and marked **[plan]**. A draft defect found while implementing is raised to John as amendment 2, never absorbed.

**Gates (John, 2026-10-13, "bundle these approvals into bigger gates"):** gate A, John accepts this plan (then it merges and implementation proceeds without further asks); gate B, the code PR is reviewed and ready (then it merges and the pilot go is requested); gate C, the pilot results are read (confirmatory go). GPU runs stay explicit gos. No code before gate A.

## Global Constraints

- Test command: `PYTHONPATH=src /usr/bin/python3 -B -m unittest discover -s tests`; CPU only; the 168 existing tests stay green at every commit. A single test: `PYTHONPATH=src:tests /usr/bin/python3 -B -m unittest <module>.<Class>.<test> -v`.
- Trust policy (`scripts/check_policy.py`): no `Any`, `cast`, `.get`, `type: ignore`, `noqa`; every public function with an `object`-typed parameter (anywhere in an annotation, string annotations included) or named `parse_*`, `from_json` or `from_dict` carries exactly one `@t3_boundary(test="tests/<file>.py::<Class>.<test>", fingerprint="<sha256 of the test's AST>")`; recompute the fingerprint whenever the paired test changes (`test_fingerprint` in the script).
- `pyproject.toml`: mypy strict with `disallow_any_explicit`, `warn_unreachable`; ruff `E,F,W,I,UP,B,C4,RUF`, line length 120; `ruff format --check` on `src` and `tests`.
- Quality gate: `scripts/check_quality.py --ruff <path>` (project has no ruff install; the Nyx CI copy under `/tmp/nyx-ci-981ef46-py312/bin/ruff` was used at 948f522).
- Data seal: the CIFAR test batch is never opened; `install_seal()` stays on every worker path; weights, optimiser states and raw logs stay under ignored `runs/` or `/tmp`.
- `PILOT_SECONDS` (10,800) is untouched; Gate 3 ledgers are checked against it (`pilot.py:514,517,569`). M0 ceilings are new constants: `M0_PILOT_SECONDS = 14_400`, `M0_CONFIRMATORY_SECONDS = 43_200`.
- Margins frozen by the draft, copied verbatim: `X = 0.02`, `X2 = 0.01`, `M_S = 0.02`, `RHO = 0.8`, `HARM = 0.01`, `FOLD_TOLERANCE = 0.005`, `CERTIFIED_FRACTION = 0.75`, `S_STATIC_SD_LIMIT = 0.0275`.
- Grids frozen by the draft: module rates `(1.0, 0.3, 0.1)`; M-aff and re-tune rates `(1.0, 0.3, 0.1, 0.03, 0.01)`; one run per rate with checkpoints `(200, 800, 3200)`; M-small-B `(400, 1600, 6400)`; minibatch 256; init N(0, 1e-3); R-all 800 steps, checkpoints every 40.
- Data: fits on 2,000 seen examples (Task 5); selection on the existing 3,000; endpoint the existing 5,000 development examples; the 2,000 held-out conditioning examples only for P1 and M-sq-fresh.
- Every §8 kernel is one process per card; the M0 supervisor admits one process per card.
- Hosts: the s = 1.0 host must keep layer order, parameter and buffer names, dtypes, shapes and RNG consumption bitwise identical to today's `Host(0)`; the attempt-4 `final_state_sha256` strings are pinned in source and replayed on GPU in pilot V0 only.
- Gate 3 tests that stay byte-for-byte untouched: all of `tests/test_v42_ladder.py` and, in `tests/test_screen.py`, `test_impairment_exact_inverse_and_identity`, `test_hungarian_and_two_sided_transport`, `test_complete_projection_counts_all_frozen_arms`, `test_all_frozen_state_and_zero_site_controls`, `test_magnitude_ladder_exhaustion_never_uses_failed_rung`, `test_ordered_pilot_holds_before_median_and_on_nonfinite_timing`, `test_raw_cheap_nonfinite_dev_never_changes_selection`, `test_global_p6_rate_retains_unavailable_p5_fit_without_fallback`, `test_oracle_retains_finite_schedule_and_all_failed_evidence` (its fake `fit_sgd` accepts only `seed, rate, steps, minibatch`, so `oracle()`'s call must not change).
- Import isolation: no module under `m0/` except `m0/pilot.py` and `m0/confirmatory.py` (supervisor processes) imports `sultai_screen.screen` or `sultai_screen.pilot`; `former_bridge` and `alignment` are never on a worker's import path.
- Git: branch `nyx/m0-implementation` from `main`; one commit per task; merge commit only, branch kept.

## Review Focus

1. **BatchNorm recalibration against the freeze hook.** `freeze()` installs a permanent pre-hook that raises if a BatchNorm enters train mode, and `load_host` always freezes. R-bn must load unfrozen, recalibrate, then freeze; a recalibration attempted on a frozen host must raise, not silently keep stale statistics. No frozen predicate exists today (`freeze` keeps no hook handle), so `freeze()` gains a `host.sealed = True` flag (initialised `False` in `Host.__init__`, a plain attribute outside `state_hash`). The guard hook fires only on a forward in train mode, so tests must run a forward to observe it. Pinned in Tasks 4 and 9.
2. **Checkpointed single run equals separate runs.** The draft claims the 200/800/3,200-step snapshots of one run are bitwise the same as three separate runs under a seed-derived stream. This holds only if the selection evaluation at a checkpoint consumes no RNG and the data order stream is identical. Pinned in Task 6.
3. **Fold residual on the border ring.** The bias fold over-adds on the zero-padded border; the residual must be measured on real positions, not asserted zero, and V0 must refuse to open V1 above 0.005 nats. Pinned in Tasks 8 and 14.
4. **S-static host initialisation.** The static module must be created with zero tensors and no RNG draw, after the host, so the host's tensors at step 0 are bitwise the anchor's. Pinned in Task 11.
5. **Ceiling default binding in the shared supervisor.** The existing test patches the module global `PILOT_SECONDS`; a default argument bound at definition time would silently break that test and the Gate 3 ledger check. The ceiling default must resolve at call time. Pinned in Task 15.

## File structure

| Path | Responsibility |
|---|---|
| `src/sultai_screen/m0/__init__.py` | docstring only |
| `src/sultai_screen/m0/identity.py` | pinned document hashes, `m0-v1` composition, `protocol_identity`, `source_identity`, margins and ceilings |
| `src/sultai_screen/m0/seeds.py` | Appendix A registry and admission by lineage |
| `src/sultai_screen/m0/data.py` | seen-fit sub-partition `m0-seen-fit-v1` |
| `src/sultai_screen/m0/provenance.py` | environment identity, determinism flags and the receipt envelope, reimplemented without importing `screen.py` |
| `src/sultai_screen/m0/modules.py` | module arms as adapters: widths, M-exp, slicing, lottery survivors |
| `src/sultai_screen/m0/fold.py` | M-aff fold into `tail[0]`, residual |
| `src/sultai_screen/m0/calibrate.py` | R-bn, R-temp, R-cal, the N0 rule |
| `src/sultai_screen/m0/retune.py` | declared parameter subsets, complement hash, R-272s, R-aff, R-all fits |
| `src/sultai_screen/m0/static.py` | S-static training (module present from step 0) |
| `src/sultai_screen/m0/instrument.py` | P1-only materiality, post-hoc and check |
| `src/sultai_screen/m0/stats.py` | per-anchor quantities, anchor bootstrap, sign test, sizing |
| `src/sultai_screen/m0/verdicts.py` | V0, V1, V2 and labels as pure functions |
| `src/sultai_screen/m0/costs.py` | one-per-card projection, extended admission, confirmatory projection |
| `src/sultai_screen/m0/worker.py` | M0 child actions and CLI with seed admission |
| `src/sultai_screen/m0/pilot.py` | §9 stages, receipts, custody, admission |
| `src/sultai_screen/m0/confirmatory.py` | §10 launcher with the 12 GPU-hour supervisor |
| `src/sultai_screen/host.py` | additive: `widths`, `train_epochs`, schedules, optimiser state, unfrozen load |
| `src/sultai_screen/fit.py` | additive: `logits`, `accuracy`, generalised `fit_sgd`, parameterised plateau |
| `src/sultai_screen/pilot.py` | additive: `Supervisor(ceiling=, per_card=, module=)`, `Job.extra` |
| `tests/test_m0_*.py` | one test module per task group, listed per task |

---

### Task 1: M0 identity and constants

**Files:**
- Create: `src/sultai_screen/m0/__init__.py`, `src/sultai_screen/m0/identity.py`
- Test: `tests/test_m0_identity.py`

**Interfaces:**
- Produces: `PREREG_PATH`, `STEERING_PATH`, `AMENDMENT_PATH`, `AMENDMENT_2_PATH` (repo-relative `str`); `PREREG_SHA`, `STEERING_SHA`, `AMENDMENT_SHA`, `AMENDMENT_2_SHA`, `PROTOCOL_VERSION = "m0-v4"`, `PROTOCOL_SHA` (`str`); `X`, `X2`, `M_S`, `RHO`, `HARM`, `FOLD_TOLERANCE`, `CERTIFIED_FRACTION`, `S_STATIC_SD_LIMIT` (`float`); `M0_PILOT_SECONDS`, `M0_CONFIRMATORY_SECONDS` (`int`); `MODULE_RATES`, `RETUNE_RATES`, `CHECKPOINTS`, `CHECKPOINTS_B` (`tuple`); `protocol_identity() -> Record` with keys `protocol_version`, `protocol_sha256`, `preregistration_sha256`, `steering_sha256`, `amendment_sha256`, `amendment_2_sha256`; `source_identity() -> Record` (commit, `source_sha256`, `source_files`, the five identity fields), raising `PilotHold` naming the document whose bytes differ.

- [ ] **Step 1: Write the failing tests**

```python
class IdentityTests(unittest.TestCase):
    def test_four_documents_and_composed_identity_are_pinned(self) -> None:
        self.assertEqual(file_sha(REPO / PREREG_PATH), PREREG_SHA)
        self.assertEqual(file_sha(REPO / STEERING_PATH), STEERING_SHA)
        self.assertEqual(file_sha(REPO / AMENDMENT_PATH), AMENDMENT_SHA)
        self.assertEqual(file_sha(REPO / AMENDMENT_2_PATH), AMENDMENT_2_SHA)
        composed = hashlib.sha256(f"m0-v4\n{PREREG_SHA}\n{STEERING_SHA}\n{AMENDMENT_SHA}\n{AMENDMENT_2_SHA}\n".encode()).hexdigest()
        self.assertEqual(PROTOCOL_SHA, composed)
        self.assertEqual(PROTOCOL_SHA, "3a76c6dbe31ca3f80ec301ed0523b1f382148f8114ee02a164ae4ae9557f6e5d")
        self.assertEqual(set(protocol_identity()), {"protocol_version", "protocol_sha256", "preregistration_sha256", "steering_sha256", "amendment_sha256", "amendment_2_sha256"})

    def test_margins_and_ceilings_are_the_frozen_values(self) -> None:
        self.assertEqual((X, X2, M_S, RHO, HARM, FOLD_TOLERANCE), (0.02, 0.01, 0.02, 0.8, 0.01, 0.005))
        self.assertEqual((M0_PILOT_SECONDS, M0_CONFIRMATORY_SECONDS), (14_400, 43_200))
        self.assertEqual(PILOT_SECONDS, 3 * 3600)  # Gate 3 constant untouched

    def test_source_identity_names_the_changed_document(self) -> None:
        # fixture repo as in tests/test_protocol_identity.py::protocol_fixture, with the three M0 documents copied;
        # append one byte to each in turn and assert PilotHold matches "preregistration" / "steering" / "amendment 1" / "amendment 2".
```

- [ ] **Step 2: Run them to verify they fail** (`ModuleNotFoundError: sultai_screen.m0`).
- [ ] **Step 3: Implement `identity.py`.** Hashes from freeze certificate 2; `source_identity()` reimplements the git clean-tree and tracked-file aggregate of `screen.source_identity` (same `git ls-files src scripts tests pyproject.toml` set, same aggregate rule) without importing `sultai_screen.screen`, and pins the four M0 documents, never the Gate 3 chain. Add `m0/provenance.py`: `environment_identity()` with the same expected versions as `screen.py:97-116` (torch wheel 2.9.1, torch 2.9.1+cu128, CUDA 12.8, numpy 2.3.5), `provenance()` adding `host.flags()` and `physical_gpu`, and `envelope(action, identity, payload, seal, started, peak) -> Record` reproducing the `screen.worker` receipt keys; test that `import sultai_screen.m0.identity, sultai_screen.m0.provenance` in a subprocess leaves `sultai_screen.screen`, `sultai_screen.pilot`, `sultai_screen.former_bridge`, `sultai_screen.alignment` out of `sys.modules`.
- [ ] **Step 4: Run the tests and the whole suite; Gate 3's `test_protocol_identity` must be unchanged and green.**
- [ ] **Step 5: Commit** `M0: identity constants pinned to the freeze certificate`.

### Task 2: Seed registry and admission

**Files:**
- Create: `src/sultai_screen/m0/seeds.py`
- Test: `tests/test_m0_identity.py`

**Interfaces:**
- Produces: `Lineage` (a `Literal` of `"anchor"`, `"frontier-0.25"`, `"frontier-0.5"`, `"frontier-0.75"`, `"frontier-1.5"`, `"cooled-20"`, `"confirmatory"`, `"restart"`); `BLOCKS: dict[Lineage, range]` = anchor `range(3000, 3006)`, frontier-0.5 `range(6000, 6006)`, frontier-0.75 `range(6006, 6012)`, frontier-1.5 `range(6012, 6018)`, cooled-20 `range(6018, 6024)`, frontier-0.25 `range(6024, 6030)`, confirmatory `range(7500, 7600)`, restart `range(7700, 7800)`; `RESERVED_M2 = range(6100, 6200)`; `admit(seed: int, lineage: Lineage) -> int` raising `PilotHold` otherwise; `width_scale(lineage) -> float`.

- [ ] **Step 1: Write the failing tests:** every block matches Appendix A; blocks are pairwise disjoint and disjoint from `RESERVED_M2` and from the Gate 3 blocks `range(4000, 4364)`, `range(5000, 5800)`; `admit(3000, "frontier-0.5")` raises; `admit(6100, "anchor")` raises (reserved); `admit(7500, "confirmatory") == 7500`.
- [ ] **Step 2: Run, verify failure.** **Step 3: Implement.** **Step 4: Run, pass.** **Step 5: Commit** `M0: seed registry and admission`.

### Task 3: Width-tuple host constructor

**Files:**
- Modify: `src/sultai_screen/host.py:56-72` (`Host.__init__`), `host.py:81-86` (`make_host`)
- Test: `tests/test_m0_host.py`

**Interfaces:**
- Produces: `Host(width: int = 0, widths: tuple[int, int, int, int] = (64, 64, 128, 128))`; `make_host(seed, width, device, *, widths=(64, 64, 128, 128))`; `FRONTIER_WIDTHS: dict[float, tuple[int, int, int, int]]` in `m0/modules.py` later (Task 7) but the counts are pinned here; `parameter_count(host) -> int`.

- [ ] **Step 1: Golden test first, before editing `host.py`.** Build `make_host(3000, 0, cpu)` and `make_host(3000, 4, cpu)` with today's code, record `state_hash` strings and parameter counts, and write them into the test as literals:

```python
def test_reference_host_hash_and_count_are_unchanged(self) -> None:
    host = make_host(3000, 0, torch.device("cpu"))
    self.assertEqual(state_hash(host), "<golden sha from today's code>")
    self.assertEqual(sum(p.numel() for p in host.parameters()), 284_634)
    self.assertEqual([n for n, _ in host.named_parameters()], [<today's name list>])
```

Run it against the unedited code: it must pass (this is the one test in the plan written to pass first; it is a golden pin, not TDD of new behaviour).

- [ ] **Step 2: Write the failing tests for the tuple:**

```python
def test_frontier_counts_match_the_frozen_table(self) -> None:
    for s, widths, count in ((0.25, (16, 16, 32, 32), 33_978), (0.5, (32, 32, 64, 64), 85_274),
                             (0.75, (48, 48, 96, 96), 168_826), (1.0, (64, 64, 128, 128), 284_634),
                             (1.5, (96, 96, 192, 192), 613_018)):
        host = Host(0, widths=widths)
        self.assertEqual(sum(p.numel() for p in host.parameters()), count)
        self.assertEqual(sum(p.numel() for p in host.front.parameters()), 14_928)
def test_explicit_default_tuple_is_bitwise_the_reference(self) -> None:
    a, b = make_host(3000, 0, cpu), make_host(3000, 0, cpu, widths=(64, 64, 128, 128))
    self.assertEqual(state_hash(a), state_hash(b))
def test_bottleneck_and_tuple_compose(self) -> None:  # Host(4, widths=(32,32,64,64)) builds and classifies a (2,3,32,32) batch
```

- [ ] **Step 3: Run; the tuple tests fail with `TypeError: unexpected keyword 'widths'`.**
- [ ] **Step 4: Implement.** Replace the four literals in `self.tail` by `widths`; `self.classifier = nn.Linear(widths[3], 10)`; store `self.widths`. Validate each width is a positive int. Nothing else moves.
- [ ] **Step 5: Run all of `test_m0_host.py` and the suite; the golden test still passes.**
- [ ] **Step 6: Commit** `host: explicit tail-width tuple, reference host bitwise unchanged`.

### Task 4: Trainer with schedules and optimiser state

**Files:**
- Modify: `src/sultai_screen/host.py:129-177` (`train_host`), add `train_epochs`, `Schedule`, `optimizer_state` boundary, `load_host(..., frozen=)`
- Test: `tests/test_m0_host.py`

**Interfaces:**
- Produces: `train_epochs(model: nn.Module, hashed: Host, data: Dataset, seed: int, *, epochs: range, rates: Callable[[int], float], optimizer: torch.optim.SGD) -> tuple[list[str], Tensor, float]` where the forward is `model(normalized(batch))`, the gradient set is every parameter of every `optimizer` param group (so a wrapper's extra tensors train when they are in a group), `hashed` is the `Host` whose `state_hash` is recorded per epoch, and `rates(step)` gives the learning rate at global optimiser step `step` (step counts from 0 at epoch 0; 1,250 steps per epoch on the 40,000-example partition at batch 32); `recipe_optimizer(parameters: Iterable[nn.Parameter]) -> torch.optim.SGD` building the two groups (`ndim > 1` weight decay 5e-4, else 0; lr 0.05, momentum 0.9, nesterov) exactly as `train_host` does today; `train_host(host, data, seed, *, epochs=10)` becomes `train_epochs(host, host, data, seed, epochs=range(10), rates=constant(0.05), optimizer=recipe_optimizer(host.parameters()))` and keeps its `ContractViolation` on `epochs != 10` and its return triple; `constant(rate) -> Callable`; `linear_cooldown(start_step: int, stop_step: int, rate: float = 0.05) -> Callable` returning `rate` before `start_step`, `rate * (1 - (step - start_step) / (stop_step - start_step))` on `[start_step, stop_step)` and `0.0` at or after `stop_step`; `TCOOL = linear_cooldown(12_500, 18_750)` **[plan: decay per optimiser step over 6,250 steps, epochs 10–14; the last executed update uses 0.05/6,250 and zero is the boundary value after training]**; `COOLED_20 = linear_cooldown(18_750, 25_000)` **[plan: epochs 0–14 at 0.05, epochs 15–19 decayed per step]**; momentum persistence as a flat mapping: `momentum_buffers(optimizer, host) -> dict[str, Tensor]` (parameter name → `momentum_buffer`, every one of the 30 parameters present after the first step), saved with `torch.save` to `optimizer.pt`; `@t3_boundary optimizer_state(value: object, expected: dict[str, Tensor]) -> dict[str, Tensor]` validating a `dict` whose keys are exactly the host's parameter names with finite tensors of the matching shape and dtype (the `checkpoint_state` pattern, `host.py:184-194`); `restore_optimizer(host, buffers: dict[str, Tensor]) -> torch.optim.SGD` = `recipe_optimizer` with every buffer installed before the first resumed step; `freeze()` sets `host.sealed = True` (`Host.__init__` sets `False`); `load_host(path, seed, width, device, *, widths=..., frozen: bool = True)`.

- [ ] **Step 1: Golden test first.** With today's `train_host`, train `Host(0)` on a CPU fixture `Dataset` of 64 images (random uint8, labels, `train = arange(64)`, other partitions `arange(8)`) for 10 epochs; pin `hashes[-1]` and `stream` SHA as literals. Run it: passes on today's code.
- [ ] **Step 2: Write the failing tests:**

```python
def test_train_host_is_train_epochs_with_constant_rate(self) -> None:  # same fixture, same pinned hashes
def test_linear_cooldown_values(self) -> None:
    self.assertEqual(TCOOL(12_499), 0.05); self.assertAlmostEqual(TCOOL(12_500 + 3_125), 0.025); self.assertAlmostEqual(TCOOL(18_749), 0.05 / 6_250)
def test_resume_from_saved_momentum_continues_the_stream_bitwise(self) -> None:
    # train epochs 0-1 in one call; train epoch 0, save momentum_buffers (30 entries), restore_optimizer, train epoch 1;
    # final state_hash and concatenated loss streams equal bitwise; dropping one buffer before restore changes the hash.
def test_optimizer_state_boundary_rejects_missing_keys_wrong_shape_and_nonfinite(self) -> None:
def test_linear_cooldown_is_zero_at_and_after_stop(self) -> None:
def test_load_host_unfrozen_allows_a_train_mode_forward_and_frozen_raises_on_it(self) -> None:
    # frozen=False: host.sealed is False, a forward in train mode runs; frozen=True: host.sealed is True and the same forward raises PilotHold
```

- [ ] **Step 3: Run; fail on missing names.**
- [ ] **Step 4: Implement.** Factor the loop body of `train_host` into `train_epochs`; set `group["lr"] = rates(step)` before each `optimizer.step()`; RNG labels stay `order:{epoch}`, `crop:{epoch}`, `flip:{epoch}` with the global epoch index so a resume at epoch 10 draws `order:10`. `train_host` must stay byte-identical in name, signature and return triple (`screen.py:47` imports it; no test patches it).
- [ ] **Step 5: Run; the golden test still passes; suite green.**
- [ ] **Step 6: Commit** `host: train_epochs with per-step schedules and resumable optimiser state`.

### Task 5: Seen-fit sub-partition and accuracy evaluator

**Files:**
- Create: `src/sultai_screen/m0/data.py`
- Modify: `src/sultai_screen/fit.py:55-79` (add `logits`, `accuracy` beside `losses`)
- Test: `tests/test_m0_fit.py`

**Interfaces:**
- Produces: `SEEN_FIT_SHA: str`; `seen_fit(data: Dataset) -> Tensor` **[plan: `data.train[torch.randperm(40_000, generator=generator(20261013, "m0-seen-fit-v1"))[:2000]]`, int64 indices into the image tensor; the SHA pins the little-endian int64 bytes]**; `features_from_images(host, data, indices) -> Features` (front and bottleneck outputs for arbitrary indices, mirroring `cached.part`); `logits(host, data: Features, adapter: Adapter | None) -> Tensor` `[N, 10]` through the same padded 128-row chunks as `losses` (a sibling loop; `losses` keeps computing cross-entropy inside each padded chunk exactly as today, `fit.py:64-70`, and is not rewritten to call `logits`); `accuracy(host, data, adapter) -> Tensor` `[N]` float 0/1 in caller order; `mean_accuracy(...) -> float`.

- [ ] **Step 1: Write the failing tests:** `seen_fit` has 2,000 unique indices, all in `data.train`, none in conditioning/selection/development, SHA equals `SEEN_FIT_SHA` (compute once from the derivation, pin the literal); `accuracy` equals `(logits.argmax(1) == labels)` computed by an independent unpadded forward on a 129-row fixture (crosses a chunk boundary); `losses` output bytes unchanged by the addition (compare to a pinned value on the fixture).
- [ ] **Step 2: Run; fail.** **Step 3: Implement.** `logits` duplicates the padded chunk loop; `losses` is untouched so the existing `test_fixed_evaluation` pins stay bitwise and the chunk-shaped cross-entropy is preserved. **Step 4: Run; pass.** **Step 5: Commit** `M0: seen-fit partition and padded accuracy evaluator`.

### Task 6: Generalised fitter and parameterised plateau

**Files:**
- Modify: `src/sultai_screen/fit.py:20-37` (`Adapter`), `:159-228` (`fit_sgd`), `:279-304` (`oracle`)
- Test: `tests/test_m0_fit.py`

**Interfaces:**
- Produces: `Adapter` accepts 2, 3 or 4 tensors: 3 = M-exp `(w1 [16,64], w2 [64,16], b)` applying `base + w1 @ (w2 @ tanh(read)) + b` **[plan]**; `Adapter.count() -> int`; `initial_adapter(seed, *, hidden: int | None, nonlinear: bool, expanded: bool, device) -> Adapter` drawing N(0, 1e-3) from `generator(seed, "oracle-init")` in a fixed tensor order; `fit_sgd(host, conditioning, selection, *, seed, rate, steps, minibatch, wide=False, start: Adapter | None = None, hidden: int | None = None, nonlinear: bool = True, expanded: bool = False, checkpoints: tuple[int, ...] = ()) -> Candidate` with `Candidate.snapshots: tuple[tuple[int, Adapter, float | None], ...]` (step, detached adapter, selection loss) for each checkpoint, and `Candidate.step0_selection: float | None`; `grid(host, conditioning, selection, seed, *, rates, checkpoints, **kind) -> Grid` with `Grid.best: Candidate`, `Grid.edge_of_grid: bool`, `Grid.plateau: bool`, `Grid.plateau_evidence: str`, `Grid.report()`; `plateau(early: Candidate, late: Candidate, selection: Features, seed) -> tuple[bool, float, float]` applying the v4 Table B rule to any two schedules (last two 40-step windows of `late`).
- Gate 3 `oracle()` keeps its signature, its outputs and **its `fit_sgd` call bytes**: every new `fit_sgd` keyword is a default that `oracle()` never passes (the fake in `tests/test_screen.py:822` accepts only `seed, rate, steps, minibatch`).

- [ ] **Step 1: Golden pin.** On a CPU fixture, pin `fit_sgd(... rate=0.3, steps=4, minibatch=True)` adapter tensor SHAs with today's code; passes today.
- [ ] **Step 2: Write the failing tests:**

```python
def test_checkpointed_run_is_bitwise_three_separate_runs(self) -> None:
    one = fit_sgd(..., steps=12, minibatch=True, checkpoints=(4, 8, 12))
    for step, adapter, _ in one.snapshots:
        separate = fit_sgd(..., steps=step, minibatch=True)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(adapter.weights, separate.adapter.weights)))
def test_affine_fit_updates_without_tanh(self) -> None:  # nonlinear=False; gradient equals the affine closed form on one step
def test_hidden_width_and_expanded_shapes_and_counts(self) -> None:  # hidden=8 -> 280; hidden=64 -> 2128; expanded -> 16*64+64*16+16; two-tensor -> 272
def test_warm_start_reports_step0_selection(self) -> None:
def test_edge_of_grid_flag_and_plateau_rule(self) -> None:  # best at the top rate flags True; plateau on (800, 3200) windows 3120-3200
def test_oracle_outputs_unchanged(self) -> None:  # pinned from today's code on the fixture; and oracle() still runs under the test_screen fake fit_sgd
```

- [ ] **Step 3: Run; fail.** **Step 4: Implement.** Snapshots are taken after the step whose 1-based count equals the checkpoint; the selection evaluation uses `losses()`, which draws no RNG. **Step 5: Run; pass; suite green.** **Step 6: Commit** `fit: checkpointed grids, affine and expanded adapters, warm start, parameterised plateau`.

### Task 7: Module arms, slicing and lottery

**Files:**
- Create: `src/sultai_screen/m0/modules.py`
- Test: `tests/test_m0_modules.py`

**Interfaces:**
- Produces: `FRONTIER_WIDTHS` (Task 3 table); `contribution(adapter: Adapter, read: Tensor) -> Tensor` `[h]` = `‖V[:, i]‖₂ · RMS over examples and positions of tanh(U z + c)_i`; `activation_means(adapter, read) -> Tensor` `[h]` = mean over the fit examples and positions of `tanh(U z + c)_i`; `cut(adapter, keep: Tensor, means: Tensor) -> Adapter` slicing `U, c, V` to the kept rows/columns and folding `d ← d + Σ_removed V[:, i] · means[i]`; `cut_unfolded(adapter, keep)`; `rank_cut(adapter, read, h) -> Cut` and `random_cut(adapter, read, h, seed) -> Cut` (`generator(seed, "random-cut")`), both computing `means` from the same fit `read` and both returning `Cut(folded: Adapter, unfolded: Adapter, survivors: tuple[int, ...])`; ties in `contribution` broken by the lower unit index (`torch.argsort(..., stable=True)` on the negated contribution); `leave_one_out(host, data, adapter) -> Tensor` `[h]` loss increase per removed unit, with `ablation_cut(adapter, read, h, ablation) -> Cut` keeping the `h` units whose removal costs most; `contract(expanded: Adapter) -> Adapter` (`w1 @ w2`, exact); `lottery_start(seed, keep) -> Adapter` (the M-wide initial tensors regenerated from `initial_adapter(seed, hidden=64)` and sliced to `keep`, biases included).

- [ ] **Step 1: Write the failing tests:** the §7 known answer (a 64-unit adapter with 56 columns of `V` exactly zero: after `rank_cut` to 8 with fold, output within 1e-6 of the original on random `read`; `random_cut` to 8 differs; a negative control that keeps the wrong 8 differs); a **nonzero-removed-unit fixture**: a 4-unit adapter cut to 2 on fixed `read`, where the folded bias equals `d + Σ_removed V[:, i] · mean(a_i)` computed independently in `float64` within 1e-6 and `cut_unfolded` omits that term; `survivors` recorded for every cut and ties resolved to the lower index; printed counts `33h + 16` for h ∈ {16, 8, 4, 2} = 544, 280, 148, 82; `contract` reproduces the expanded output within 1e-6; `lottery_start` equals the sliced initial tensors.
- [ ] **Step 2: Run; fail.** **Step 3: Implement.** **Step 4: Run; pass.** **Step 5: Commit** `M0: module arms, contribution slicing with mean fold, lottery start`.

### Task 8: M-aff fold into `tail[0]` and the residual

**Files:**
- Create: `src/sultai_screen/m0/fold.py`
- Test: `tests/test_m0_modules.py`

**Interfaces:**
- Produces: `fold_affine(host: Host, adapter: Adapter) -> Host` (a fresh host with the same state except `tail[0].weight` and `tail[0].bias`), frozen; `fold_residual(host, adapter, folded, development: Features, images_development: Tensor) -> float` = mean dev loss of `folded` evaluated from images minus mean dev loss of `host + adapter` on the cached site (the same fitted `adapter` that produced `folded`; both terms recomputed inside the function, never passed in).

Formula **[spec, written out]**: with `u = z + W z + b` and `tail[0]` weight `K [64, 16, 3, 3]`, bias `c [64]`:
`K′[o, i, :, :] = Σ_j K[o, j, :, :] · (I + W)[j, i]` (`torch.einsum("ojhw,ji->oihw", K, I + W)`), `c′[o] = c[o] + Σ_j Σ_{h,w} K[o, j, h, w] · b[j]`; exact where every tap sees a real pixel, over-adding on the one-pixel border ring.

- [ ] **Step 1: Write the failing tests:** on a CPU `Host(0)` and random `z [4, 16, 16, 16]`: interior positions of `folded.tail[0](z)` versus `host.tail[0](u)` agree within 1e-6 (`float32`); some border positions differ when `b ≠ 0`; the `b = 0` fold is exact everywhere; the folded host's `state_hash` differs from the original only in `tail.0.weight` and `tail.0.bias`; `fold_residual` is a finite float, is 0.0 bitwise when `W = 0, b = 0`, and is nonzero on a `b ≠ 0` fixture.
- [ ] **Step 2: Run; fail.** **Step 3: Implement.** **Step 4: Run; pass.** **Step 5: Commit** `M0: exact-interior fold of the affine module and its residual`.

### Task 9: Calibration arms and the published no-op

**Files:**
- Create: `src/sultai_screen/m0/calibrate.py`
- Test: `tests/test_m0_modules.py`

**Interfaces:**
- Produces: `recalibrate_bn(host: Host, images: Tensor, indices: Tensor) -> Host` (requires `host.sealed is False`, else `PilotHold`; sets every `BatchNorm2d` to `momentum=None`, resets running statistics, one no-grad pass in train mode over the fit examples in `indices` order in chunks of 128 with the partial last chunk included **[plan: cumulative statistics, fixed chunk order]**, then `freeze`); `temperature(logits, labels) -> float` and `class_bias(logits, labels) -> tuple[float, Tensor]` fitted on the fit examples by LBFGS on CPU `float64`; `apply_calibration(logits, t, bias) -> Tensor`; `no_op(raw: Host, recalibrated: Host, data: Dataset) -> tuple[Host, str]` evaluating each candidate on **features recomputed from that candidate** (`cached(candidate, data).selection`, since R-bn changes the front's BatchNorm buffers and invalidates the raw host's cached site) and returning the host with the lower selection loss and `"R-bn"` or `"raw"`; `save_noop(host, output) -> str` writing `noop.pt` and returning its `state_hash`, which every later arm on that anchor loads (`load_noop(root, expected_hash, device)`), so no arm is fitted on the raw host after the choice.

- [ ] **Step 1: Write the failing tests:** `recalibrate_bn` on a `load_host(..., frozen=True)` host raises `PilotHold` before any forward; on `load_host(..., frozen=False)` it changes only BatchNorm `running_mean`/`running_var`/`num_batches_tracked` and leaves every parameter bitwise; the result is frozen; `no_op` picks each branch on a constructed pair (recalibrated better; raw better), never reads development, and on a fixture where recalibration materially changes the front BatchNorm statistics evaluates the recalibrated candidate on recomputed features (assert the two candidates' selection features differ); `load_noop` with the raw host's hash refuses when N0 was R-bn; `temperature` of logits scaled by 2 recovers ≈ 2 within 1e-3; `class_bias` has 11 numbers.
- [ ] **Step 2: Run; fail.** **Step 3: Implement.** **Step 4: Run; pass.** **Step 5: Commit** `M0: BatchNorm recalibration, logit calibration and the no-op rule`.

### Task 10: Re-tune arms with declared subsets

**Files:**
- Create: `src/sultai_screen/m0/retune.py`
- Test: `tests/test_m0_retune.py`

**Interfaces:**
- Produces: `Subset = dict[str, Tensor]` (parameter name → boolean mask of the same shape); `R272S(host) -> Subset` = `tail.0.weight[0:16, :, 1, 1]` (256 entries) and `tail.0.bias[0:16]`; `RAFF(host) -> Subset` = `front.8.weight`, `front.8.bias` (producer BatchNorm, 32), `tail.1.weight`, `tail.1.bias` (128), `classifier.bias` (10); `RALL(host) -> Subset` every parameter; `count(subset) -> int` (272, 170, 284,634); `complement_hash(host, subset) -> str` (SHA-256 over every parameter and buffer with the subset entries zeroed before hashing); `fit_retune(host, subset, *, data: Dataset, fit_indices, selection, seed, rate, steps, checkpoints, from_images: bool) -> RetuneCandidate` with the same minibatch stream labels as `fit_sgd` (`oracle-order`), gradients masked by the subset, host in evaluation mode throughout (BatchNorm statistics frozen), `from_images=False` for R-272s (trains on the cached site through `tail`), `True` for R-aff and R-all (forward from normalised images without augmentation **[plan]**); `RetuneCandidate.host: Host` (a frozen copy), `.selection`, `.snapshots`, `.wall_seconds`, `.example_passes`, `.parameter_steps` = `count(subset) * steps`; `retune_grid(...)` mirroring `grid` with `RETUNE_RATES`; for R-all, `recalibrate_after(candidate: RetuneCandidate, data, fit_indices) -> RetuneCandidate` applies R-bn to the selected candidate after fitting (§5: "BN recalibrated after fitting") and records both the pre- and post-recalibration selection and development losses; the complement-hash invariant is checked **before** that recalibration (R-all has no complement; the check applies to R-272s and R-aff) and the recalibration is recorded as a separate, permitted buffer change.

- [ ] **Step 1: Write the failing tests:** counts 272 / 170 / 284,634; after `fit_retune` with R-272s for 3 steps, `complement_hash` is unchanged and the 272 declared entries changed; the same for R-aff; a mask covering nothing raises `ContractViolation`; `from_images=True` on R-aff changes `front.8` while every convolution weight is bitwise unchanged; `parameter_steps == 272 * 3`.
- [ ] **Step 2: Run; fail.** **Step 3: Implement.** Clone the host (`copy.deepcopy` after unfreezing a fresh `load_host(..., frozen=False)` copy), set `requires_grad` only on subset tensors, multiply gradients by masks before the update. **Step 4: Run; pass.** **Step 5: Commit** `M0: re-tune arms on declared parameter subsets with complement hashing`.

### Task 11: S-static training

**Files:**
- Create: `src/sultai_screen/m0/static.py`
- Test: `tests/test_m0_retune.py`

**Interfaces:**
- Produces: `StaticHost(nn.Module)` holding a `Host` and module tensors `w [16, 16]`, `b [16]` as `nn.Parameter`s created with `torch.zeros` after the host (no RNG draw), forwarding `host.classify(host.bottleneck(front(x)) + w·tanh(z) + b)`; `train_static(seed, data, device, widths) -> tuple[StaticHost, list[str], Tensor, float]` calling `train_epochs(static, static.host, data, seed, epochs=range(10), rates=constant(0.05), optimizer=recipe_optimizer(static.parameters()))` so `w` (`ndim > 1`, weight decay 5e-4) and `b` (0) join the recipe's groups and the per-epoch hashes are the inner host's; `static_adapter(static: StaticHost) -> Adapter`; `static_hash(static) -> str` = SHA-256 over the inner host's `state_hash` and the module tensors (`w`, `b`) so the trained static model has one identity; `save_static(static, output)` writing `host.pt`, `static-module.pt` and `optimizer.pt`; `load_static(root, seed, widths, device) -> StaticHost` verifying `static_hash`; the S-static arm on the frozen static host is `losses(static.host, cached(static.host).development, static_adapter(static))` (**the complete model, module present**) for the no-refit D5 term, and for the refit term the module grid run with `start=static_adapter(static)` **[plan: "refitted" read as a warm start from the learned module; a fresh-init refit is reported beside as a descriptive]**.

- [ ] **Step 1: Write the failing tests:** `StaticHost` at construction has `state_hash(static.host) == state_hash(make_host(seed, 0, cpu))` and its forward equals the plain host forward bitwise; after one training step on the fixture `w` is nonzero; the host's 10-epoch stream under `train_static` differs from the plain host's (the module is in the graph) while the data-order stream labels are identical (assert the first batch indices equal); `save_static`/`load_static` round-trip reproduces `static_hash` and a tampered module tensor is refused; the no-refit D5 term uses the complete model (assert it differs from the inner host alone when `w ≠ 0`) and the refit's `step0_selection` equals the complete model's selection loss.
- [ ] **Step 2: Run; fail.** **Step 3: Implement.** **Step 4: Run; pass.** **Step 5: Commit** `M0: static-from-step-zero host`.

### Task 12: P1-only instrument

**Files:**
- Create: `src/sultai_screen/m0/instrument.py`
- Test: `tests/test_m0_instrument.py`

**Interfaces:**
- Produces: `P1_LADDER = (0.1, 0.3, 0.03)`; `materiality(host, cache, seed) -> Record` (P1 rungs only; exact-inverse check 1e-4 as `screen.materiality_worker`); `materiality_report(records: tuple[Record, ...]) -> Record` (per-rung gains and whether each rung clears 0.05 nats on every anchor, descriptive only); `ALPHA = 0.1` (§2: the P1 known answer is "A1, A2, A3, B3 at α = 0.1"; no ladder fallback replaces it, and α is not a pilot-fill field under §9); `posthoc(host, cache, seed) -> Record` at `ALPHA` with arms `A1`, `A2`, `A3`, `B3` (B3 via the unchanged Gate 3 `oracle()`), `oracle_gain_fraction_of_A3`; `check_instrument(records) -> None` raising `InstrumentHold` unless every anchor has B3 ≥ `RHO` · A3 and the A1/A3 known answer within 0.01.
- Consumes: `impair.make_impairment("P1", ...)`, `fit.ridge`, `fit.oracle`.

- [ ] **Step 1: Write the failing tests:** on a CPU fixture `materiality` returns only `P1`; `posthoc` runs at `ALPHA = 0.1` and has no `alpha` parameter; `check_instrument` holds when one anchor's fraction is below 0.8 and holds (does not fall back to another α) when P1 at 0.1 fails to clear 0.05 nats on an anchor; **import isolation**: after `import sultai_screen.m0.instrument` in a fresh subprocess, `sultai_screen.screen`, `sultai_screen.pilot`, `sultai_screen.former_bridge` and `sultai_screen.alignment` are absent from `sys.modules` (the instrument uses `impair`, `fit.ridge` and `fit.oracle` directly).
- [ ] **Step 2: Run; fail.** **Step 3: Implement.** **Step 4: Run; pass; the nine Gate 3 tests named in Global Constraints and `test_v42_ladder` untouched (`git diff --stat tests/test_screen.py tests/test_v42_ladder.py` empty).** **Step 5: Commit** `M0: P1-only materiality, post-hoc and instrument check`.

### Task 13: Quantities, bootstrap, sign test, sizing

**Files:**
- Create: `src/sultai_screen/m0/stats.py`
- Test: `tests/test_m0_stats.py`

**Interfaces:**
- Produces: `AnchorResult` (frozen dataclass holding, per arm, the **aligned per-example dev loss vector and 0/1 correctness vector** in caller order, the certification flag, the selected rate with its edge-of-grid flag, the fold residual, the cut survivors and the raw-candidate reports) and `quantities(result: AnchorResult) -> Record` with `D1`, `D1prime` (`G(M-sq) − max(G(R-272s), G(R-aff))` taken **per anchor** before any aggregation), `D2`, `D2s`, `D2_random`, `D2s_random`, `D3`, `G_tcool`, `D4`, `D5`, `D5_norefit`, `r_aff`, `delta_acc`, `G_rtemp`, `G_rcal`, `retune_fraction` = `G(M-sq)/G(R-all)` when R-all ran (reported `None` with reason `"denominator ≤ 0"` when `G(R-all) ≤ 0`), `unfolded` losses for every cut, the `leave_one_out` ceiling cut's gain, survival curves `G(M-wide→h)/G(M-wide)` for sliced and refit (`None` when `G(M-wide) ≤ 0`), and `harm` = the list of every raw candidate of every arm whose dev loss exceeds N0's by more than `HARM` (inherited v4 §7 harm screen; no anchor dropped, no reselection on dev); interval conventions **[plan, fixed here so two implementers compute the same numbers]**: `paired_bound(difference, seed, *, lower)` = mean(difference) ∓ 1.645 · `fit.paired_se(difference, seed)` (the v4 §7 normal bound on the per-example paired SE, one-sided 95 %); `median(values)` = `statistics.median` (mean of the two middle order statistics at even n); `anchor_bound(values, seed, *, lower, samples=10_000)` = resample the anchors with replacement `samples` times under `generator(seed, "anchor-bootstrap")`, take the median of each resample, and return the 500th smallest for `lower` and the 9,500th smallest for `upper` (order statistics, no interpolation); `achieved_half_width(values, seed)` = max(median − lower, upper − median); `sign_test_bound(values, *, lower)` = the exact binomial one-sided 95 % confidence bound on the population median by order statistics (the k-th smallest for `lower` with k the largest integer such that P(Bin(n, ½) < k) ≤ 0.05; symmetric for `upper`; exact zeros count against the direction tested; returns `None` with reason when n < 6); `sd(values)` = `statistics.stdev` (sample SD, n − 1); `confirmatory_count(sd_d1: float) -> tuple[int, int]` = `contracts.anchor_count(sd_d1)`; `frontier_slope(points: dict[float, tuple[float, ...]], seed) -> Slope` where `points` maps each core width's parameter count to its six seeds' N0 dev losses, the fitted response is mean dev loss against log2(parameters) by ordinary least squares over the three width means, **slope reported as the loss decrease per doubling** (the negated fitted slope), interval by resampling the six seeds **within each width** independently 10,000 times under `generator(seed, "frontier-bootstrap")` with 500th/9,500th order statistics; `gain_fraction(gain_per_anchor: tuple[float, ...], points, seed) -> Fraction` bootstrapping the reference-host module gain and the slope **jointly** (the reference width's resampled seeds are the same draw in both numerator and denominator) and returning the median fraction with 500th/9,500th bounds, or `None` with reason `"slope ≤ 0"`; `frontier_slope_descriptive(points_all_10_epoch, seed)` the same fit over every 10-epoch width that ran, labelled, the 20-epoch point excluded, no extrapolation.

- [ ] **Step 1: Write the failing tests (numerical reference cases, not inequalities):** `D1 = G(M-sq) − G(M-aff)` on a constructed record; `D1prime` on a record where R-aff beats R-272s on one anchor and not another equals the per-anchor maximum; `D3` is `L(T-cool N0) − L(M-sq | T-cool)` and not a difference against the anchor's N0; `paired_bound` on a fixed 200-element vector equals mean ∓ 1.645 · SE to 1e-9 with the same seed; `anchor_bound` of six identical values is that value and on the fixed sample `(0.01, 0.02, 0.03, 0.04, 0.05, 0.06)` with seed 0 equals a literal pinned once from the implementation and re-derived by an independent 20-line script in the test; `sign_test_bound` at n = 6 returns the 1st order statistic for `lower` and the 6th for `upper` and at n = 32 the 11th and 22nd, `None` at n = 5, and a zero in the sample counts against; `achieved_half_width` on that sample equals max of the two distances; `confirmatory_count(0.0275)` returns `(32, 32)` and `(0.03)` returns `(39, 39)`; `frontier_slope` on three width means lying exactly on a line of slope −0.1 per doubling reports `+0.1` with a zero-width interval when all seeds are identical; `gain_fraction` with gain 0.02 and that slope reports 0.2; `retune_fraction` and survival curves return `None` with the reason on a zero denominator; `harm` lists exactly the two raw candidates in a fixture that exceed N0 by 0.011 and omits one at 0.009.
- [ ] **Step 2: Run; fail.** **Step 3: Implement.** **Step 4: Run; pass.** **Step 5: Commit** `M0: per-anchor quantities, bootstrap bounds, sign test, sizing`.

### Task 14: Verdicts as pure functions

**Files:**
- Create: `src/sultai_screen/m0/verdicts.py`
- Test: `tests/test_m0_stats.py`

**Interfaces:**
- Produces: `v0(replays: Record, instrument: Record, identities: Record, slicing: Record, residuals: tuple[float, ...]) -> Record` with `passed: bool` and a `failed: list[str]` naming each §7 V0 clause (replay, A1 1e-4, B3 ≥ ρ·A3, zero-module identity, printed counts 33h + 16, slicing known answer, complement hash, S-static step 0, `|r_aff| ≤ FOLD_TOLERANCE` on every anchor); `v1(d1_lower, d1_upper, maff_certified_fraction, msq_certified_fraction) -> str` in `{"present", "re-tuning equivalent", "indeterminate"}`; `labels(delta_acc_lower, d3_lower, d3_upper, d1prime_lower, d4_lower) -> list[str]` from the §7 label set; `v2(d2_lower, d2_upper, msmallb_certified, refit_certified) -> str` in `{"helps", "hurts", "indeterminate"}`; `static_reading(d5_lower, in_confirmatory_set: bool, n: int) -> str | None`.

- [ ] **Step 1: Write the failing tests:** one assertion per branch: V1 present needs lower ≥ 0.02 **and** M-aff certified ≥ 0.75; equivalent needs upper < 0.02 and M-sq certified ≥ 0.75; both unmet → indeterminate; V2 helps needs lower ≥ 0.01 and M-small-B ≥ 0.75; hurts needs upper < 0 and refit ≥ 0.75; `"loss gain without accuracy movement"` appears when `delta_acc_lower ≤ 0` and never changes `v1`; a residual of 0.006 fails V0 naming the fold clause; `static_reading` is `None` at n = 6.
- [ ] **Step 2: Run; fail.** **Step 3: Implement.** **Step 4: Run; pass.** **Step 5: Commit** `M0: V0, V1, V2 and labels`.

### Task 15: Supervisor ceiling and one-per-card; M0 costs

**Files:**
- Modify: `src/sultai_screen/pilot.py:131-141` (`Supervisor.__init__`), `:142-149` (`run` admission), `:264-272` (ledger)
- Create: `src/sultai_screen/m0/costs.py`
- Test: `tests/test_m0_supervision.py`

**Interfaces:**
- Produces: `Supervisor(root, data, opening_charge=0.0, *, ceiling: float | None = None, per_card: int = 3, module: str = "sultai_screen.screen")` with `self.ceiling = PILOT_SECONDS if ceiling is None else ceiling` resolved in the body before the opening-charge check at `pilot.py:135` (so the existing `patch("sultai_screen.pilot.PILOT_SECONDS", 10.15)` still binds), every former `PILOT_SECONDS` read inside the class (`:135,150,268,278`) replaced by `self.ceiling`, `per_card` replacing the literal 3 at `:146`, the ledger's `ceiling_gpu_process_seconds` = `self.ceiling`, `-m {self.module}` in the child command; `Job` gains a trailing field `extra: tuple[str, ...] = ()` appended to the command after `--magnitudes` (M0 passes `("--lineage", ..., "--scale", ...)`); with the defaults the command list is byte-identical to today's (`pilot.py:163-184`); `m0/costs.py`: `PilotKernels` (frozen dataclass of one-per-card maxima: host seconds per width, grid seconds per arm kind, front-rerun seconds, evaluation seconds, fixed stage seconds); `project_core(k: PilotKernels) -> Record` (the §8 core lines from measured kernels, never from bookings); `EXTENDED_ORDER` = the seven §5 items in order; `admit_extended(charged: float, core_projection: float, k: PilotKernels, ceiling: float) -> list[str]` returning the admitted prefix and stopping at the **first** item that does not fit; `Workloads` (frozen dataclass of measured one-per-card seconds per unit: host per width, T-cool resume, static training, each grid kind, each refit kind, front-rerun arm, fold and calibrations, evaluation, the fixed V0 stage total) with `per_anchor(w: Workloads, arms: tuple[str, ...]) -> float` summing the measured units of the arms named (never a literal); `project_pilot(charged: float, remaining: tuple[str, ...], w: Workloads) -> Record` = charged so far plus the remaining stages' measured workloads; `project_confirmatory(n: int, w: Workloads, arms: tuple[str, ...], fixed: float) -> Record` with `within_ceiling` against `M0_CONFIRMATORY_SECONDS`; `with_static(n, w, arms_with_static, fixed)` using the pilot's **measured** static training and refit cost, carrying `ceiling_seconds: 50_400` and `requires_14h: True`.

- [ ] **Step 1: Write the failing tests:** `Supervisor(root, root, ceiling=14_400.0, per_card=1)` refuses two jobs on one card and writes `ceiling_gpu_process_seconds: 14400.0`; `Supervisor(root, root)` still writes 10,800 (the Gate 3 ledger check in `linked_pilot_two` keeps passing: run `test_protocol_identity` and `test_screen.test_supervisor_real_concurrency_cleanup_and_budget` unchanged); the child command for a default `Job` equals today's list literally (capture `subprocess.Popen` args) and a `Job(extra=("--lineage", "frontier-0.5"))` under `module="sultai_screen.m0.worker"` appends those two tokens and runs `-m sultai_screen.m0.worker`; `admit_extended` with a 2,000 s margin admits nothing after the first miss even if a later cheaper item would fit; the core-over-ceiling case returns a stop before V1; with a `Workloads` fixture whose §10 arms sum to 1,010 s per anchor, `project_confirmatory(32, w, CONFIRMATORY_ARMS, 1500)` projects 33,820 s within 43,200; with a measured static unit of 110 s, `with_static` projects 37,340 s and carries `ceiling_seconds: 50_400` and `requires_14h: True` (S-static's exclusion under 12 h is the §10 rule, not arithmetic); `project_pilot` on a fixture with 3,000 s charged and two remaining stages of 1,000 s projects 5,000 s, not 7,000.
- [ ] **Step 2: Run; fail.** **Step 3: Implement.** **Step 4: Run; pass; full suite green.** **Step 5: Commit** `supervisor: call-time ceiling, per-card admission, worker module and extra args; M0 cost projection and extended admission`.

### Task 16: M0 worker actions and CLI

**Files:**
- Create: `src/sultai_screen/m0/worker.py`
- Test: `tests/test_m0_pilot.py`

**Interfaces:**
- Produces: actions `train` (frontier/anchor host at `widths`, saves `host.pt`, `optimizer.pt`, `loss-stream.bin`; payload adds `widths`, `schedule`, `final_state_sha256`), `tcool` (loads host and optimiser unfrozen, `train_epochs(epochs=range(10, 15), rates=TCOOL)`), `cooled20` (`range(20)`, `COOLED_20`), `static` (Task 11), `instrument` (Task 12 materiality + post-hoc in one child), `noop` (R-bn, R-temp, R-cal, N0 choice; publishes the chosen host hash), `modules` (the module arms for one anchor on the saved N0 (`load_noop`), restricted by `--arms`: M-sq, M-aff + fold residual, M-sq-fresh, M-wide, M-wide→8 by contribution rank and by random rank each sliced-with-fold, unfolded and refit on the full module grid, M-wide→{16, 4, 2} sliced-with-fold and **one refit at the selected M-wide rate for 800 steps** (§5, not the grid), the leave-one-out ceiling cut, M-small, M-small-B, M-exp, lottery, and on the T-cool host (after its own N0 rule) M-sq | T-cool; on the static host the no-refit complete-model term and the warm-start refit; each candidate's report, per-example losses and correctness, snapshots, and the harm inventory of raw candidates), `retune` (R-272s, R-aff; R-all when admitted), `replay` (train seed 3000 at widths 1.0 and compare `final_state_sha256` to the pinned attempt-4 string), `timing` (one module grid on the new fitter plus the first front-re-running arm, one-per-card); `ATTEMPT4_FINAL_STATE: dict[int, str]` = the six strings from `docs/results/gate3-pilot-2026-10-13-linked-3/intact-<seed>/result.json` (3000 `f607728b…`, 3001 `b9550263…`, 3002 `e7f1a21f…`, 3003 `ea015274…`, 3004 `f9a39442…`, 3005 `aa5b5af4…`, full 64-character strings in source); receipts carry `m0.identity.protocol_identity()` through `m0.provenance.envelope` (never `screen.worker`); `load_training(root, seed, lineage, widths, device, *, frozen) -> Host` is the M0 analogue of `screen._training` and additionally refuses a receipt whose `payload["widths"]` or `payload["lineage"]` differ from the request (the §2 "training check that reads the width tuple"); `main()` admits seeds through `seeds.admit(seed, lineage)` with `--lineage` and `--scale`.
- Consumes: Tasks 3–12.

- [ ] **Step 1: Write the failing tests:** `main(["train", "--seed", "3000", "--lineage", "frontier-0.5", ...])` exits with a parser error; `"--seed", "6000", "--lineage", "frontier-0.5"` is admitted; `ATTEMPT4_FINAL_STATE[3000]` equals the string read from the published receipt; `load_training` raises `PilotHold` on a receipt whose `widths` is `(32, 32, 64, 64)` when `(64, 64, 128, 128)` is requested; a subprocess that imports `sultai_screen.m0.worker` has neither `sultai_screen.screen` nor `sultai_screen.pilot` in `sys.modules`; a `noop` payload on a CPU fixture (patched `configure(cuda=False)`) records `"R-bn"` or `"raw"` and never a development number; a `modules` payload on the fixture with `steps` patched to 2 contains every arm name from §5's core table, `delta_acc`, the harm inventory, unfolded losses and survivors for every cut, and runs the h = 16/4/2 refits as single 800-step runs at the M-wide rate (assert one candidate each); a `--arms` list excluding `M-exp` leaves it absent; every receipt's `identity` has `protocol_version == "m0-v4"`.
- [ ] **Step 2: Run; fail.** **Step 3: Implement.** Follow `screen.worker` for the receipt envelope and `failure.json`. **Step 4: Run; pass.** **Step 5: Commit** `M0: worker actions, pinned attempt-4 hashes, seed-admitting CLI`.

### Task 17: M0 pilot supervisor and admission

**Files:**
- Create: `src/sultai_screen/m0/pilot.py`
- Test: `tests/test_m0_pilot.py`

**Interfaces:**
- Produces: `check_cards() -> Record` (nvidia-smi: both pinned UUIDs present, utilisation 0, no compute apps; **[plan: the Simic terminal receipts and the 2026-10-11 date in Gate 3's `check_handover` are not re-checked; both cards are granted to Sultai after Simic rung 4 and attempt 4's admission recorded the terminal receipts]**); `check_certificate(approved: Record) -> Record` against `m0.identity` (fields `quality_passed`, `runtime_tests_passed`, `independent_panel_accepted`, `implementation_source_sha256`, `implementation_commit`, ≥ 3 hashed in-repository evidence paths; under the bundled gates "independent panel" is satisfied by the preserved per-item exact-source reviews of Task 19); `CUSTODY = ".weft/m0-pilot-attempt.json"` (never one of the five Gate 3 receipts); `pilot(root, data, certificate) -> int` running §9 stages in order with receipts `01-replay-timing.json` … `07-development-medians.json` and `pilot-result.json`, `Supervisor(root, data, ceiling=M0_PILOT_SECONDS, per_card=1, module="sultai_screen.m0.worker")` with each `Job.extra` carrying `--lineage`/`--scale`, `m0_stop_report(error, stage, charged) -> Record` (the Gate 3 `stop_report` shape with `**m0.identity.protocol_identity()`, never the Gate 3 identity); stage 1 runs the host replay on both cards, the timing children and (no co-tenancy replay; withdrawn in v4 §14) the **GPU acceptance matrix** on both pinned cards, each booked in the ledger under the §8 600 s line: (i) resume versus uninterrupted, one seed, `train` to 12 epochs straight against `train` to 10 then `tcool`-style resume for 2 under `constant(0.05)`, final hashes equal; (ii) one checkpointed grid against three separate fits of the checkpoint lengths, adapter bytes equal; (iii) the complete static model at step 0 forwards bitwise as the plain host; (iv) `logits`/`losses`/`accuracy` on one anchor's dev set agree with each other and `losses` matches a padded reference; (v) the selected M-aff fold's residual recomputed on GPU equals the receipt value within 1e-6; stage 4 projects from one-per-card maxima as **charged so far plus remaining work**, so no stage is counted twice, publishes `04-admission.json` with the admitted extended prefix, and publishes `04-schedule.json`: the schedule axis as run (default checkpoints `(200, 800, 3200)` / `(400, 1600, 6400)`) or a **raise-only extension** decided from the certification fractions with the doubled M-small-B axis changed in step so the two D2 routes keep matched optimiser steps and example passes; every later arm reads its checkpoints from that receipt; stage 5 publishes `05-spreads.json` with per-anchor SDs of D1, D2, D3, D5 **before** any median, and `static_admitted: bool` = `SD(D5) ≤ S_STATIC_SD_LIMIT` with the SD beside it; stage 6 publishes `confirmatory_count` and the projection from measured per-anchor workloads (the measured static-training and refit cost included when `static_admitted`) plus the fixed V0 term and raises `FeasibilityHold` above the ceiling; the core-over-ceiling stop fires in stage 4 before any V1 field is published; stage 7 medians flagged `development_only`; the `train` and `static` receipts bind `optimizer.pt` and `static-module.pt` by SHA-256 exactly as `host.pt` and `loss-stream.bin` are bound today (`screen.py:135-142`).

- [ ] **Step 1: Write the failing tests** (CPU, children patched as in `test_screen.test_supervisor_real_concurrency_cleanup_and_budget`): custody is claimed before any child and cannot be reset by a new output root; an existing M0 custody receipt holds; a Gate 3 custody receipt's presence is irrelevant; a replay hash mismatch at stage 1 stops with `PilotHold` before stage 2; a fold residual of 0.006 on one anchor stops at stage 2 naming the fold clause; the SD receipt (`05-spreads.json`, with `static_admitted`) exists before `07-development-medians.json` and the medians are absent when stage 6 holds; `04-schedule.json` exists before any module child launches and a raised axis doubles M-small-B's in the same receipt; a stop at any stage publishes `pilot-result.json` carrying `protocol_version == "m0-v4"`; the projection in stage 4 equals charged plus remaining on a fixture where two stages have already run; the ledger ceiling is 14,400; no job list puts two jobs on one card.
- [ ] **Step 2: Run; fail.** **Step 3: Implement.** **Step 4: Run; pass.** **Step 5: Commit** `M0: ordered pilot with its own custody, ceiling and receipts`.

### Task 18: Confirmatory launcher

**Files:**
- Create: `src/sultai_screen/m0/confirmatory.py`
- Test: `tests/test_m0_pilot.py`

**Interfaces:**
- Produces: `CUSTODY = ".weft/m0-confirmatory-attempt.json"`; `confirmatory(root, data, certificate, pilot_result: Path, *, static: bool = False) -> int`: reads the committed pilot result (count `n`, admitted arms, kernels), refuses if `pilot_clean` is not true or the pilot's identity differs, draws anchors `7500 … 7500 + n − 1` (restart block `7700 …` only with an explicit `--restart` and a disposition path), runs V0 then the §10 arm set on every anchor under `Supervisor(root, data, ceiling=M0_CONFIRMATORY_SECONDS, per_card=1, module="sultai_screen.m0.worker")` (`50_400` with `static=True`, which requires both `pilot_result["static_admitted"] is True` and `certificate["confirmatory_ceiling_hours"] == 14`, John's separate authorisation recorded in the admission certificate); `CONFIRMATORY_ARMS` is the §10 whitelist (N0 and calibrations, M-sq, M-aff with residual, R-272s, R-aff, T-cool, M-sq | T-cool, M-wide, M-wide→8 both rankings sliced and refit, M-small, M-small-B; S-static only under the 14 h flag) and the worker's `modules`/`retune` actions take an `--arms` list that the confirmatory restricts to it, so no extended or pilot-only arm (frontier, M-sq-fresh, M-exp, lottery, R-all, h = 16/4/2) can run; `--restart` is admitted only with a disposition path whose SHA-256 is listed in the certificate's evidence and with the prior `.weft/m0-confirmatory-attempt.json` present and hash-bound in the new custody receipt (the `linked_pilot_*` pattern, `pilot.py:438-577`); reads V1 and V2 once, publishes `confirmatory-result.json` carrying the M0 identity; a V0 failure ends the attempt and marks the seed block burnt in the receipt.

- [ ] **Step 1: Write the failing tests:** refuses a pilot result with `pilot_clean: false`; refuses `static=True` when the pilot did not admit S-static or the certificate lacks the 14 h authorisation; a job for an extended arm (`M-exp`) is refused by the whitelist; `--restart` without the disposition hash in the certificate or without the prior custody receipt is refused; ceiling 43,200 in the ledger, 50,400 with static; the child command runs `-m sultai_screen.m0.worker`; anchors are exactly `7500..7500+n-1`; a V0 failure publishes `verdict: "instrument-failed"` and no V1/V2 fields; the result carries the M0 identity.
- [ ] **Step 2: Run; fail.** **Step 3: Implement.** **Step 4: Run; pass.** **Step 5: Commit** `M0: confirmatory launcher with a 12 GPU-hour hard stop`.

### Task 19: Gate B evidence, per-item reviews and admission certificate

**Files:**
- Create: `docs/results/m0-implementation-<date>/quality.json`, `runtime.json`, `runtime.txt`; `docs/reviews/m0-capacity/implementation/<item>.md` (one per §2 item below); `docs/certificates/<date>-m0-pilot-admission.json`
- Modify: `docs/RESUME.md` (state), `README.md` (pointer); `CLAUDE.md` already carries the three bundled gates (landed with this plan at `91412df`)

- [ ] **Step 1: Run the gates on the final commit:** `scripts/check_policy.py`, `scripts/check_quality.py --ruff <path> --output docs/results/m0-implementation-<date>/quality.json`, `mypy src scripts`, the full suite with output captured to `runtime.txt`; write `runtime.json` (commit, test count, OK).
- [ ] **Step 2: Exact-source reviews, one file per §2 item**, each by a fresh reviewer briefed read-only with the item's draft text, the diff and the tests: width-tuple constructor and bitwise reference host; pinned attempt-4 hashes and replay; seed admission; trainer schedules and optimiser state; seen-fit partition; accuracy evaluator; generalised fitter and checkpoint bitwise claim; module arms and slicing known answer; M-aff fold and residual; calibration and N0; re-tune subsets and complement hash; S-static; P1-only instrument and import isolation; statistics and verdicts; supervisor ceiling, one-per-card and costs; worker, pilot and confirmatory custody. Findings are dispositioned in the same file; a finding that changes a frozen value is amendment 2 and stops the PR.
- [ ] **Step 3: Admission certificate JSON** with the M0 identity, `implementation_commit`, `implementation_source_sha256`, the three flags, the evidence paths and hashes, `pilot_gpu_hour_ceiling: 4`, `confirmation_cleared: false`, `pr_url`.
- [ ] **Step 4: Commit** `M0: gate B evidence, per-item exact-source reviews, admission certificate`; open the PR (merge commit only); request gate B.

---

## Self-review against the spec

**§2 coverage.** Hosts and training: Tasks 3 (tuple, bitwise reference), 16 (pinned hashes, replay action), 2 and 16 (seed admission beyond 3000–3005, `--lineage`), 4 (optimiser state, T-cool, 20-epoch decay), 16 (published frontier hashes, width-tuple check at load). Fitting: Task 6 (rate sets, checkpointed single run, hidden width, warm start, affine SGD path, plateau at 800/3,200 and 1,600/6,400), Task 10 (re-tune paths with complement hash and front re-run), Task 9 (recalibration, logit calibration), Task 11 (static host), Task 7 (slicing with mean fold, ranking, known answer), Task 8 (fold and residual), Task 5 (accuracy evaluator). Instrument carve-out: Task 12 (P1-only materiality, post-hoc, check; import isolation), Task 15 (replacement cost projection), tests in new modules with the nine named Gate 3 tests and `test_v42_ladder` untouched. Supervision: Tasks 16–18 (worker actions, identity pins, tests), Task 15 (M0 ceiling as a new constant, `PILOT_SECONDS` untouched, one-per-card), Task 18 (confirmatory ceiling with its own supervisor instance), Task 13 (anchor bootstrap, sign test, aggregation). "A stage that depends on new code reads nothing until that code's review is preserved": Task 19 before any pilot go.

**§6 reports** each have a producer: D1, D1′ (per anchor), D2, D2s and their random-rank versions, D3 and G(T-cool), D4, D5 with and without refit, r_aff, ΔAcc, R-temp/R-cal gains, the re-tune fraction, unfolded slices, the leave-one-out ceiling, survival curves, the frontier slope with its fraction interval and the descriptive all-width fit, and the inherited harm inventory (Task 13 with Task 16's roster). **§3–§7 values** are copied into Global Constraints and the task interfaces; §9 stage order and the "SDs before medians" rule are pinned by Task 17's tests; §10's S-static exclusion under 12 h by Task 18; §11 identity by Task 1 and seeds by Task 2.

**Type consistency.** `Adapter`, `Candidate`, `Grid`, `Features`, `Cached`, `Dataset`, `Host`, `Subset`, `RetuneCandidate`, `StaticHost`, `AnchorResult`, `PilotKernels` are each defined in one task and used by name afterwards. `train_epochs` takes the optimiser and returns the Gate 3 triple; `tcool` and `cooled20` rebuild it with `restore_optimizer` from the saved momentum buffers.

**Proportion.** Code bodies appear only for the fold formula, the slicing criterion and fold, the schedules and the seen-fit derivation; everything else is a signature and a test assertion. The §2 items that are prose in the draft (import isolation, call-time ceiling binding, zero-init static module) each have a named test.

**Open to John at gate A, not blocking:** (a) T-cool and the 20-epoch point decay per optimiser step rather than per epoch **[plan]**; (b) re-tune arms that re-run the front use un-augmented normalised images **[plan]**; (c) M0's admission does not re-check the Simic terminal receipts **[plan]**; (d) the 0.5-host extended pair uses all six frontier-0.5 seeds as its anchors **[plan: anchor count six, matching the reference set]**; (e) "the test rewrite" in §2 and "the identity tests are rewritten" in §11 are read as new `tests/test_m0_*.py` modules with the Gate 3 tests untouched, which keeps the Gate 3 chain verifiable; (f) S-static's "module grid refitted on the frozen static host" is read as a warm start from the learned module, with a fresh-init refit reported beside as a descriptive **[plan]**; (g) the interval conventions fixed in Task 13 (normal per-example bound on the paired SE, order-statistic percentile anchor bootstrap, exact binomial sign-test bound by order statistics) **[plan]**. Any of these can be changed in one line before implementation starts.

**Amendment candidates, resolved.** Both reviews raised three spec defects (the co-tenancy replay under one-per-card, the programme claim versus the V1 gate, the example-pass arithmetic). John decided all three on 2026-10-13 and they are taken as preregistration v4 §14 with gate-ladder amendment 2 and freeze certificate 2 (identity `m0-v4` = `3a76c6dbe31c…`). The plan implements v4: no co-tenancy stage (Task 17), the frozen V1 with the ranked programme reading stated in reports (Task 14), and actual example passes recorded (Tasks 6, 13).
