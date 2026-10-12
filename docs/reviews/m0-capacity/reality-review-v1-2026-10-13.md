# Reality review of the M0 draft v1 against the repository

Reviewed bytes: `docs/plans/2026-10-13-capacity-frontier-screen.md` as
committed at `5aeee56`, checked against `src/sultai_screen/*`, `tests/`
and the attempt-4 receipts. Reviewer: fresh read-only subagent
(`axiom-planning:plan-review-reality`), 2026-10-13; CPU-only checks, no
GPU, nothing edited or committed. Report preserved verbatim below;
dispositions are in [the disposition record](v1-dispositions-2026-10-13.md).

---

# Reality review: docs/plans/2026-10-13-capacity-frontier-screen.md (draft v1)

Repository /home/john/sultai at da06e26 plus uncommitted docs. Read-only; CPU-only checks (torch 2.13 CPU from /home/john/simic/.venv for the Host count; stdlib JSON over the attempt-4 receipts). No GPU, no edits, no commits.

## Findings

**1. MAJOR — §5 "Everything else is a composition of existing code" is false for most arms.** Evidence:
- `oracle()` never passes `wide=True` and hard-codes schedule `(200, 800)` and `ORACLE_RATES = (0.3, 0.1, 0.03, 0.01)` (/home/john/sultai/src/sultai_screen/fit.py:279-287, contracts.py:48). M-wide "on the oracle grid", the 3-rate set and the 3,200-step arm are all new code.
- `fit_sgd(wide=True)` builds only `((64,16),(64,),(16,64),(16,))` (fit.py:177) and has no warm-start/initial-adapter argument. M-small (h=8 from scratch) and every "refit after truncation" need a new entry point.
- `Host.__init__` accepts only bottleneck widths `(0,2,4,8)` and hard-codes tail widths 64/64/128/128 (host.py:59-71); `Host(16)` raises `ContractViolation` (verified). F(s) needs a new constructor parameter.
- `train_host` raises on `epochs != 10` (host.py:130-131): T-more and the optional 20-epoch frontier need a change.
- `screen.main` rejects any seed outside 3000-3005 (screen.py:442-443); `pilot()` is the Gate 3 stage sequence end to end (pilot.py:626-866) with no anchor-bootstrap, verdict or N_eq aggregation anywhere.
- The plateau rule hard-codes the `[720:800]` tail (fit.py:298); an 800-vs-3,200 comparator with "last two 40-step windows" is new.
Correction: replace the sentence with an explicit list: existing = padded evaluator, `losses()` on any 4-tensor adapter, `paired_se`, `identity_check`, host training at s=1.0/10 epochs, custody/ledger supervisor; new = frontier constructor, epoch override/resume, oracle grid (rates, schedules, wide flag, warm start, h parameter), truncation + ranking, R-* fit paths, S-static training graph, verdict/bootstrap/N_eq aggregation, new worker actions and seed admission.

**2. MAJOR — R-all and R-aff break the cached-site design, not only the hash check.** `cached()` computes `read`/`base` once from the frozen front (host.py:220-236); `gradient_step`/`losses` consume that cache (fit.py:140-156, 55-75). The producer BN named in R-aff (32 params) is the last BN of `host.front` (host.py:61); R-all touches front too. Both require recomputing the site from images every step, a different kernel and determinism surface. R-272 (first tail conv, host.py:64) can use the cache. The §5 sentence names only R-272 and R-all as needing a new path; R-aff is omitted and §8's "4× module kernel" hedge covers only R-all. Also R-272's "16→64, 9,216 weights" subset does not exist on the 0.5 host (16→32, 4,608 weights); §5 says the Q1 pair runs there. Correction: list R-aff with R-272/R-all; state that R-aff and R-all re-run the front per step (or cache the pre-BN producer output for R-aff); define R-272's subset for the 0.5 host.

**3. MAJOR — §11 S-static pairing is self-contradictory.** `train_host` derives order/crop/flip from the seed it is given (host.py:152-154) and `make_host` seeds the global CPU RNG once with `derive(seed, "host-init")` then constructs all layers in order (host.py:83-85). A host run under seed 6100 therefore shares neither anchor 3000's data stream nor its host init. Either S-static runs under seed 3000 with module tensors drawn from a separately labelled generator (then 6100-6105 are not host seeds and the table row must say "module-init labels" or be dropped), or it runs under 6100 and "same data stream" is false. In either case module tensors must be drawn outside the `fork_rng` block or after the classifier, or the host's own init diverges. Correction: rewrite the row and note as "S-static uses the anchor seed for stream and host init; module tensors from `derive(anchor, 'static-module-init')`; the identity test compares `epoch_state_sha256[0]` of the host-only tensors is not available, so the test must compare the data stream tensors directly."

**4. MAJOR — §8 "Replays, twins, evaluations: 300 s" is ~3× low.** Attempt-4 ledgers (docs/results/gate3-pilot-2026-10-13-linked-3/ledger-*.json) charge replay 115.4 s (3 jobs), co-tenancy training 612.7 s (6 jobs at ~102 s each because three co-tenants share a card and each is charged its own wall), timing 23.9 s, co-tenancy fit 130.7 s: 882.7 process-seconds for v4 §9 items 1-3, which §9 item 1 of the draft reproduces. Correction: book ~900 s. Note the ceiling is in process-seconds (pilot.py:153-154 charges N concurrent processes N×); "two cards halve the wall" does not reduce the charge against 10,800. All other kernel lines are within 30%: hosts charged 44.5 s each (267 s vs 240 booked), oracle worker 103 s per two 24-fit grids (candidate wall sums 49-52 s per grid), posthoc 56 s, wide grid 1.85 s for 2×200 steps, full-batch step 0.057 s, dev eval 0.038 s (timing-0-a/result.json). The 18-candidate grid is 25,200 steps vs 12,000 (2.1×); per-step cost is linear (0.94 s/200, 3.21 s/800), so 125 s/case is conservative. Table arithmetic sums to 8,500 exactly. R-all per-step cost is unmeasured and plausibly 2-3× the module step (front forward/backward added), so the 750 s R-all share is likely low by ~2×; with finding 4 the pilot is ~9,500-10,000 s against a 10,800 s ceiling.

**5. MAJOR — §11 seed collision claim is wrong for 7000-7099.** `src/sultai/lifecycle.py:359-362` uses 7001-7004 (training/conditioning/audit/random-adapter) and `random.Random(7004)` at :157. Different package and RNG family, but the draft's own list already counts correction-protocol synthetic seeds, so by its standard this is a collision. 6000-6023 and 6100-6105 collide with nothing found in src/, docs/plans, docs/HYBRID_CONTRACT.md or the correction protocol. The "already allocated" list also omits `src/sultai/formation.py:39-41` dev 2000-2007 and test 3000-3007 (the latter already overlaps Gate 3 dev anchors). Correction: move confirmatory anchors to a clean block (e.g. 7500-7599) or declare the lifecycle seeds and justify; add 2000-2007 to the list.

**6. MAJOR — §11 identity does not describe what contracts.py does.** `protocol_sha256` is a composed hash of version+prereg+amendment (contracts.py:42), not a file hash; `protocol_identity()` emits four fixed fields including `amendment_sha256` (contracts.py:72-79); `source_identity()` hard-fails unless the v4, v4.1 and v4.2 bytes are present and unchanged (screen.py:61-66); `tests/test_protocol_identity.py:60` pins `"gate3-v4.2"`; `_training()` refuses any receipt whose protocol/source hash differ (screen.py:128-132), so attempt-4 checkpoints cannot be loaded under m0-v1 and the §3 replay must compare the fresh receipt's `final_state_sha256` to the published string. Correction: say "m0-v1 replaces `PROTOCOL_VERSION`, `PROTOCOL_SHA` (composition rule to be stated), `protocol_identity()` fields (add `steering_sha256`, drop `amendment_sha256`), `source_identity()` pins and the identity tests; replay compares against the attempt-4 receipt string, not a loaded checkpoint."

**7. MAJOR — §3 bitwise replay constrains the frontier constructor.** Adding tail-width parameters to `Host.__init__` must leave the s=1.0 layer order and RNG consumption identical, or the 3000-3005 replay fails by construction. State this as a design constraint.

**8. MINOR — §4 "certified 1 of 24 fits at 800 steps [measured]".** Receipts: 2 of 24 bottleneck grids certified (oracle4-3003 P5 and P6), 0 of 6 P1 grids, 2 of 30 overall; the unit is "oracle grid" (24 fits each), not "fit". The hold certificate's "one of twelve width-4 fits" (docs/certificates/2026-10-13-gate3-linked-pilot-3-hold.md:51) is also wrong; surface separately. Correction: "2 of 30 oracle grids (both families on anchor 3003 at width 4)".

**9. MINOR — §1 "0.04 to 0.24 nats".** Hold table range is 0.037 (3004, w2, P5) to 0.260 (3001, w2, P6). Correction: "0.04 to 0.26".

**10. MINOR — §9 item 5 "n = max(16, n*)".** `anchor_count` returns `max(32, required)` with the 1.25 inflation (contracts.py:182-186); `project_cost` raises for `count < 32` (costs.py:210-215); the 16 h ceiling is hard-coded (pilot.py:825-826). Label as a code change.

**11. MINOR — T-more "continue the host's own recipe".** Only `state_dict` is saved (screen.py:159); SGD momentum buffers are not, so resume is not exact continuation without saving optimizer state. State which.

**12. MINOR — §3 "the 0.5 host is the starved host of the Claude design (P10)".** The TCD's starved variant is "about 0.07M" (docs/research/claude-handoff/sultai-concept-demonstrator-design.md:19,102); 0.5 is 85,274. Say "corresponds to".

**13. MINOR — §13 cites `docs/reviews/m0-capacity/`, which does not exist yet.** Fine as intent; mark "to be created".

**14. SUGGESTION — oracle rate set.** The attempt-4 winner sat at rate 0.3, the top of the grid, in 28 of 30 grids; dropping 0.01 costs nothing, but a rate-limited plateau is not cured by more steps. Consider adding 1.0.

## Verified clean (no change needed)
- §2 facilities: `EVAL_CHUNK = 128` with last-row padding (fit.py:40-52, tests/test_fixed_evaluation.py); `assert_frozen` hash rejection (host.py:113-117, called in `losses`/`fit_sgd`); plateau rule exactly v4 Table B (fit.py:295-303); `paired_se` 10,000 resamples (fit.py:247-256); custody receipts and ledgers (pilot.py:418-422, 264-272); P1 known answer with `RHO = 0.8` enforced per anchor (contracts.py:44, pilot.py:335-338), passed on all six anchors (fractions 1.09-1.59); data seal and 40,000/2,000/3,000/5,000 partition (data.py:493-494, 556-569); `HARM = 0.01`; `PILOT_SECONDS = 3*3600`.
- §3 counts: Host(0) = 284,634 (receipt and torch agree); frontier 85,274 / 168,826 / 284,634 / 613,018 recomputed from `_block` and match; 33h+16 gives 2,128 / 544 / 280 / 148 / 82; square 272; conv 16→64 = 9,216; R-aff 170.
- §3/§5: 4-tensor `Adapter.apply` is shape-agnostic in h (fit.py:30-33), so a sliced module evaluates through `losses()` unchanged; test_fixed_evaluation.py:131 exercises the 4-tensor path.
- §8 remaining: 4,593 s Gate 3 remainder matches execution-summary.json; steering "6,207 process-seconds" matches 6,206.67.
- Links: hold certificate, CONCEPT, diffusion report, DECISIONS D26, correction protocol all exist.

## Coverage
Checked: host.py, fit.py, screen.py, pilot.py, data.py, contracts.py, costs.py, impair.py in full; test_protocol_identity.py and greps of tests/; all attempt-4 receipts named in the brief plus ledger-000..044; v4 plan §3/§6/§7/§8/§9, v4.1/v4.2 seed blocks, correction protocol, HYBRID_CONTRACT, src/sultai seeds. Not checked: GPU timing of frontier widths, S-static or R-all steps (no GPU authorised); the 1,300 s frontier and 270 s S-static lines are estimates I cannot bound beyond "plausible".

## Verdict
Not ready as written. The parameter counts, carried-over facilities and measured kernels are accurate, but §5's new-code boundary, the R-aff/R-all site caching, the S-static seed pairing, the 7000-block collision, the identity description and the 300 s replay line each need correction before the implementation plan is sized.
