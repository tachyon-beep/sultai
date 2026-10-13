# Independent PyTorch and operations candidate review

Reviewer: `/root/gate3_torch_ops_panel`, Astra; separate read-only pass.
Snapshot: `ff4744ae662bf94e9f60ef8a5a728dca1461f4ef`.
The report below is preserved from the reviewer response; line numbers refer
to that snapshot. No tests or experiments were run by the reviewer.

---

**BLOCK bounded pilot at `ff4744ae662bf94e9f60ef8a5a728dca1461f4ef`.** Three concrete defects require bounded fixes.

1. **P1 — The documented output location makes the first worker reject its own run.** Protocol line 468 requires `docs/results/gate3-pilot-…/`. `pilot.py:418` creates admission/output files there, and line 189 creates stdout/stderr before launching the child. That path is not ignored. `screen.py:53` rejects every untracked file during child provenance checking. Consequently, the first worker fails after the durable attempt has already been consumed. Use an explicitly owned, excluded runtime subtree or an ignored staging location with a defined publication step; retain strict source coverage.

2. **P1 — One failed oracle schedule aborts an otherwise available arm, losing its candidate evidence.** `fit.py:263` independently selects from the 200-step and 800-step banks before selecting across the complete bank. If either schedule is wholly nonfinite, `OracleUnavailable` aborts even when the other schedule has finite repairs. Such an arm should remain selectable but plateau-uncertified. When the entire bank fails, `screen.py:377` publishes only an exception receipt, losing the completed candidate bank and its costs. Preserve complete candidate evidence in both cases.

3. **P1 — Raw cheap candidates receive the selected-candidate fatal rule.** `screen.py:308` calls `mean_loss` for every candidate finite on selection. A candidate that becomes nonfinite on development therefore stops the pilot even if it would never be selected. Record unselected numerical failures, then enforce the fatal rule when choosing the relevant candidate; retain the no-op fatal rule.

**Additional missing coverage:** `pilot.py:447` replays host training with three co-tenants per card, but all B3 fit replays at line 452 are serial. Protocol §9 item 3 calls for replay under confirmatory co-tenancy. Add the fit replay under that load, or explicitly resolve whether the frozen clause was intended to cover hosts alone.

The inspected tensor shapes, frozen BN/parameter ownership, chunked mean-gradient accumulation, pinned-data decoding, checkpoint state validation, conservative aggregate process-time accounting, exclusive attempt receipt, and ordinary owned-child cleanup otherwise support a bounded pilot. GPU replay agreement and actual timings remain pilot measurements, not prerequisites requiring speculative GPU tests.

Checks performed: source/test/document inspection, full commit resolution, clean-tree inspection, and independent protocol SHA-256 verification. Saved reports record 20 quality checks across 23 modules and 133 CPU tests against aggregate `574f947eae3bbde42c69f626928288ad280aa599af173de7a161bd2735f4682b`; I did not rerun them. No files, tracker state, workloads, or experiments changed.
