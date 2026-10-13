# Independent scientific candidate review

Reviewer: `/root/gate3_science_panel`, Astra; read-only final-candidate pass.
Snapshot: `ff4744ae662bf94e9f60ef8a5a728dca1461f4ef`.
The report below is preserved from the reviewer response; line numbers refer
to that snapshot. No tests or experiments were run by the reviewer.

---

**BLOCK the bounded pilot candidate pending two numerical-policy fixes.** Reviewed commit `ff4744ae662bf94e9f60ef8a5a728dca1461f4ef`; frozen protocol hash matches.

1. **B3 incorrectly requires both schedules to succeed.** `fit.py:263` calls `choose()` separately for 200- and 800-step candidates. One entirely failed schedule therefore aborts an otherwise usable oracle bank. Select the overall finite winner independently; report an unavailable schedule comparison as **plateau uncertified**. If all candidates fail, retain their complete reports and costs in the unavailable-arm evidence before holding; the current failure receipt (`screen.py:377`) loses that bank.

2. **An unselected cheap candidate can trigger a forbidden stop.** `screen.py:308` applies fatal development-finiteness checking to every selection-finite rate. Frozen §9 reserves this stop for selected candidates and no-op evaluations. Record nonfinite development outcomes explicitly for unselected rates; enforce the stop when the candidate is actually used by η_B or per-anchor B2′. Development finiteness must never influence selection.

Required validation: controls for either schedule failing while the other succeeds, all-bank failure with retained evidence, and unselected versus selected cheap-candidate development nonfiniteness.

No other blocking scientific departure found in this pass. Host/sites, impairment operators, exact inverse, residual ridge, paired former, oracle/cheap definitions, magnitude ladders, width-8 hold, ordered decisions, SD-before-median and uncapped sizing are faithful. The cost projection explicitly includes B4/B5/P8 and cheap diagnostics; overlapping charges are conservative engineering choices. B4’s permuted fixture and B5 fits supply timing evidence only.

Saved evidence reports **133 CPU tests passed in 25.961 seconds**, plus passing strict typing, policy and mutation controls, bound to source aggregate `574f947eae3bbde42c69f626928288ad280aa599af173de7a161bd2735f4682b`. I inspected source, tests and saved evidence; ran no tests, imports, jobs or GPU work and modified nothing. This review grants no confirmation authorization.
