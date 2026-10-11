# Independent final bounded-pilot acceptance

Accepted implementation: `4936d28c90e9ab3a4e7358c772ae0b7acbaffa54`.
Source aggregate: `74f1c403562d13fa37c724ab6ca358642dbaf025c265a82598303c60f4c086a8`.
Frozen source protocol: `7668d20ad55c3a62384975a7cf1f3989dbe69bf701ce02bc9c68ac2601d0d42a`.
Reports below are the final reviewer responses, with relative source links.
Reviewers remained independent/read-only; root was the sole implementation writer.
Initial blocks and the exact repair/recheck dispositions remain in adjacent files.

## Scientific specialist

**ACCEPT the bounded development-pilot source. The remaining scientific blocker is fixed.**

`cheap_rate_decision` (`src/sultai_screen/pilot.py:424`) chooses η_B solely from P6, retains failed P5 fixed-rate fits as explicitly unavailable, and preserves fatal checking for evaluated selected-candidate nonfiniteness. No rate fallback occurs. The regression test (`tests/test_screen.py:796`) covers both cases.

Accepted commit: `4936d28c90e9ab3a4e7358c772ae0b7acbaffa54`
Source aggregate: `74f1c403562d13fa37c724ab6ca358642dbaf025c265a82598303c60f4c086a8`

Saved evidence reports **20 passing quality checks and 137 CPU tests in 26.391 seconds**. Frozen protocol hash remains unchanged.

Read-only narrow closure check; no edits, imports, tests or jobs performed. Acceptance covers the bounded pilot implementation; GPU replay and scientific outcomes remain unmeasured.

## PyTorch and operations specialist

**ACCEPT the bounded development pilot at `4936d28c90e9ab3a4e7358c772ae0b7acbaffa54`.** The exact remaining finding is fixed.

`cheap_rate_decision` (`src/sultai_screen/pilot.py:424`) chooses η solely from P6 selection losses. A failed P5 fit is explicitly retained as `fit-unavailable`, requiring unevaluated development evidence and no loss. It neither changes η nor triggers the selected-evaluation stop. Finite selected candidates still stop on nonfinite development loss.

The added six-anchor regression test covers the reported failure, preserves η=3 despite another available P5 rate, and checks the selected-P6 nonfinite contrast.

Saved evidence reports 20 quality checks across 23 production modules and 137 CPU tests passing, bound to source aggregate `74f1c403562d13fa37c724ab6ca358642dbaf025c265a82598303c60f4c086a8`.

This was a read-only narrow closure check; no tests, imports, jobs, or edits. Earlier findings remain resolved. Acceptance covers the bounded pilot with its existing admission gates; GPU determinism remains a required pilot measurement, and confirmation is not cleared.
