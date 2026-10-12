# Linked pilot 2 (attempt 3) — costed proposal for John's go

Status: **proposal; nothing here is authorised or executed.** The launcher
refuses to start linked pilot 2 until a review certificate names John's
disposition document `docs/certificates/2026-10-13-gate3-linked-pilot-2-disposition.md`,
which does not exist. Creating that document is John's decision.

Governing text: [protocol amendment v4.1](2026-10-13-gate3-headroom-screen-v4.1-amendment.md)
§6, over the unchanged [v4 base](2026-10-10-gate3-headroom-screen.md) §9.

## What the pilot does

The same development pilot as attempts 1 and 2, on fresh hosts, under the
v4.1 evaluator and identity:

1. Fresh training of six intact and six width-4 development hosts (seeds
   3000–3005), per-host timing and peak memory, bitwise replay on the same
   card, the other card and under three-per-card co-tenancy, B3 fit replay,
   timing admission and the seed-count decision (items 1–3).
2. Materiality and known answers on development anchors: exact inverses,
   A1 gains along the ladders, B3 versus A3 on P1, P6 − P5 known answer at
   width 4 then 2 then 8 (item 4). The twin assertion runs here on every
   anchor; under v4.1 every evaluation chunk has the same shape.
3. η_B on development P6 anchors (item 5); per-anchor SD of H′, the count
   rule, then the development medians for the record (item 6).

Stops, ordering and permitted fills are unchanged from v4 §9. Any stop is a
hold that returns to John; no automatic retry.

## Cost and allowance

| Quantity | Process-seconds |
|---|---|
| Cap, unchanged, shared across linked attempts | 10800 |
| Carried charge (attempts 1 and 2 plus the diagnostic) | 1473.27 |
| Remaining before this pilot | 9326.73 |
| Attempt 2 measured through timing admission (12 hosts, replays, co-tenancy) | 1466.98 |
| Rough projection for the remaining stages, see below | 1500 to 2500 |

Measured in attempt 2: one development host trains in 40 to 48 s on one
card; a B3 fit takes about 1.0 s at 1× and 3.3 s at 4×; the wide grid 1.9 s.
One oracle is 24 fits, about 50 s. The remaining stages run roughly 18
oracles (P1 on six intact anchors, P5 and P6 on six bottleneck anchors),
the cheap banks, the post-hoc arms and the evaluations, plus data loading per
worker. The projection is an order-of-magnitude figure from attempt 2's
timings, not evidence; v4.1 §6 requires every timing to be remeasured.
If the pilot ran to the cap the supervisor would stop it; the projection
leaves more than half the remaining allowance unused. The consequence of
cap exhaustion must be understood when deciding: the stop is recorded as
execution-invalid, the attempt is spent with no further attempt authorised,
and every later stage (post-hoc arms with their oracles, bottleneck oracles,
a possible width-2 ladder, the cheap banks) has never been ledgered on this
host, so the figures above are the only guide.

## What the disposition document must say

John's document, transcribed by whoever writes it, must state: approval of
exactly one linked pilot 2 under v4.1; fresh hosts, no checkpoint inheritance;
the carried charge 1473.267264504917 s and the unchanged 10,800 s cap; that
the three prior custody receipts remain byte-preserved; and that no
confirmation, further attempt or merge is granted. The review certificate for
the launch then lists that document in its evidence with its hash, alongside
the quality, runtime and independent-review evidence of the exact source
commit.

## Launch sequence once John has decided

1. Independent exact-source acceptance of the v4.1 package recorded under
   `docs/reviews/gate3-headroom/` and a bound admission certificate committed.
2. Fresh read-only host check: both pinned GPU UUIDs idle, no screen process,
   no compute process, the three custody receipts unchanged.
3. `python3 -B -m sultai_screen.pilot --output runs/gate3-pilot-<date>-linked-2 --data-root <cifar-root> --review-certificate <certificate> --linked-pilot-2`
   under GNU time, raw output in ignored `runs/`, sanitised evidence published
   to `docs/results/` afterwards.
4. On a clean pilot: fill the permitted values, commit the filled protocol,
   and return to John for the confirmatory go under D-P.

## Not proposed

Checkpoint inheritance from attempt 2 (saves roughly 560 s of a 9,327 s
allowance at the cost of a reviewed provenance bridge), any change to
thresholds, counts or seed rules, and any confirmatory execution.
