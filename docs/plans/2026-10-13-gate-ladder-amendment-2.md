# Gate-ladder amendment 2 to the hybrid correction protocol (2026-10-13)

**Status: taken by John on 2026-10-13, in his words below, on the two
reviews of the M0 implementation plan. It authorises no execution and
changes no issued verdict.** It restates one row of
[gate-ladder amendment 1](2026-10-13-gate-ladder-amendment.md), whose bytes
are untouched (SHA-256 `af12d474…`, pinned in both M0 freeze certificates).
Everything else in amendment 1 stands.

## The restated row

Amendment 1's Gate 3 row reads, in its prospective column: "a nonlinear
module at a known site must beat every named cheaper route to the same loss
at matched retained parameters, data and budget (M0 V1)". The M0
preregistration's V1 conditions on D1 (the nonlinear module against its
foldable affine twin) alone, with the re-tune and finishing comparisons as
labelled readings, and John clarified the programme's intent:

> "it doesn't have to strictly beat it, technically speaking 'noop with no
> substantial change' or 'same but with fewer parameters' are also valid
> outputs, they're less valuable than 'more with a little bit' but they're
> all strictly valid."

| Gate | Prospective reading after this amendment |
|---|---|
| 3 | **Retired as not established, not passed, and replaced** by a prospective capacity-headroom requirement read by rungs M0 and M1 ([STEERING §5](../STEERING.md)): a nonlinear module at a known site is **compared against** every named cheaper route to the same loss at matched retained parameters, data and budget (M0 V1). The outcomes are ranked and all are valid: more capacity at a small parameter cost (V1 present, which alone opens rung M1) above the same loss with fewer retained parameters (V1 re-tuning equivalent) above no substantial change (indeterminate or null). "Beats every named re-tune" grades the value of a present reading and is not a gate. A builder must then find that capacity more cheaply than SGD (M1). Repair families remain instrument validation only. |

## What this amendment does not do

It does not change V1's thresholds, labels or the rule that rung M1 opens
only on V1 present (M0 preregistration §7). It does not reopen any Gate 3
evidence, change the Gate 2 verdict or any pinned hash, or authorise any
execution. Its SHA-256 is bound into the M0 identity beside amendment 1
(M0 preregistration v4 §11, [freeze certificate 2](../certificates/2026-10-13-m0-freeze-2.md)).
Any later change is amendment 3 with its own date and hash.
