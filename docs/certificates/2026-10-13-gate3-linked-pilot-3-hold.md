# Gate 3 linked pilot 3 (attempt 4) — instrument hold at the primary family, 2026-10-13

**HOLD: instrument-failed at the bottleneck known answer. Software valid,
execution valid. This is the screen's own stop-before-freeze outcome (v4 §5):
the primary headroom family is undefined on this host and module design.
No further attempt is authorised; John must disposition.**

Launched from certificate commit `027a258` with implementation source
`1dfe62f` under protocol identity `gate3-v4.2`, after clean host checks.
Wall 41:29, GNU time exit 2 (the launcher's stop code), peak RSS
1,565,672 KiB. This attempt charged **3272.33 process-seconds**; cumulative
**6206.67 s** of the unchanged **10,800 s** cap; **4593.33 s** remain.
Receipts: [published evidence](../results/gate3-pilot-2026-10-13-linked-3/manifest.json),
[execution summary](../results/gate3-pilot-2026-10-13-linked-3/execution-summary.json).
Sixty child receipts, every return code zero. Both cards idle afterwards;
all four prior custody receipts and the new one intact.

## What passed

- Twelve development hosts (intact and width 4) and then six width-2 hosts
  trained; same-card, cross-card and three-per-card replays bitwise; B3 fit
  replay bitwise; timing admission with three oracle seeds.
- **The retired P3 rungs replayed attempt 3 exactly** on freshly trained
  hosts, so the determinism contract held across the change to the decision
  code, and κ = 2.0 cleared 0.05 nats on every anchor (0.28 to 1.02 nats).
  Magnitudes filled: α-pair {0.1, 0.3}, s_min 0.1, κ 2.0.
- **Post-hoc known answer passed on every anchor.** On P1 at α = 0.1 the
  exact inverse A1 and the affine ridge A3 restore the intact loss to the
  last digit, and the loss-fitted module B3 reaches 109% to 159% of the A3
  gain. B3 does better than restoring the intact loss: it lands 0.55 to 0.58
  nats against intact 0.60 to 0.77. The former arms A7 and A8 are worse than
  the impaired no-op on every anchor (descriptive V2).

## What stopped it

v4 §5: the primary family is defined only if (B3 gain at P6, bypass) −
(B3 gain at P5, post-bottleneck) ≥ 0.05 nats on every development host at
some width in {4, 2}; width 8 is tried only after degenerate width-4
training, which did not occur.

| Anchor | w = 4: P5 gain | P6 gain | P6 − P5 | w = 2: P5 gain | P6 gain | P6 − P5 |
|---|---|---|---|---|---|---|
| 3000 | 0.114 | 0.125 | 0.011 | 0.227 | 0.248 | 0.020 |
| 3001 | 0.060 | 0.062 | 0.002 | 0.243 | 0.260 | 0.016 |
| 3002 | 0.137 | 0.145 | 0.008 | 0.082 | 0.103 | 0.021 |
| 3003 | 0.095 | 0.102 | 0.008 | 0.066 | 0.073 | 0.007 |
| 3004 | 0.046 | 0.049 | 0.003 | 0.037 | 0.037 | 0.001 |
| 3005 | 0.104 | 0.122 | 0.018 | 0.095 | 0.121 | 0.026 |

Every paired difference is below 0.05 at both widths. Oracle plateau was
certified on one of twelve width-4 fits and none of twelve width-2 fits at
the 800-step schedule, so every B3 gain is a lower bound under the frozen
schedule axis.

## Reading

The module's gain is dominated by generic re-tuning of the downstream, not
by information recovery. The same 272-parameter module, fitted by loss
gradient, improves an intact host by about 0.1 to 0.2 nats on P1, and
improves the bottlenecked host by a similar amount whether it reads the
pre-bottleneck tensor or not. Offering the bypass adds at most 0.026 nats.
On this host, a 10-epoch CIFAR-10 CNN with a 1×1 bottleneck of width 4 or 2,
the information the bottleneck destroys is small compared with what a small
loss-fitted module recovers by re-shaping the input statistics the
downstream sees. This is the effect the P8 null calibrator was designed to
subtract, and the screen stopped before P8 was opened. It is a finding
about the instrument's primary family on this host, not evidence for or
against headroom in general, and not a Gate 3 verdict.

## Disposition requested from John

1. **Recommended: accept the stop as the Gate 3 pilot finding** and close
   the task with it: the preregistered primary family cannot be defined on
   this host and module design, so the confirmatory screen is moot and the
   remaining 4593 s stay unspent. The finding is informative in its own
   right: on a conventional host the dominant repair effect of a small
   loss-fitted module is re-tuning, not information.
2. **Bounded diagnostic before deciding on a redesign** (separate
   authorisation, about 600 s on one card): on two anchors at width 2, fit
   P5 and P6 with a longer schedule and with the already-implemented wide
   64-hidden module, to see whether P6 − P5 grows with optimisation or
   module capacity. Descriptive only; it cannot rescue this attempt.
3. **New draft (v5)** with a redesigned primary family, for example a
   bottleneck that removes class-relevant information explicitly, a wider
   module family at S2b, a longer schedule axis, or a known answer stated
   net of the P8 re-tuning effect. That is a new study with its own ceiling.

Confirmation remains blocked regardless. Nothing in this hold changes X, ρ,
Hm, U, the η sets, counts, arms, verdict logic or the seed rules. The Gate 2
verdict is unchanged.
