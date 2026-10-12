# Dispositions of the three v1 reviews, producing draft v2

Reviewed bytes: v1 at `5aeee56`. Reports preserved verbatim:
[alignment](alignment-review-v1-2026-10-13.md),
[scientific](scientific-review-v1-2026-10-13.md),
[reality](reality-review-v1-2026-10-13.md). Every finding is adopted,
adopted in modified form, or disposed with a reason; "v2 §n" cites the
revised draft. Author `sultai-claude-nyx`, 2026-10-13.

## Alignment review (A1–A13)

| # | Finding | Disposition |
|---|---|---|
| A1 | Static, T-more and N_eq not data-matched (module arms see 2,000 fresh examples) | **Adopted, by changing the regime** rather than adding R-all to every host: primary fit data is a fixed 2,000-example subset of the host's own training partition; the held-out conditioning set serves only the P1 instrument and the M-sq-fresh arm (v2 §4). Frontier, static and finishing hosts are now data-matched; the declared residual advantage is selection on 3,000 held-out examples |
| A2 | R-272 is the weakest zero-added route; R-aff only descriptive | **Adopted in modified form**: the primary D1 comparator is the foldable affine module M-aff (zero retained capacity by construction, S2 below); R-aff and a structured 272 re-tune (centre taps plus biases) form D1′ "beats every named re-tune" as a bounded secondary. Random R-272 removed. A per-anchor max over several selection-chosen controls was not adopted as the primary: it is conservative for "present" but makes "re-tuning equivalent" trivially reachable (v2 §5–§7, J-route) |
| A3 | V2 refit budget not matched to M-small | **Adopted**: refit gets the full module grid; M-small-B at twice the schedule is the matched-budget comparator; as-sliced with fold is the strict secondary (v2 §4–§7, J-D2) |
| A4 | Ladder gates M2 behind M1; V2 laundered into Gate 5; D02 near-empty start silently relaxed | **Adopted**: M2 opens independently of M1 (M1 chooses the builder; SGD admissible); V2 is a truncation precedent and Gate 5 is read by M2; the M2 starting point is John's decision J-D02 (STEERING §5–§6; v2 §12) |
| A5 | Attempt-4 narrative overstated; wrong ranges; plateau count | **Adopted**: "every impaired host"; 0.037–0.260 and 0.048–0.207 from the receipts; no unimpaired-host fit was run; 2 of 30 grids (v2 §1, §4; STEERING §3). Hold-record correction placed in the final disposition |
| A6 | Loss chosen as unit without surfacing it; accuracy guard | **Adopted**: J0; dev-accuracy co-read A1 is a condition of "present" with the "calibration, not capacity" label; CONCEPT.md attributes the formalisation to the author (v2 §6–§7, §12; STEERING §4 rule 1) |
| A7 | J-table has the author's decisions and omits John's | **Adopted**: J5 and J8 moved to the implementation plan; J0, J-route, J-D2, J-host, J-retire, J-D02 added; X labelled inherited and a detectability floor (v2 §12) |
| A8 | "The instrument is sound" | **Adopted** (STEERING §3) |
| A9 | Site S1 is a privileged choice | **Adopted** (v2 §1; STEERING §5) |
| A10 | Label M-sq/M-small as the plug-in route | **Adopted** (v2 §1, §3, §5) |
| A11 | V1 on both hosts or drop the 0.5 pair | **Adopted in modified form**: the 0.5 pair moves to the extended set, last in priority; J-host records the primary-host choice (v2 §5, §12) |
| A12 | Overrun rule unstated | **Adopted**: core/extended tiers with a declared drop order; core never dropped; stop before V1 if core projects over (v2 §8) |
| A13 | Size | **Adopted in part**: STEERING §7 shortened; v1 §11's history moved to Appendix A; §13 reduced. v2 §1 keeps one paragraph of evidence because it states the calibration signature the arms are built against |

## Scientific review (S1–S20)

| # | Finding | Disposition |
|---|---|---|
| S1 | BLOCKER: no zero-parameter controls (BN recalibration, temperature, class bias); accuracy co-read | **Adopted**: N0 is the better-on-selection of raw host and R-bn, every arm is fitted on N0, R-temp and R-cal reported; "calibration, not capacity" label (v2 §4, §5, §7) |
| S2 | BLOCKER: R-272 is a straw man; the foldable affine 272 module is the matched control; rate grid; edge-of-grid flag | **Adopted**: M-aff is the primary comparator; structured 272 re-tune (centre taps + biases) and R-aff as D1′; re-tune rates {1, 0.3, 0.1, 0.03, 0.01}; edge-of-grid flag on every selected rate. The 0.003 rate was not added (five rates bound the re-tune cost) (v2 §3–§5) |
| S3 | T-more cannot detect undertraining; add T-cool and M-sq on the T-cool host | **Adopted**: T-cool replaces T-more (resume from saved optimiser state, rate decayed linearly to 0 over 5 epochs); D3 = G(M-sq \| T-cool) is a bounded secondary with "survives finishing" / "harvests undertraining" labels (v2 §5–§7) |
| S4 | Fresh-data confound untested | **Adopted, inverted**: seen data is primary; M-sq-fresh measures the fresh-data effect D4 (v2 §4–§6) |
| S5 | S-static not apples-to-apples; init and weight decay unstated; pairing noise | **Adopted**: anchor seed, W = 0 and b = 0 init (host init bitwise identical), weight decay stated, refit reading added, descriptive at the pilot, confirmatory-only verdict at M_s with the power caveat (v2 §5–§7) |
| S6 | Plateau rule has the wrong sign for a difference of fitted arms | **Adopted**: "present"/"helps" require the control arm certified on ≥ 75 percent of anchors; "re-tuning equivalent" requires M-sq certified; R-all excluded; schedule axis raise-only in the pilot (v2 §4, §7, §9) |
| S7 | V2 compute asymmetry; Scratch-B; lottery arm | **Adopted**: refit on the full grid; M-small-B at twice the schedule defines D2; lottery cut in the extended set (v2 §4–§5) |
| S8 | R-all undercosted; checkpointed single run | **Adopted**: R-all moved to the extended set at one seed, checkpoints every 40 steps to 800, BN recalibrated after; all grids are one checkpointed 3,200-step run per rate (v2 §4, §8) |
| S9 | Confirmatory count versus ceiling inconsistent | **Adopted**: sized on D1 only; D2 and D3 at achieved half-widths; confirmatory arm set named; ceiling 12 GPU-hours proposed with stop-before-freeze (v2 §8–§10) |
| S10 | N_eq: relabel, bound, no extrapolation, drop the ratio | **Adopted**: frontier slope per doubling with seed-bootstrap interval; module gain as a fraction of a doubling; no extrapolation; ratio to 272 removed (v2 §6) |
| S11 | Premise overstated | **Adopted** (v2 §1) |
| S12 | As-sliced needs the exact mean fold; leave-one-out ceiling | **Adopted** (v2 §3, §6) |
| S13 | ExpandNets label misapplied; true analogue M-exp | **Adopted**: label corrected; M-exp in the extended set (v2 §1, §3, §5) |
| S14 | R-all with frozen BN is handicapped; finer checkpoints; book both costs | **Adopted** within the extended-set definition (v2 §5) |
| S15 | Secondary readings are verdicts in disguise; sign of D5 | **Adopted**: each secondary names its bound; D5 sign fixed (negative = insertion better) (v2 §6–§7) |
| S16 | No known answer for slicing | **Adopted** into V0 (v2 §7) |
| S17 | n = 6 statistics; sign test at confirmatory n | **Adopted**: pilot reads SDs only; exact sign-test bound beside the bootstrap at confirmatory n (v2 §6, §9) |
| S18 | X inherited from a repair margin | **Adopted**: labelled a detectability floor (v2 §7, §12) |
| S19 | 20-epoch cooled F(1.0) mandatory | **Adopted in modified form**: T-cool is in the core (the cheaper finished-host answer); the 20-epoch cooled frontier point is extended, priority 5 (v2 §5) |
| S20 | 0.5 frontier hosts doubling as Q1 anchors | **Adopted**: stated in the extended set; they are then development anchors never reused for a starved-host confirmatory (v2 §5) |

## Reality review (R1–R14)

| # | Finding | Disposition |
|---|---|---|
| R1 | "Composition of existing code" false; new-code boundary | **Adopted**: explicit carried-over and new-code lists; each new path needs its own exact-source review before the pilot (v2 §2) |
| R2 | R-aff and R-all re-run the front; cached-site design; R-272 subset absent on the 0.5 host | **Adopted**: stated for R-aff and R-all, booked at 3 × with the pilot measuring first; structured R-272s uses the cached site; the 0.5 host carries only M-sq and M-aff in the extended Q1 pair (v2 §2, §4, §5, §8) |
| R3 | S-static seed pairing self-contradictory | **Adopted**: anchor seed with zero-initialised module tensors removes the RNG question; V0 checks the host tensors bitwise at step 0 (v2 §5, §7, §11) |
| R4 | Replay line 3 × low; process-seconds, not wall | **Adopted**: 900 s booked; ledger in process-seconds; "two cards halve the wall" removed (v2 §8) |
| R5 | 7000–7099 collides with lifecycle seeds; registry incomplete | **Adopted**: confirmatory block 7500–7599, restart 7700–7799; Appendix A adds 2000–2007, 3000–3007 and 7001–7004 (v2 §11, App. A) |
| R6 | Identity paragraph wrong | **Adopted**: composition rule, fields, pins, tests and the no-checkpoint replay stated (v2 §11) |
| R7 | Frontier constructor must preserve the s = 1.0 replay | **Adopted** as a design constraint (v2 §2) |
| R8 | Plateau count and unit | **Adopted** (v2 §4); hold correction recorded in the final disposition |
| R9 | 0.04–0.24 range | **Adopted** (v2 §1) |
| R10 | 32 floor and 16 h ceiling are code | **Adopted**: the 32 floor and inflation are kept as existing code; the ceiling change is labelled code (v2 §9) |
| R11 | T-more resume without optimiser state | **Adopted**: M0 training saves optimiser state; T-cool resumes exactly (v2 §2, §5) |
| R12 | 0.5 host "corresponds to" the starved host | **Adopted** (v2 §3) |
| R13 | Reviews directory did not exist | **Adopted**: created with this record |
| R14 | Add rate 1.0 | **Adopted**: module rates {1, 0.3, 0.1}; 1.0 added and 0.03 dropped relative to v1 (Gate 3's 0.01 had already gone in v1) (v2 §4) |

## Net effect on the question

Three changes decide whether V1 measures capacity rather than calibration
or finishing: fitting on data the host already saw, defining the no-op as
the recalibrated host and fitting every arm on it, and comparing the
nonlinear module against the foldable affine module of equal count. The
rest is bookkeeping, budget and statistics.

## Targeted re-review of v2 (20dee97)

A fresh reader checked every disposition row against v2, the cross-references, the ledger arithmetic, the seed registry and the steering edits. No disposition row was misimplemented; every ledger line reproduced within 30 percent; the registry covers every seed in `src/` and `docs/plans`. Ten findings, all adopted in the commit that follows: D3's baseline restated as the T-cool host's own loss (its "harvests undertraining" label could never have fired under the blanket N0 rule); the static reading's bound corrected to the lower one-sided bound and made conditional on S-static being in the confirmatory set (J7); the accuracy co-read renamed ΔAcc to free the P1 arm name A1; seeds 6018–6023 allocated to the extended 20-epoch cooled point; R-all stated as never plateau-certified; code-change labels on the J6 option-B and J7 ceilings; the plateau comparator and the non-10 epoch path added to the new-code list; the R14 wording above; and the R-all booking's derivation stated. Verdict after the edits: ready to freeze pending John's decisions J0–J10.
