# M0 decision synthesis for John (2026-10-13)

Twelve reports reconciled against source at fdef5fc; validator corrections govern. J0, J9, J10 are John's authority; recommendations are input only. "draft" = `docs/plans/2026-10-13-capacity-frontier-screen.md`.

## 1. Decision table

| ID | Recommendation | Agent(s) | vs §12 | Strongest for | Strongest against | Conf. |
|---|---|---|---|---|---|---|
| **J0** (John) | Split: critic, loss sole primary, ΔAcc a labelled secondary; statistician, loss primary with the guard kept | J0 critic; statistician, validator | critic differs; statistician matches | Guard is new code absent from §2, unsized, inconsistent with V2 and M1–M3; John withdrew the accuracy mandate | Guard binds only on pure-calibration gains and resolves at n = 32 (hw ≈ 0.145 pts, SD assumed); loss-only "present" needs no decision to change | medium |
| **J1** | X = 0.02, relabelled "smallest margin resolvable at n = 32 for ρ ≥ 0.9, and the equivalence bound" | statistician, validator | match (label) | n* ≤ 32 iff SD(D1) ≤ 0.0275 (`contracts.py:43,185`); any lower X bets on ρ ≈ 0.99 transferring from impaired hosts | A true 0.015-nat premium is published "re-tuning equivalent"; "exactly 32" is a 5-df coincidence (n* 13–190) | medium |
| **J2** | X₂ = 0.01 | statistician, validator | match | D2 is the closest analogue of P6 − P5 (SD 0.006–0.010); ≈ 3 half-widths | D2 unsized; indeterminate at ρ = 0.9; refit variance unmeasured | medium |
| **J3** | M_s = 0.02; S-static confirmatory only if pilot SD(D5) ≤ 0.0275 | statistician, validator, J7 | match + rule | Two regimes (SD 0.086 vs 0.018); only the pilot tells which | Conditional arm invites "dropped because it looked bad"; √2 assumes independent endpoints | medium |
| **J-route** | Keep M-aff; add measured fold residual (tolerance X/4), five-rate grid for M-aff, R-272s redefined to a subset that exists | design reviewer | match (contract) | Same `Adapter`, one flag (`fit.py:21-29`): single-factor contrast; W-fold exact to 1e-6 | 23.4 % border ring keeps a 16-dof bias; no affine SGD path (`fit.py:159-168`) | high/med |
| **J-D2** | Keep matched-budget pair; name the unit (steps + passes; 4.3× parameter-step ratio disclosed); name M-small-B checkpoints 1,600 vs 6,400 | tech-selection critic | match | Wall/step flat in width (4.68 vs 4.64 ms, `01-timing-replay-decision.json`); zero cost | Search asymmetry (18 vs 9 selection events) survives any budget unit | med-high |
| **J-host** | Reference host primary; enlarge the 0.5 pair to H3 (+T-cool, +M-sq\|T-cool) and run it first | leverage analyst | match on host; differs on order/content | Only 1.0 has hashes and a prior; as drafted the 0.5 pair never runs under A (draft:332-337) | Displaces R-all under A; 1,320 s at bookings (the 600–750 s FLOP figure is undercut by J4's launch-bound finding) | medium |
| **J4** | Keep {0.5, 1.0, 1.5}; 20-epoch ahead of F(0.75); name its decay rule | plan-review reality | match on sets; order differs | 0.75 buys 3 % slope precision for 210 s | 1.5 at 10 epochs may be more undertrained than 1.0 | medium |
| **J6** | B, 14,400 s as a new constant; leave `PILOT_SECONDS` (Gate 3 ledgers check it, `pilot.py:514,517,569`) | program architect | **differs** (A) | Stop is hard and non-resumable (`pilot.py:150-162,278`); A leaves 2,000 s against an unmeasured 3× front kernel | Ceiling binds nothing at nominal (11,900/14,400) | high |
| **J7** | §10 set, n = max(32, n*), 12 h hard stop, S-static out unless 14 h; add fixed V0 term | program architect, validator | match + exclusions | n = 32 without S-static survives 1.38× stress; with it 1.24× | D5 never becomes a verdict; no confirmatory ceiling exists in code | med-high |
| **J-retire** | Retire as §2; P1 stays; "retired" = no verdict/report/confirmatory role **and** P1-only workers, checks, cost projection, tests are new code | shaping; reality | match (mechanism corrected, §2 below) | Every retired item is repair; P6 − P5 ≤ 0.026; M-sq-fresh + V0 replay are the bridge | Carve-out is code, not prose; loses full-rank/bias-only known answers | high |
| **J-D02** | Fix the principle now (0.5 is not near-empty absent an explicit D02 relaxation; near-empty = s ≤ 0.25); number at M2 draft; five M0 recordings | shaping architect | differs in degree | Alignment review already caught drift to the 30 % host; costs nothing | s ≤ 0.25 trainability unmeasured; 0.5 is M0's only hashed starved host | medium |
| **J9** (John) | Separate dated, SHA-pinned, prospective amendment to the 2026-10-09 protocol; bytes untouched; bound into M0 identity; seven gates mapped | J9 critic; J9 Codex | form, not substance | Frozen file is SHA-pinned (`correction.py:23-24`) and cannot point forward; §6 "closed"/"unchanged" misstate certified status | One more file; D26 + README pointer could carry it | med-high |
| **J10** (John) | John's authority; no recommendation sought | — | — | Bears on it: J9 critic "nothing is durable until merged"; Codex: merge authorises no execution; J9's identity change and J-D02's seed block precede the M0 SHA pin | — | — |

## 2. Conflicts resolved against source

**J-retire shaping vs reality: reality governs.** "P1 runs through `posthoc_worker` unchanged, retired families land as unread bytes" is false: `screen.py:54,236-247` ladders and runs P1/P2/P3; `pilot.py:288-319` raises `PilotHold` on P3 replay mismatch and `InstrumentHold` if P2 or P3 exhausts; `pilot.py:322-338` requires former agreement and all four families before reading B3 ≥ ρ·A3; `costs.py:44-57` projects the Gate 3 arm set; `tests/test_screen.py:135,299,339,378-384,748-767` and all of `test_v42_ladder.py` pin retired families. The decision stands; the mechanism is a code carve-out, so draft:60-69 ("P1 carried unchanged") is wrong and the 56 s P1 line (draft:323) inherits the whole posthoc wall.

**Extended order (J-host, J-D02, J4, J6): J6-gated, not a four-way conflict.** Booked order R-all 900, M-exp 360, lottery 360, F(0.75) 210, 20-epoch 570, 0.5 pair 720 (draft:210-217, 332-337). Under B (5,600 s headroom) J4's reorder, J-D02's F(0.25) swap (~150 s) and J-host's H3 (+600 s) all fit at nominal (≈ 12,700 s). Under A (2,000 s) H3 at bookings (1,320 s) and R-all (900 s) cannot both run, and F(0.75) and the 20-epoch point never reach the line, so J4's reorder is moot. **Under A: split, John decides.** R-all first: cheaper-route coverage of V1 (J-retire's rule-2 argument). H3 first: two-host reading of the undertraining loop.

**J9 critic vs Codex: the same recommendation.** Both: separate, dated, SHA-pinned, prospective amendment to `docs/plans/2026-10-09-correction-protocol.md`; bytes untouched; Gate 3 "not passed"; Gate 5 read by M2; gates 4/6/7 not "unchanged"; SHA bound into the M0 identity. Differences are additive: the critic gives the seven-row form and README/D26 pointers; Codex adds independent review and reserving M2's trajectory/immediate/final removal controls. Both cite a real gap: draft:402-408 drops `amendment_sha256`; `contracts.py:42` is the v4.1 precedent.

**J0 critic vs statistician/validator: split, John decides.** Source settles the shared facts (no accuracy code in `src/`; no evaluator at draft:73-87; guard conditions "present" at draft:270-271; V2 at draft:290-295 and M1–M3 at STEERING:104-107 are loss-only; §9 step 6 sizes D1 only) but not the value call. Case A: loss sole primary, ΔAcc labelled, ladder consistent. Case B: keep the guard, size it under J7, add the evaluator to §2, for an accuracy claim in print.

## 3. Dependencies and order

J0 ↔ J1–J3: guard kept re-expresses margins in points and multiplies n. J-D2 ↔ J2: X₂ is sized for the matched-budget pair only. J4 ↔ J6: any extended host point moved into core (+780 s) breaks A. J-host ↔ J4/J6: H3 runs first-under-A or anywhere-under-B. J-D02 ↔ J4: F(0.25) replaces F(0.75) only on a near-empty lean. J-route ↔ J1 (fold tolerance X/4) and J7 (R-272s redefined before §10). J3 ↔ J7: one decision stated twice. J9 ↔ J10: identity change and amendment SHA must exist before the M0 SHA is pinned and merged.

**Order.** Two independent tracks (J6 confirms J1/J2 do not touch the pilot charge), then J9 → J10. Track 1: J0 → J1 → J-D2 → J2 → J3 → J-route → J7. Track 2: J6 → J-retire → J4 → J-host → J-D02.

## 4. Flags

- **Against the §12 default:** J6 (B). Partial: J0 critic (guard), J-host (order, content), J4 (order), J-D02 (principle), both J9 (form).
- **Options the brief did not ask for:** J-host's H3 arms (not in §5) and J-D02's F(0.25) (not in §3); reasoned, but new arms.

## 5. v3 worklist: defects in the draft

| Defect | Where | Report |
|---|---|---|
| "Folds exactly"/"272 → 0" overstated; exact in the interior, 16-dof bias on the 23.4 % border ring (draft:118 already says "approximately at the border"; §1 and §5 do not) | draft:35-36, :195 | J-route |
| R-272s does not exist: `tail[0]` is Conv2d(16, 64, 3) (`host.py:52-53,63`): 1,024 centre taps, 64 biases | draft:196 | J-route |
| "33h+16 at h=8 = 272": **not a draft defect**; draft says 280 (draft:120, :202-205); caller-side conflation | — | J-route |
| M-small-B plateau checkpoints unstated | draft:166, :205 | J-D2 |
| Accuracy evaluator absent from §2 while ΔAcc conditions V1 | draft:73-87, :222, :270 | J0 critic, validator |
| No confirmatory ceiling enforced; 16 h `within_ceiling` only, two sites | `pilot.py:352-353, 825-826` | J7, validator |
| §8 kernels are 1-per-card; supervisor allows 3 (`pilot.py:146`); co-tenancy hosts 102 s vs 44.5 s (ledger-044) | draft:307-346 | J6 |
| `main()` refuses seeds outside 3000–3005 | `screen.py:443` | J4 |
| No attempt-4 host hash pinned in `src/`; `verify_replay` compares two fresh receipts | draft:93-95 | J4 |
| §2 claims P1 carried unchanged; entangled with P2/P3/A7/A8, can hold the pilot | draft:60-69, :323 | reality |
| 20-epoch point has no decay rule (`train_host` constant lr, raises on epochs ≠ 10, `host.py:129-131`); `_training` checks only `width` (`screen.py:133`); 0.5 host booked 25 s but launch-bound (~35 s); §6 silent on whether F(0.75) enters the slope | draft:214, :321, :242-246 | J4 |
| No affine SGD path; M-aff grid may be off-scale | `fit.py:159-168` | J-route |
| Confirmatory projection omits a fixed V0 term (n = 44 only if < 740 s) | draft:348 | J7, validator |
| Identity drops `amendment_sha256` | draft:402-408 | J9 both |
| Appendix A reserves no M2 starting-host block | draft:438-449 | J-D02 |
| "Matched budget" unit unnamed | draft:235, :290; STEERING:104 | J-D2 |
