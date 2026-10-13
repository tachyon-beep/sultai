# Exact-source review: supervisor-ceiling-one-per-card-costs (commit e1df6e1)
**Verdict:** ACCEPTED WITH FINDINGS

Scope: `src/sultai_screen/pilot.py` (Supervisor `ceiling` / `per_card` / `module`, plus `Job.extra`), `src/sultai_screen/m0/costs.py`; tests `tests/test_m0_supervision.py::SupervisionTests` (12 tests, all pass under `/usr/bin/python3 -B -m unittest test_m0_supervision.SupervisionTests` from `tests/` with `PYTHONPATH=src`). Constants and call sites outside the listed files (`m0/identity.py`, `m0/pilot.py`, `m0/confirmatory.py`, `contracts.py`) were read only where a claim depends on them. Mutation checks were run on `git archive` copies in the scratchpad, one copy per mutation; the working tree was not touched.

## Claims and evidence

| Claim (prereg) | Source file:line | Pinning test | Verdict |
|---|---|---|---|
| §2 l.127-128 / §8 l.495-497: M0 pilot ceiling is a **new constant**, 4 GPU-hours = 14,400 process-seconds | `m0/identity.py:34` `M0_PILOT_SECONDS = 14_400`; wired at `m0/pilot.py:1090` `Supervisor(root, data, ceiling=M0_PILOT_SECONDS, per_card=1, module=WORKER)`; Supervisor uses it for opening check, budget, deadline, ledger and post-cleanup cap at `pilot.py:144,150,166,284,294` | `SupervisionTests.test_m0_ceiling_and_one_process_per_card` (ceiling stored, ledger records 14,400, opening charge 12,000 accepted and 14,400 refused against the M0 ceiling — fails if `self.ceiling` were not threaded into the checks). Literal value pinned outside the suite: `test_m0_identity.py:90`, `test_m0_pilot.py:413-414` | Satisfied |
| `PILOT_SECONDS` (10,800) untouched; Gate 3 ledgers still checked against it | `contracts.py:47` `PILOT_SECONDS = 3 * 3600` (not in diff); default `ceiling=None` → `PILOT_SECONDS`, resolved at construction (`pilot.py:144`); Gate 3 callers unchanged `pilot.py:694,696`; ledger checks `pilot.py:530,533,585` still compare with `PILOT_SECONDS` | `test_gate3_defaults_keep_the_ledger_ceiling_and_bind_the_patched_constant_at_call_time` (default ceiling == PILOT_SECONDS, ledger 10,800, patched constant binds at call time) | Satisfied (see Minor 3 on byte-level pin) |
| §2 l.127-131 / J7: a confirmatory ceiling constant (12 h = 43,200) with its **own supervisor instance**; 14 h (50,400) only with S-static (§8 l.514-516, §10) | `m0/identity.py:35` `M0_CONFIRMATORY_SECONDS = 43_200`; `m0/costs.py:22` `M0_CONFIRMATORY_STATIC_SECONDS = 50_400`; `project_confirmatory` `costs.py:324-329`, `with_static` `costs.py:332-345` (requires both static arms, records `requires_14h`); own instance `m0/confirmatory.py:475` | `test_confirmatory_projection_from_measured_units` (ceiling field, 33,820 within / 43,920 over); `test_static_projection_requires_fourteen_hours` (50,400 literal, both static arms required). Own-instance ledger ceilings: `test_m0_pilot.py:1250-1251, 1321-1322` | Satisfied |
| §2 l.131 / §4 l.266-269 / §8 heading: **one process per card**, which is what the ceiling counts | `pilot.py:140,145,160` per-card limit parameterised (Gate 3 default 3); M0 pilot `m0/pilot.py:1090` and confirmatory `m0/confirmatory.py:475` pass `per_card=1`; charging per process wall unchanged `pilot.py:270-279` | `test_m0_ceiling_and_one_process_per_card` (two jobs on card 0 hold before any Popen or process receipt; one job per card admitted — fails if `> self.per_card` reverted to `> 3` or became a global cap). Confirmatory schedule per-batch uniqueness `test_m0_pilot.py:1252-1256, 1373-1379` | Satisfied; the pilot call site's `per_card=1` is unpinned (Important 2) |
| Gate 3 child command and hold message unchanged | `pilot.py:179-200` default `module="sultai_screen.screen"`, `extra=()` appended last; message keeps "three" when `per_card == 3` (`pilot.py:163-164`) | `test_child_command_is_unchanged_by_default_...` (token-for-token Gate 3 commands; M0 module and extra args appended); `assertRaisesRegex(... "reviewed maximum three processes/card required")` | Satisfied |
| §8 l.484-493 extended set in priority order: 0.5-host pair, R-all, M-exp, lottery, cooled-20, F(0.25), F(0.75) | `costs.py:122` `EXTENDED_ORDER` | `test_extended_admission_stops_at_the_first_miss` asserts the tuple literally | Satisfied |
| §8 l.488-493 overrun rule: extended admitted in order only while the measured charge projects to fit; stop at the **first** misfit; core never dropped; core alone over ceiling → stop before V1 | `admit_extended` `costs.py:270-283` (break at first miss, `costs.py:280`; empty when core over, `costs.py:275`); `core_over_ceiling` `costs.py:256-258` (strict `>`); consumed by `m0/pilot.py:279-290, 659` | `test_extended_admission_stops_at_the_first_miss` (F(0.25) would fit but is not reached — fails if `break`→`continue`); `test_core_over_ceiling_stops_before_v1` (boundary 14,400 vs 14,400.5 — fails if `>` became `>=`) | Satisfied |
| R23: 5% reserve on extended admission | `costs.py:124` `EXTENDED_RESERVE = 0.05`; `extended_limit` / `extended_fits` `costs.py:261-267` | `test_extended_admission_stops_at_the_first_miss` (0.05, 13,680, and admission at 12,400 + 1,650 = 14,050 ≤ ceiling but > limit → `[]`) | Satisfied (stricter than §8 by ruling, admits less, never more) |
| R19/R23 projection = charged + remaining, no stage counted twice | `project_pilot` `costs.py:236-253` (rejects duplicate stages, charged ≥ 0 finite) | `test_pilot_projection_is_charged_plus_remaining` (3,000 + 1,000 + 1,000 = 5,000; empty remaining = charged; duplicates and unknown stage raise) | Satisfied |
| §8 "kernels per §8 line", projections from measured one-per-card maxima (R9, R17): §10 per-anchor 1,010 s decomposition; S-static +110 s | `ANCHOR_UNITS` `costs.py:87`, `ARM_UNITS` `costs.py:88-117` (coverage enforced `costs.py:118`); `per_anchor` `costs.py:188-190` | `test_confirmatory_projection_from_measured_units` (`per_anchor(CONFIRMATORY_ARMS) == 1_010` at §8 bookings; doubling the measured front unit moves the projection by 32 × 250 — fails if R-aff were a literal or its unit count changed); `test_static_projection_requires_fourteen_hours` (+110) | Satisfied |
| R9: front re-run costed per 1,000 steps (R-aff 16 units, R-all 4); starved pair has no host term | `costs.py:75-83`; `extended_cost("starved-pair")` `costs.py:201-204` | `test_confirmatory_projection_...` (16 × 15.625), `test_extended_admission_...` (starved pair 1,650 = §8 l.485-487; R-all 6 × (4 × 15.625 + 100 × 0.5)) | Satisfied |
| R23: host items as whole children (train + noop + evaluate); R-all by its own terms incl. 100 from-images evaluations | `extended_cost` `costs.py:211-224`; `R_ALL_EVALUATIONS` `costs.py:79` | `test_extended_admission_...` (F(0.25) 6 × (20 + 40), cooled-20 6 × (90 + 45), R_ALL_EVALUATIONS == 100); `test_unmeasured_frontier_widths_...` (0.5 fallback, §8 l.491-492) | Satisfied |
| §8 core table (module-grid cases incl. M-sq-fresh and S-static refit, M-aff, M-small-B, two M-wide→8 refits, R-272s, R-aff, cuts/folds line) | `CORE_CASE_ARMS` `costs.py:128`; `STAGE_WORKLOADS["core"]` `costs.py:146-149`; at §8 bookings = 6 × 1,090 = 6,540 s (§8 core lines 6,360 + 3 × 6 × refit-single for M-wide→{16,4,2}, §5 l.292 / R17) | **None.** `test_pilot_projection_...` only asserts the stage keys exist; `test_m0_pilot.py:763` recomputes `stage_seconds("core", w)` from the code under test | Source correct; **unpinned** (Important 1) |
| I3/R28: measured confirmatory projection = fixed + n × measured per-anchor | `measured_confirmatory` `costs.py:286-302` | `test_measured_confirmatory_projection_...` (700 + 33 × 1,100; n = 31 and zero per-anchor raise) | Satisfied |
| n ≥ 32 (§9 l.540, J7) | `CONFIRMATORY_MINIMUM = 32` `costs.py:20`, enforced `costs.py:288, 306` | n = 31 raises in both projection tests | Satisfied |
| Carve-out: costs off the retired import path | `costs.py` imports only `sultai.trust`, `contracts`, `m0.identity`, `m0.stats` | `test_costs_import_isolation` (screen, pilot, former_bridge, alignment absent from `sys.modules`) | Satisfied |

## Frozen values checked

| Value | Prereg | Source | Match |
|---|---|---|---|
| M0 pilot ceiling | 14,400 s (§8 l.495) | `m0/identity.py:34` 14_400 | Yes |
| Gate 3 `PILOT_SECONDS` | 10,800 s, untouched (§2 l.128, §8 l.496) | `contracts.py:47` 3 × 3600; not in diff | Yes |
| Confirmatory ceiling | 12 h = 43,200 s (§8 l.514, J7) | `m0/identity.py:35` 43_200 | Yes |
| Confirmatory ceiling with S-static | 14 h = 50,400 s (§8 l.516, §10 l.563) | `m0/costs.py:22` 50_400 | Yes |
| Processes per card (M0) | 1 (§4 l.266-269, §8 heading) | `m0/pilot.py:1090`, `m0/confirmatory.py:475` `per_card=1` | Yes |
| Gate 3 processes per card | 3 (unchanged) | `pilot.py:140` default 3 | Yes |
| Development anchors | 6 (3000–3005, §9 l.522) | `costs.py:19` 6 | Yes |
| Confirmatory minimum n | 32 (§9 l.540) | `costs.py:20` 32 | Yes |
| Extended order | 7 items, §8 l.484-492 | `costs.py:122` | Yes |
| Re-tune rates (R-aff/R-all) | five | `m0/identity.py:39` five rates | Yes |
| R-aff last checkpoint | 3,200 steps | `CHECKPOINTS[-1]` = 3200 → 16 units | Yes |
| R-all | 800 steps, checkpoint every 40 (§5 l.315), "five rates, 4,000 steps" (§8 l.484) | `RALL_CHECKPOINTS` 40..800 → 4 units, 100 evaluations | Yes |
| Extended reserve | none in prereg; 5 % by ruling R23 (cited in the brief) | `costs.py:124` 0.05 | Ruling, stricter than §8 |

No frozen value needs to change for code and document to agree.

## Findings

**Important 1 — core stage workload is unpinned.** `src/sultai_screen/m0/costs.py:128,146-149`. What: no test asserts the content or total of `STAGE_WORKLOADS["core"]`; `test_pilot_projection_is_charged_plus_remaining` checks only that the keys exist and `test_m0_pilot.py:763` recomputes the value from `stage_seconds` itself. Mutation on its own scratch copy — `CORE_CASE_ARMS = PILOT_ARMS - {"T-cool", "M-sq-fresh"}` (drops two §8 module-grid cases, −720 s at bookings) — passes all 60 tests in `test_m0_supervision` and `test_m0_pilot` (no other test module references `m0.costs`/`m0.pilot`). Why: the remaining-core figure feeds `core_over_ceiling` (§8 "stop before V1") and `admit_extended`; an under-count lets extended work be admitted, or the pilot proceed, when the core does not fit, and a stop during the core burns the single attempt. Fix: add `assertEqual(stage_seconds("core", workloads()), 6_540.0)` with the derivation in a comment (§8 core lines 6,360 + 6 × 3 × 10 for the M-wide→{16,4,2} refits), and pin `CORE_CASE_ARMS` against the §8 module-grid list.

**Important 2 — pilot's `per_card=1` at its call site is unpinned.** `src/sultai_screen/m0/pilot.py:1090,1105` (outside the listed source; the claim depends on it). What: the single mutation `per_card=1` → `per_card=3` at `m0/pilot.py:1090`, on its own scratch copy, passes all 60 tests in `test_m0_supervision` and `test_m0_pilot`. The Supervisor guard itself is pinned, but nothing pins that the pilot asks for one per card. The opening receipt writes the literal `"processes_per_card": 1` instead of the supervisor's value. The confirmatory has a behavioural per-batch uniqueness check (`test_m0_pilot.py:1252-1256, 1373-1379`), the pilot has none. Why: §4 l.266-269 fixes one process per card for the pilot because co-tenancy inflated a 44.5 s host to about 102 s. The guard is the only enforcement if a schedule change places two pilot children on one card. Without it the process-seconds charge would inflate about 2.3× against the 14,400 s hard stop and could burn the single attempt. This is the same criterion as Important 1: a frozen §8 claim whose violation no test catches. Fix: assert `per_card == 1` on the pilot's Supervisor (or add the per-batch uniqueness loop to the pilot launch test) and record `run.supervisor.per_card` in `00-opening.json`.

**Minor 1 — Supervisor accepts any `per_card` and any `ceiling`.** `src/sultai_screen/pilot.py:144-145`. What: no guard that `1 ≤ per_card ≤ 3` (the reviewed maximum) or that `ceiling` is finite (`inf` passes the opening check). Current callers pass only 1, 3 or the frozen constants, so nothing reachable is wrong. Fix: raise `PilotHold` outside those bounds.

**Minor 2 — `STAGE_WORKLOADS["hosts"]` and `["V0"]` don't follow R23 and aren't used in production.** `src/sultai_screen/m0/costs.py:140-150`. What: "hosts" costs the 0.5/1.5 hosts as train child + in-process evaluation kernel, not R23's three whole children, and omits §8's six 1.0 reference hosts, which ruling R17 (progress.md:90) puts inside `fixed_v0` ("total charge at the end of stage 2"). Production `admission` (`m0/pilot.py:279-290`) passes only the core (separately) and admitted extended items to `project_pilot`, so these lines are never projected; the only user is a test (`test_m0_pilot.py:750`). Why: a later caller projecting from stage 1–3 would under-cost. Fix: remove the two entries or cost "hosts" as whole children, with a comment that the reference hosts are in V0.

**Minor 3 — Gate 3 ledger byte-identity is pinned by value, not bytes.** `tests/test_m0_supervision.py` (Gate 3 defaults test) / `src/sultai_screen/pilot.py:144,284`. What: `assertEqual(..., 10_800)` also passes for `10800.0`, and the Gate 3 ledger checker (`pilot.py:530,585`, `!= PILOT_SECONDS`) would accept it too. A `float(...)` coercion of `self.ceiling` would change the Gate 3 receipt bytes (`10800` → `10800.0`) with no test failing. Today the int passes through unchanged, so the bytes are identical. Fix: assert `type(...) is int`, or compare the raw ledger text.

**Minor 4 — `M0_CONFIRMATORY_STATIC_SECONDS` is defined outside `m0/identity.py`.** `src/sultai_screen/m0/costs.py:22`. What: its siblings (14,400 and 43,200) live in `identity.py` and are pinned together at `test_m0_identity.py:90`; the 50,400 constant lives in `costs.py`. It is still pinned by literal at `test_m0_supervision` (static test) and `test_m0_pilot.py:1322`. Fix (optional): move it beside the other ceilings and add it to the identity pin.

**Unrequested behaviour (brief item 4): none found.**
- No extra arms: `ARM_UNITS` equals `KNOWN_ARMS` (`costs.py:118`), and the projections reject arms outside the §10 set (`costs.py:326-328, 339-342`).
- No randomness: `costs.py` has no RNG and no seeds.
- No changed RNG order: the Supervisor change touches no seed or RNG path.
- No Gate 3 receipt byte changed:
  - the default child command is unchanged token for token (pinned);
  - `process.json` fields are unchanged (`remaining` is still the ceiling less the charge, and equals `PILOT_SECONDS` less the charge by default);
  - the ledger ceiling is still the int `PILOT_SECONDS`;
  - the hold message is unchanged at `per_card == 3`;
  - `Job` is never serialised (`asdict`/`fields` appear nowhere on the Gate 3 paths), so the new `extra` field writes no byte.
- Extras that are required: `module` and `Job.extra` route children to the M0 worker (§2 "new worker actions"); the 5 % reserve and `measured_confirmatory` come from rulings R23 and I3/R28, both named in the brief.

Counts: BLOCKING 0, Important 2, Minor 4.

## Dispositions (controller, 2026-10-14)

- Important 1: FIXED (item C10). Important 2: FIXED (item B5).
- Minor 2: FIXED (item B6). Minor 3: FIXED (item C11). Minor 4: FIXED (item B7). Minor 1: deferred, recorded (Gate 3 `Supervisor` file; bounds guard not added to keep the Gate 3 surface untouched).

Items marked FIXED land in the post-review commit 978850a, rechecked in `docs/reviews/m0-capacity/implementation/post-review-recheck.md`; the review above was taken at e1df6e1. Deferred items are recorded in the SDD ledger and carry no verdict role.
