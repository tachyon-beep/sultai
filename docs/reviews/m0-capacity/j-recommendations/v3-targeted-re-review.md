# Targeted re-review of M0 draft v3 (5c664a1)
Agent: plan-review-reality   Date: 2026-10-13   Files: draft sha256 f5bf2168…, amendment f66046c0… (both verified at HEAD 5c664a1). Line numbers are `cat -n` of the draft unless prefixed.

## Check 1. V3-CHECKLIST rows, SYNTHESIS §1 decisions, §5 defects

| Item | Verdict | Evidence | Change |
|---|---|---|---|
| J0 | IMPLEMENTED | §5:266–267, §6:328–329, §7:377–378, 392–395, §12:580 | none |
| J1, J2, J3 | IMPLEMENTED | §7:380–390, 411–416, 403–407; §9:513–516; §10:535–537; §12:581–583 | none |
| J-route | IMPLEMENTED | §1:41–43, §3:177, §4:222–225, §5:276–277, §6:324–327, §7:373–375 | none |
| J-D2 | PARTIAL (low) | §4:242–252 names the unit and 4.3×/7.6×; §6:341–342 says "at matched budget (§4)"; §7 V2 (409–417) never names the unit; STEERING §5 M0 row:113 says "matched budget" undefined | §7:411: "median D2 ≥ X₂ (matched optimiser steps and example passes, §4)" |
| J-host | IMPLEMENTED | §5:295–305, §7:423–425, §12:586 | none |
| J4 | IMPLEMENTED | §3:152–168, §5:312–317, §6:349–356, §12:587 | none |
| J6 | IMPLEMENTED, derived numbers STALE | §8:475–483 (see check 2) | see finding 2 |
| J7 | IMPLEMENTED, one source claim MISIMPLEMENTED | §2:123–125, §8:485–495, §9:513–518, §10:530–539 | see finding 3 |
| J-retire | IMPLEMENTED | §2:108–119, 130–138; §8:448 | "six tests" at 115 not independently verified (synthesis cites five locations) |
| J-D02 | IMPLEMENTED (five recordings: §11:552, 555–557; §2:82–84; §3:168; §3:192–193) | §3:154–155, §5:313–315, §12:591 | §3:154 and §5:313 say the frontier "reaches below" the near-empty start; at s ≤ 0.25 F(0.25) reaches it, not below. Replace "below" with "to" |
| J9 | IMPLEMENTED | amendment file; STEERING §6 diff; §11:559–568; §12:592; README, D27 | see check 6 |
| J10 | IMPLEMENTED (pending) | §12:593 | none |
| Defects 1–16 | IMPLEMENTED | fold §1:41–43/§3:177/§5:276; R-272s §5:277; M-small-B §4:231,242; evaluator §2:104–106; conf. ceiling §2:123–125/§10:537–539; one-per-card §4:257–261/§8:429/§9:508; seed admission §2:89; hash pins §2:87–88/§3:143–145; P1 §2:77–78,116–119/§8:448; decay rule §3:164–168, width check §2:90–91, 35/85 s §8:446; affine SGD §2:96–97; fixed V0 §8:488–490/§9:517; amendment_sha256 §11:565; Appendix A:616–617; matched budget §4:242 | defect 6 wording, finding 3 |
| Stale v2 wording | CLEAN | grep: "folds exactly" only inside the §12 correction note (584); "calibration, not capacity" only as "the former … condition, now a label" (394); "16×16" only in the §5 correction note (277); "3 GPU-hour" only as the PILOT_SECONDS description (476); no "option A", "carried unchanged", "decided at the M2 draft", "exactly 32" | RESUME.md:34 "drop or keep the accuracy guard" is narrative history, not a contradiction |

## Check 2. §8 ledger arithmetic

| Item | Verdict | Evidence | Change |
|---|---|---|---|
| Core lines | OK | each count × unit matches; P1 6×56 = 336 booked 340 (rounding); sum 9,040 ≈ "about 9,050" (456) | none |
| Extended | OK | 1,650+900+360+360+570+150+210 = 4,200 (466) | F(0.25) at 6×25 s contradicts the launch-bound note (431–433: narrower hosts ≈ 41–42 s, 0.5 booked 35). Book 6×35 = 210; extended 4,260; combined 13,300 |
| Combined | OK | 9,040+4,200 = 13,240 ≈ 13,250 < 14,400 | none |
| 1.23×, 1.11×, 1.64× | STALE | 10,800/8,800, 10,800/9,700, 14,400/8,800 are v2 core figures; against the file's 9,050: 1.19×, 1.09×, 1.59× (478–480) | finding 2 |
| "part of the rest" at 1.3× | WRONG | 1.3×(9,050+1,650) = 13,910 leaves 490 s; R-all ×1.3 = 1,170 misses, and the rule (468–471) stops at the first miss | finding 2 |
| Confirmatory per anchor | WRONG itemisation | 45+25+30+300+100+120+120+80+250 = 1,070, not 1,010 (485–488). §10 has four nonlinear grids (M-sq, M-sq \| T-cool, M-wide, M-small): 240 gives 1,010 | finding 1 |
| n = 32 | OK | 32×1,010+1,500 = 33,820 s = 9.39 h; 43,200/33,820 = 1.28× | none |
| S-static margin "1.2×" | marginal | 32×1,120+1,500 = 37,340; 43,200/37,340 = 1.16× | "about 1.16×" (491) |
| §4 parameter-steps | OK | 2,128×9,600+280×9,600 = 23.1 M; 280×19,200 = 5.4 M; 4.3×; strict 7.6× | none |

## Check 3. Cross-references and links

All §n references land on the section holding the claim (§4 budget, §6 residual/D2, §7 V0 bound and SD(D5) rule, §9 step 6, §10, §12). All four draft links and seven amendment links resolve. §11:559–561 names three existing files. CLEAN.

## Check 4. Seeds

Every §11 block (6000–6005, 6006–6011, 6012–6017, 6018–6023, 6024–6029, 6100–6199, 7500–7599, 7700–7799) appears in Appendix A:616–618. `grep -rnE '\b(60[0-2][0-9]|61[0-9]{2}|75[0-9]{2}|77[0-9]{2})\b' src tests docs/plans` outside the draft returns nothing. CLEAN.

## Check 5. §2 source claims

| Claim | Verdict | Evidence |
|---|---|---|
| main() refuses seeds outside 3000–3005 | VERIFIED | `screen.py:56` INTACT_SEEDS = range(3000, 3006); `:57` BOTTLENECK_SEEDS = INTACT_SEEDS; `:442` refusal |
| fit_sgd nonlinear-only | VERIFIED | `fit.py:159–168` has no `nonlinear` argument; `:176–180` builds `Adapter(...)` with the default `nonlinear=True` (`:23`); affine only via `ridge(..., nonlinear=)` `:96` |
| no argmax / accuracy under src/ | VERIFIED | grep empty |
| materiality/posthoc/check entanglement | VERIFIED | `screen.py:54` ladders; `:259` P1/P2/P3/P3prime loop, `:271` A7/A8; `pilot.py:288–318` magnitude_decision holds (`:297` P3 replay, `:314` exhaustion); `:322–338` check_posthoc (former `:325`, four families `:327`, B3 `:338`); `screen.py:191–194` B4/B5 timing; `costs.py:44–56` |
| PILOT_SECONDS checked by Gate 3 ledgers | VERIFIED | `contracts.py:47`; `pilot.py:514,517,569` |
| "nothing enforces a confirmatory ceiling; sizing only reports within_ceiling" | MISIMPLEMENTED | `pilot.py:352–353` reports, but `:825–832` raises `FeasibilityHold` when `within_ceiling` is not True (16 h / 200 anchors). Pre-run enforcement exists; what is absent is a supervisor ceiling on the confirmatory run |
| state hash kind:name:dtype:shape | VERIFIED | `host.py:93` |
| tail[0] Conv2d(16, 64, 3), 1,024 centre taps, 64 biases | VERIFIED | `host.py:48–52` (`_block(16, 64)`, kernel 3, padding 1) |
| width-tuple "not only s ∈ {…}" | loose | `Host.__init__` takes only bottleneck `width ∈ {0,2,4,8}` (`host.py:46–49`); no `s` exists today. Low |

## Check 6. Amendment

| Row | Verdict | Evidence |
|---|---|---|
| Gate wording 1–7 | MATCH | protocol `:153–170` |
| 1 delivered | OK | protocol `:155` "v2 delivered" |
| 2 "certified … [link to 2026-10-09 certificate] … verdict stands as issued" | MISLINKED | the linked certificate's table says "Partial: single-run checks pass" (`2026-10-09-hybrid-correction.md:118`); completion is recorded in `2026-10-10-gate2-nyx-final.md:3,41` ("COMPLETE WITH RETAINED EXECUTION QUALIFICATION") |
| 3 not established; hold "not a Gate 3 verdict"; transfer never tested | OK | hold `:67–68`; disposition `:3–4`; protocol `:157–158` |
| 4, 6, 7 | OK | cert `:120,122,123`; protocol `:166–170` |
| 5 "specificity fails" | OK | cert `:121` |
| Converts failed/retired evidence to a pass | NO | row 3 "not passed", row 7 "never marks a gate passed", `:44–48` |
| SHA abbf3f52 | OK | `sha256sum` of the protocol = abbf3f52d1ef…; cert `:15`; manifests. `correction.py:24,78,88` pins the path and recomputes the hash against `plan_sha256` rather than holding the literal; "pinned in" is acceptable |
| Awaits acceptance | OK | `:3–5`, `:52` |
| STEERING §6 pointer | OK | diff; §6 carries an outline plus the link, while amendment `:54–55` says "only a link" (future state). Minor |

## Check 7. J0 consistency

No verdict depends on ΔAcc (§6:328–329, §7:392–395 "moving no threshold"); evaluator in §2:104–106; STEERING §4 rule 1 and §5 M0 row (diff) agree with §7:377–378 and §12:580. CLEAN.

## Findings needing an edit

1. **§8:486–487** "five nonlinear module grids 300" → "four nonlinear module grids (M-sq, M-sq | T-cool, M-wide, M-small) 240". The booked 1,010 and every downstream figure then hold.
2. **§8:478–482** replace "the core alone survives only a 1.23 × kernel overrun and core plus R-all a 1.11 ×" with "1.19 × … 1.09 ×"; "the core survives 1.64 ×" with "1.59 ×"; delete "and part of the rest" (at 1.3 × the rule stops at R-all). Add "(recomputed on the v3 ledger)" after "arithmetic".
3. **§2:124 and §10:538** replace "nothing enforces a confirmatory ceiling today; the sizing receipt only reports `within_ceiling`" with "today the only confirmatory ceiling is a pre-run feasibility check: the sizing step raises `FeasibilityHold` above 16 GPU-hours or 200 anchors (`pilot.py:825–832`); no supervisor charges or stops the confirmatory run itself".
4. **Amendment row 2** replace "certified on the synthetic lifecycle instrument ([certificate](../certificates/2026-10-09-hybrid-correction.md))" with "complete with retained execution qualification on the synthetic lifecycle instrument ([Nyx final certificate](../certificates/2026-10-10-gate2-nyx-final.md); the [development certificate](../certificates/2026-10-09-hybrid-correction.md) records it as partial at issue)".
5. **§8:465** "F(0.25) 150 s" → "F(0.25) 210 s (launch-bound, as the 0.5 host)"; "about 4,200 s" → "about 4,260 s"; §8:466 "13,250" → "13,300".
6. **§8:491** "leaves only a 1.2 × margin" → "leaves about a 1.16 × margin".
7. **§3:154, §5:313** "reaches below rung M2's near-empty start" → "reaches rung M2's near-empty start".
8. **§7:411** (low) append "(matched optimiser steps and example passes, §4)".

Verdict: **not ready: four text corrections precede the SHA pin (confirmatory itemisation 1,070 ≠ 1,010; stale v2 overrun multipliers and "part of the rest"; FeasibilityHold contradicts "nothing enforces"; amendment row 2 links the certificate that records Gate 2 as Partial).** All are one-line edits; after them, ready to freeze pending John's acceptance of the amendment and J10.
