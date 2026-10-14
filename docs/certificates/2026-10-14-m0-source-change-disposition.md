# M0 source-change disposition: the confirmatory runs the amendment-5 implementation over the pilot's source (2026-10-14)

**Status: John's disposition under preregistration v4 §10 (rulings R27, R28),
recorded by `sultai-claude-nyx` on 2026-10-14. It authorises no execution.**

The one M0 pilot attempt ran on 2026-10-14 from `main` 0d4179a under
admission certificate 3 (identity `m0-v4.2`) with implementation source
SHA-256 `a21584a3085872df85eca3948dec0e702f20f77479a569fa9934ada126b4778b`
(`docs/results/m0-pilot-2026-10-14-0d4179a/pilot-result.json`). §10 says the
confirmatory runs the pilot's implementation source unless a hash-listed
disposition in the admission certificate names the change.

John's words, 2026-10-14, after the three independent reviews of the pilot
receipts: "Please fix all the majors and minors". The change is
[amendment 5](../plans/2026-10-14-m0-amendment-5.md) (A5.1 the survival
refit carries M-wide's selected step count; A5.2 the receipt additions the
reviews asked for) under identity `m0-v4.3`
([freeze certificate 5](2026-10-14-m0-freeze-5.md)), implemented on
`nyx/m0-amendment-5` with two exact-source reviews
(`docs/reviews/m0-capacity/amendment-5/`).

**Admitted confirmatory implementation source SHA-256:**
`b2c3553715abf7816ed690eeadbb59531cf1bdbab3aa9a6dd17a3609e5a83083`
(the `source_sha256` of `src`, `scripts`, `tests` and `pyproject.toml` at
the certified commit named in
[admission certificate 4](2026-10-14-m0-pilot-admission-4.json)).

What the change does not do: it re-derives no pilot value, moves no margin,
rule, arm set, seed block or ceiling, and admits the pilot of record as it
stands (amendment 5, A5.1). The pilot's measured kernels are carried with the
survival refit uplift the launcher publishes. The confirmatory go, its
ceiling (43,200 s, or 50,400 s with S-static under the 14-hour rule) and the
pilot-filled values remain John's decisions at gate C.

## Revision 2026-10-15: the S-static selection binds the launch mode (PR #13 review, P1)

An external review of PR #13 found that the launcher accepted `--static`
independently of the certificate's recorded gate C selection, so a flag
omitted at the command line could claim the single custody receipt under the
43,200 s non-static plan. The fix is one validation in `admit_launch`,
before custody: when the certificate carries `confirmatory_static_included`,
it must equal the launch flag, or the launch is refused naming the field and
the flag to use. A certificate without the field binds nothing. One test pins
both directions and the pre-custody refusal. No pilot value, margin, rule,
arm set, seed block or ceiling moves; the Gate 3 path is untouched.

**Admitted confirmatory implementation source SHA-256 (this revision,
superseding the hash above):**
`af2db4e72d89b8305340544ff0253673e45edb0917d0b765f5c48e3f5bb26bfa`
(the `source_sha256` at the certified commit named in
[admission certificate 5](2026-10-15-m0-pilot-admission-5.json), which
supersedes certificate 4 and carries its gate C fields).
