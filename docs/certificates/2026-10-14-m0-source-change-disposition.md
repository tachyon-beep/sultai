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
`dcb432e02e50add4ae3c3b89a59e95f1600f5b1aeb8b50713fb0d28d1133bc09`
(the `source_sha256` of `src`, `scripts`, `tests` and `pyproject.toml` at
the certified commit named in
[admission certificate 4](2026-10-14-m0-pilot-admission-4.json)).

What the change does not do: it re-derives no pilot value, moves no margin,
rule, arm set, seed block or ceiling, and admits the pilot of record as it
stands (amendment 5, A5.1). The pilot's measured kernels are carried with the
survival refit uplift the launcher publishes. The confirmatory go, its
ceiling (43,200 s, or 50,400 s with S-static under the 14-hour rule) and the
pilot-filled values remain John's decisions at gate C.
