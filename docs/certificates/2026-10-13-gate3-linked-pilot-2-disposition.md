# Gate 3 linked pilot 2 (attempt 3) — John's disposition

John, 2026-10-13 (Nyx Claude session, replying to the delivered v4.1 package
and the costed proposal at `docs/plans/2026-10-13-gate3-linked-pilot-2-proposal.md`):

> Go and change filigree as well if you can

This transcribes the authorisation; it is not a cryptographic signature.
"Go" answers the proposal's request for the linked pilot 2 go. "Change
filigree" authorises reassigning the tracker claim `sultai-ef0da96d8a` to
`sultai-claude-nyx`, which succeeded on 2026-10-13.

**Dispositioned, per protocol amendment v4.1 §6 over the frozen v4 §9:**

1. Exactly one linked development pilot, attempt 3, under the v4.1 composed
   identity, launched by `sultai-claude-nyx` from the reviewed source commit
   named in the bound admission certificate.
2. Fresh training of the development hosts; no checkpoint from attempts 1
   or 2 is inherited.
3. Opening charge **1473.267264504917 process-seconds** carried from
   attempts 1 and 2 and the diagnostic; total ceiling **10,800 s**
   unchanged; **9326.732735495083 s** remain. The supervisor stops the run
   at the ceiling and that stop spends the attempt; no automatic extension.
4. The three prior custody receipts (`.weft/gate3-pilot-attempt.json`,
   `.weft/gate3-pilot-linked-retry-1.json`,
   `.weft/gate3-twin-diagnostic-attempt.json`) and the terminal charge
   records of attempt 2 and the diagnostic remain byte-preserved and are
   bound by hash at admission.
5. Every §9 stop rule and the ordering rule apply; any stop is a hold that
   returns to John. No further attempt, no confirmatory execution, no
   merge to `main` and no change to thresholds, counts or seed rules is
   granted by this disposition.
6. Raw output is staged under ignored `runs/`; sanitised evidence is
   published under `docs/results/` afterwards; weights and raw logs stay
   outside Git.
