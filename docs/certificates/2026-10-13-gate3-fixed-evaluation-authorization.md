# Gate 3 fixed-evaluation repair package — authorisation record

John, 2026-10-12 (AEDT evening, 12 October UTC), in the Nyx Claude session
after a read-only state examination that listed the decisions required:

> yes, please authorise and execute - codex is frozen but I'm authorising
> you because I have claude credit spare right now

This transcribes the authorisation; it is not a cryptographic signature.

**Authorised scope**, as presented to John immediately before the reply:

1. The repair package: evaluator change, protocol extension with a new hash,
   CPU controls and gates, independent exact-source review, commit and push
   on `codex/gate3-screen`.
2. Executor: this session, `sultai-claude-nyx`, while `codex-sultai-gate3`
   is paused under John's quota instruction. The tracker claim could not be
   reassigned because the session's tracker writes are blocked by
   permissions; the claim record still names the Codex owner.

**Not authorised by this record**: any GPU execution, a new pilot attempt,
checkpoint inheritance, confirmatory execution, or merge to `main`. The
linked pilot 2 allocation and its disposition document are a separate
decision, as is the confirmatory seed-range allocation recorded in the
amendment for John's confirmation at the confirmatory freeze.

Implementation choice recorded for John: the padded fixed-128 shared
evaluator recommended by both independent reviewers was implemented, not the
divisor-chunk alternative this session had raised, because the divisor rule
would force every CPU fixture onto multiples of the chunk for no scientific
gain. The alternative is recorded in the amendment.
