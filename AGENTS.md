# Sultai working instructions

Current correction authorization: John assigned the cloud owner the bounded
HTTYE correction and normal draft-PR/review/merge workflow. The frozen extension
is docs/plans/2026-10-09-correction-protocol.md. Prior source/results stay
immutable in Git and their saved artifacts remain byte-preserved. The parent
owns Page edits, STR scheduling and Nyx sync/burn; do not alter other schedules.
Only short CPU validation occurs in cloud; no Simic changes or paid/GPU work.
The referenced Nyx parent instructions/Filigree store are absent in cloud;
John explicitly confirmed exclusive correction ownership. Do not fabricate
tracker actions or let that historical path prevent the authorized correction.

John's operation-relative policy governs: T1 guarantees type plus required value
invariants, T2 guarantees type with explicit documented recoverable domain
failures, and T3 validates/coerces unknown inputs. T3 boundaries require a paired
test AST fingerprint. Run scripts/check_policy.py with the usual strict quality
and runtime checks. Never conflate InputDataError/FitUnavailable with an internal
ContractViolation, or convert any missing evidence into a successful no-op.

The ccdfd52 independent-review snapshot is immutable. Successor typing and
required-contract hardening is separately authorized and documented in
docs/TYPING_CONTRACTS.md. Run the local quality gates with positive/negative
controls and the affected runtime tests. Keep every production module covered
by strict typing and report source identity. Never use Any, casts, ignores,
missing-field defaults or empty coverage to manufacture success. Explicit
optional absence and intentionally sparse cost-construction defaults are
allowed where their contracts distinguish them from missing required state.

Read README.md and docs/RESUME.md first. Parent /home/john/AGENTS.md applies;
use the Filigree workflow at /home/john/.agents/skills/filigree-workflow/SKILL.md.
This installation uses .weft/filigree/ rather than the historical .filigree/.

Preserve original research files byte-for-byte. Keep proposals, user reports,
mathematical checks, and measured results visibly distinct. Missing original
bytes must stay a blocker; a reconstructed text is not an exact import.

ELSPETH landing has priority. Only short CPU unit/synthetic checks are in the
current scope. Do not touch Simic/Esper repositories or running experiments.
Do not create/push a remote or claim off-host backup without verification.
Exclude credentials, private raw logs, model weights and live tracker DBs
from Git. Commit coherent milestones and update the resume brief.

Use Sol implementers and independent Astra design/correctness review when
useful; keep ownership disjoint and avoid repeated review loops. TCD means
technical concept demonstrator (John, 2026-10-09). Keep the current synthetic
instrument distinct from real-host/generated-repair and removal stages.

The active hybrid contract is docs/HYBRID_CONTRACT.md. John authorized bounded
hybrid implementation after the independent Claude review completed. The
source designs are preserved in docs/designs/ and docs/research/claude-handoff/;
do not overwrite them. Docs/DESIGN.md links active versus historical scope.
Only short, single-process CPU work is authorized. No GPU pilot, large corpus,
paid service or Simic/ELSPETH changes. The lead owns docs/integration/commits;
workers have disjoint files recorded in docs/plans/2026-10-09-hybrid-tcd.md.
Avoid `filigree session-context` during read-only inspection: this installed
version auto-refreshes instruction files and can start a dashboard. Use
`filigree list`/`show` for ownership checks; writes use the atomic workflow.
