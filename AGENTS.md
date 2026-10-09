# Sultai working instructions

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
