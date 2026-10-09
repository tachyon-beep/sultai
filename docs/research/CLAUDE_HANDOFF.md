# Claude handoff: delivered sources and completed comparison

The independent Claude TCD and a subsequent role/scope revision were delivered
on Nyx in commit `72f9eeb`, together with four historical source reports and
an index. All six source hashes match that index; `sources.json` records actual
local byte counts/checksums. The source files are preserved unchanged in
`claude-handoff/`. This supersedes the earlier empty-directory/pending status.

The Library bundle and manifest transfers previously failed. A later transfer
of the independent TCD also failed after one bounded retry with HTTP 403.
Those failures do not negate the separate verified local delivery. No more
failed-route retries are planned. The original three pasted attachments are
still absent and must not be conflated with related delivered reports.

The delivered TCD adds only a revision-pointer paragraph to the supplied
59,999-byte independent original; removing that paragraph in memory reproduces
its expected SHA-256. The subsequent Emrakul/Tamiyo revision is separately
preserved. See [source-design provenance](../designs/README.md).

John authorized comparison after completion, then implementation of the
**hybrid design**. The comparison retained information tiers, strong controls,
frozen/live separation, static comparison and cost accounting. It corrected
affine-folding interpretation, nonlinear probe symmetry, beyond-first-order
claims, small-cluster safety calibration and overly broad stop conclusions.
The [hybrid contract](../HYBRID_CONTRACT.md) records the bounded resolution.
No conclusion or implementation was sent to steer the independent design
before completion. No compute estimate from the reports is an approved budget.
