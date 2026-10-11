# Gate 3 development-pilot implementation

Owner: `codex-sultai-gate3`, sole code writer; claim `sultai-ef0da96d8a`.
John authorized implementation, tests, independent panel review, PR, and only
then the bounded development pilot. Confirmation requires the pilot-filled,
reviewed and committed protocol; this package has no confirmatory launcher.

The fixed source protocol remains
`docs/plans/2026-10-10-gate3-headroom-screen.md`, SHA-256
`7668d20ad55c3a62384975a7cf1f3989dbe69bf701ce02bc9c68ac2601d0d42a`.
It is unchanged. No original `src/sultai` module or saved research/result
artifact is edited. The package is `src/sultai_screen`; both declared roots
remain subject to strict typing, policy, lint and format gates with mutations.

## Concrete execution choices

- Six intact and six width-4 hosts use development seeds 3000 through 3005.
  Shared labels support the descriptive pilot deficit reference; confirmatory
  ranges remain those in the protocol. Width 2 is the only automatic fallback.
  Width 8 holds because "collapsed units" has no operational definition.
- Oracle seeds remain three. No seed reduction or tolerance relaxation is
  introduced. Parent environment sets CUBLAS and one physical CUDA card before
  importing torch in each fresh child. CPU generators derive all stochastic
  streams; deterministic algorithms, cuDNN flags, no TF32/AMP/compile and two
  CPU threads are recorded. Installed torch/numpy versions must match the pins.
- Full-batch gradients accumulate sum loss divided by the complete batch across
  chunks of 128, then perform one update. This is not a minibatch substitution.
  A 129-example CPU known answer crosses the chunk boundary against a direct
  full-batch gradient. Conditioning remains 2,000 examples in the actual pilot.
- Frozen execution asserts every BN mode, parameter gradient state and a hash
  of every named parameter/buffer. Zero adapters and resampled no-op twins must
  agree bitwise. Candidate numerical failures remain visible and unselectable;
  no-op and selected-candidate nonfinite evaluations stop execution. A bank with
  one failed schedule retains the other schedule's finite winner and marks
  plateau comparison unavailable. All-bank failure preserves every candidate
  report/cost. Raw cheap development nonfiniteness is retained without changing
  selection; it becomes fatal only for global-rate or per-anchor selected fits.
- CIFAR training batches and metadata are read once each, hashed before pickle
  decoding from the same verified bytes. No archive/test file is opened.
  The audit hook rejects test-batch paths. Dev-index bytes, complete permutation
  hash and partition derivation are independently checked.
- The A7 artifact is the paired `run_formation()` generator: weights and complete
  report identities are retained and compared. Position-pair fits stay separate
  from the fixed generator. A3 ridge uses the declared affine family; A2 keeps
  the tanh family and a known answer against the legacy fit.

## Budget, custody and identity

The pilot ceiling remains 10,800 aggregate GPU process-seconds (3 GPU-hours).
Each launched child is charged its elapsed time through batch cleanup, including
CPU loading and idle time. Co-tenants are each charged separately, which
conservatively overcounts occupied-card time. A batch deadline divides remaining
budget by child count and reserves ten seconds per process for cleanup. Up to
three children/card are admitted; host and B3 fit replay both include that
co-tenancy on both physical cards; every child has its own session. The supervisor
polls all children, signals only owned sessions, signals all before waiting, then
escalates remaining children to SIGKILL. Exclusive, fsynced process/ledger/result
receipts preserve failure as well as success. SIGINT/SIGTERM trigger owned-child
cleanup. An uncatchable supervisor SIGKILL is not a successful completion;
retained custody forbids an automatic replacement attempt.

`.weft/gate3-pilot-attempt.json` is published exclusively before launching any
child and binds PID, output path and identity. Changing the output path cannot
reset the attempt. Never delete that receipt to obtain another attempt. Failed
admission before this receipt launches nothing. Outputs must be new paths under ignored `runs/` or `/tmp`. Runtime placement
is validated before custody is consumed; the protocol's `docs/results/gate3-pilot-<date>-<id>/` is the later sanitized publication
destination, not raw working-tree staging.

Source identity hashes every tracked file under source/scripts/tests plus
`pyproject.toml`, rejects any dirty/untracked state and refuses an empty declared
production root. Worker checkpoints bind anchor, source/protocol hashes, saved
state and loss-stream bytes. The pilot certificate binds the reviewed source
aggregate and commit ancestry, exact protocol, required gate/review decisions
and hashed in-repository evidence. Documentation-only evidence commits may
follow the reviewed source commit; changed source invalidates the certificate.

## Complete cost projection and decision order

The timing phase exposes timing/hash/finite-status receipts, not repair gains.
In addition to B3 200/800-step and full-batch timing, it measures the frozen B5
reduced grid and the two-sided B4 alignment/moments/transport kernels. B4 timing
uses a channel-permuted calibration tensor solely as a kernel timing fixture;
it does not stand in for cross-anchor scientific retrieval. A no-op development
pass times evaluation without reporting its loss. Nonfinite B3 timing reference
fits prevent timing admission. Required cost inputs cannot be missing or zero.

The projection uses observed maxima and explicitly accounts for: 64 intact
trainings; n bottleneck trainings; B3 on P1/P5/P6/P8; all fixed-rate cheap steps
and five-rate steps 1/4/16/64; B5 on eight P6 anchors; all posthoc A-arms; B4 on
every ordered other-anchor pair, with both alignments; and development evaluation
of every raw candidate for harm reporting. Whole-worker timings intentionally
duplicate some loads/evaluations. It is a conservative projection from pilot
measurements, never claimed as measured confirmation or a guaranteed duration.
The confirmatory 16 GPU-hour and 10-hour wall ceilings remain unchanged; future
confirmation still needs its own launcher, reviewed frozen inputs and runtime
stops. No confirmatory training is permitted by this implementation.

Receipts enforce timing/replay and seed decision, then magnitudes/known answers
and width, then global cheap rate, then SD/count/cost, then development medians.
An oversized count is published without capping and stops before medians.
Materiality/known-answer failures publish `software_valid=true`,
`instrument_valid=false`, `pilot_clean=false`; feasibility holds distinguish
valid measurements from incomplete scientific conclusions. Internal contract,
missing evidence, provenance, replay, timeout and selected/no-op numerical
failures remain execution-invalid. Every stop retains `confirmatory_ready=false`.

## Validation and launch gates

The local full quality report and runtime log are under
`docs/results/gate3-implementation-2026-10-11/`. They bind source bytes; the
implementation commit and independent final-candidate reports will be recorded
in the admission certificate after panel acceptance. The mypy wrapper only
selects `/usr/bin/python3` for the already installed torch/numpy stubs; no
packages or shared environments are changed. No typing exemptions are added.

Before launch: independent scientific and PyTorch/operational review, bounded
fixes/dispositions, final affected gates, committed source, PR, bound certificate,
then a fresh read-only terminal/owner-release, GPU identity/idle, existing
custody and active-run check. Any deviation holds. The launch is only the
previously authorized pilot; no confirmation, schedule or unrelated workload
change is included. Model weights and raw runtime logs stay outside Git.
