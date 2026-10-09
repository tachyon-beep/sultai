# Bounded local repair design — proposal v0

## Question and non-claims

Can a seed form a useful nonlinear local repair when its instruction stays
coarse but its own observations become richer? This is a mechanism probe.
It does not test development from near-empty, deliberate physical removal,
or the half-final-parameter target. All designs below are proposals until
implemented and measured.

## Minimal host and adapter

Use a known 16-dimensional site h. Install residual repair
`h_repaired = h + alpha * (W tanh(h) + b)`, where W is 16 by 16, b is 16,
and alpha is an externally scheduled blend coefficient. W and b contain
256 + 16 = 272 learned parameters; alpha is a control, not a learned weight.
The fixed tanh prevents an accidentally affine parallel correction. Test
non-affinity numerically with a nonzero fixture; zero/no-op weights are
necessarily degenerate and do not invalidate the architecture.

The synthetic host and clean target generator are immutable during repair
formation. A conventional adapter baseline may fit only W and b on allowed
conditioning/training examples. Host truth and planted repair weights are
test-fixture internals, never generator inputs. A planted representable repair
is a positive control, not evidence that real defects are repairable.

## Information boundary

Tamiyo's fixed instruction contains site identity, permitted intervention,
shape and resource budget. It must not include a complete diagnosis, target
weights, private host state, or test labels. This is Sultai's proposed role,
not an assertion that the current Simic observability-only Tamiyo has changed.

The seed independently receives one of:

| Condition | Available local evidence |
| --- | --- |
| Coarse | Predeclared aggregate summaries only |
| Paired | Matched local inputs, site activations, target/residual observations |
| Paired + probes | The same paired set plus responses to a fixed probe bank |

Use sample identity to preserve pairing. Declare what target means and how
it could be observed on a real host; synthetic oracle targets do not establish
that real telemetry exists. Fixed probe identity and query budget are shared
across methods. Probes must use the same frozen snapshot and may not access
selection or test outcomes.

## Split, candidate selection, and accounting

Separate train/validation/test by independent host lineage AND bottleneck
family; branches/checkpoints/repairs of a lineage may never cross splits.
Inside each episode, separate conditioning, selection, and untouched test
examples. Use selection outcomes only for candidate selection. Evaluate test
once after freezing settings. Do not tune on synthetic smoke test outcomes.

Compare no repair, conventional adapter optimization, repair retrieval and
interpolation, deterministic generator, and conditional diffusion. Keep data,
parameter budget, probe count and evaluation horizon comparable. Report each
method's training cost, inference cost, selection cost and retained parameter
count separately. Report one generated sample and best-of-K separately, with
the same K and charged selection queries; neither may be disguised as the
other. Include shuffled-conditioning and mismatched-pair controls.

## Handover and subsequent removal stage

Evaluate both immediate effect and a predeclared short common-future learning
horizon, with a no-repair branch under identical future data. Track transient
harm and catastrophic failures separately from average gain. Do not define
success solely as immediate gain. John's Esper-lite report is evidence of
motivation, not a verified calibration of duration or threshold.

The next stage adds physically removable units, an explicit blend/handback
schedule, and retained parameter/optimizer-state measurements. Removal must
delete tensors and corresponding optimizer state or recompile a smaller
model; zero masks alone do not establish reduction. Compare the developed
path with the final architecture trained from scratch. Equal from-scratch
performance still meets the efficiency goal while weakening a trajectory
specific explanation. Do not copy the full legacy lifecycle machinery.
