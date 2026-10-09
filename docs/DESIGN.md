# Bounded technical concept demonstrator — design v1

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

The adapter is nonlinear in h but linear in its fitted weights. For direct
site residual targets and squared error, use an adequate least-squares/ridge
baseline on `[tanh(h), 1]`; report feature rank/conditioning and select
regularization only on permitted development data. This closed-form reference
does not apply unchanged to a nonlinear downstream task loss. New-host ridge
fitting is conventional optimization, even though it uses no backprop routine;
it is not forward-only amortized repair generation.

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

For the later telemetry comparison, the data-only evidence contract is:
coarse = conditioning count, aggregate task loss, per-channel activation
means/variances; paired = conditioning IDs, activations, predictions, task
labels/losses; paired + probes = these plus fixed intervention IDs/directions,
snapshot identity, paired output deltas and query counts. Oracle internal
residual targets are a separately labeled synthetic condition. No condition
has a callback that exposes hidden weights, arbitrary labels or gradients.
The instruction is fixed across episodes; condition-specific evidence belongs
to the seed, not the instruction.

Candidate banks are frozen before selection feedback. Selection returns a
winner, not an uncharged adaptive query loop; changing selection labels may
change the winner but must not change the pre-selection candidates. Keep
selection/test examples out of summaries and probe fitting. Each method must
declare all access, including fitting, probe and selection queries. An
optimizer that receives paired residuals or gradients is an information-rich
reference, not a matched coarse-condition competitor. Mark unavailable
method/condition combinations explicitly rather than silently giving them
extra evidence. The first TCD implements only the paired oracle reference and
a small fixed-probe ambiguity example, not this full comparison matrix.

## Split, candidate selection, and accounting

Separate train/validation/test by independent host lineage AND bottleneck
family; branches/checkpoints/repairs of a lineage may never cross splits.
Construct disjoint lineage and family pools, or assign connected components
of the lineage–family graph; grouping by the tuple alone is insufficient.
A single connected component cannot support the proposed joint unseen-lineage,
unseen-family evaluation. Keep a transitive-overlap negative test. Count
independent groups; if families cause shared dependence, uncertainty must use
independent components or an appropriate two-way grouped analysis.
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

This future phase must declare which host/adapter tensors learn, initial
optimizer-state copies, common future data order/randomness, alpha schedule,
step unit and evaluation times. Frozen tensors at fixed alpha cannot show
delayed learning. Keep the target immutable. If delayed value is the question,
candidate selection must use a predeclared future-horizon objective on reserved
selection trajectories, with separate untouched final test trajectories.
Until host learning and withdrawal exist, call it delayed repair evaluation,
not demonstrated handover. Keep denoising, blend influence, structural survival,
gradient strength and evidence-acquisition schedules separate.

The next stage adds physically removable units, an explicit blend/handback
schedule, and retained parameter/optimizer-state measurements. Removal must
delete tensors and corresponding optimizer state or recompile a smaller
model; zero masks alone do not establish reduction.
Count all retained parameters, including frozen tensors, plus optimizer-state
bytes and any builder/controller/retrieval assets still needed at inference.
Report trainable/frozen subsets and search-only assets separately. Compare the
developed
path with the final architecture trained from scratch. Equal from-scratch
performance still meets the efficiency goal while weakening a trajectory
specific explanation. Do not copy the full legacy lifecycle machinery.
