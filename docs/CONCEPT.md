# Sultai: consolidated technical concept

Canonical synthesis as of 2026-10-09. Source keys below resolve in
[research/SOURCE_MAP.md](research/SOURCE_MAP.md). Material differences are
recorded in [DECISIONS.md](DECISIONS.md), not hidden in this synthesis.
This document consolidates intent and hypotheses; the implemented subset and
its measurements remain in [SMOKE.md](SMOKE.md).

## Objective and mechanism

Sultai starts with very little task-model structure and develops a capable
model through **addition AND deliberate removal**. The goal is comparable
capability with roughly half the final retained parameters of a credible
reference. The exact near-empty starting budget, reference, tasks and
capability tolerance still need preregistration. Training and search cost are
secondary objectives, reported separately rather than quietly omitted. [U]

Removal is part of development: taking away temporary or retained computation
can change the learning conditions, free capacity and create pressure for
reorganization. Learning can continue after removal. It is not defined as
only terminal compression, nor as an experimental ablation. Ablations are
interventions in controlled arms that test explanations of the mechanism.
The same physical deletion can have either role, depending on the experiment.
No success of this mechanism has yet been measured in Sultai. [U, R1, R3, V]

A temporary component may matter to the path without belonging in the final
model. It may shape host representations, serve as a temporary teacher,
alter optimization or supply helpful perturbations. These explanations must
be separated from extra training, a better final architecture and better
initial weights. Addition/removal is the central bet; diffusion is one possible
builder, not its definition. Denoising is not physical parameter removal. [R1–R3, U]

## Coarse commissioning, rich local investigation

The controller supplies a bounded opportunity: a site, permitted intervention
and budget. It does not have to diagnose the complete defect or prescribe the
computation. The seed independently observes paired local examples and task
feedback and may query a frozen shadow host with bounded probes. Narrow write
authority does not imply that the seed can see only a coarse summary. [U, R2]

The evidence must preserve relationships: which activation, prediction,
label/loss and intervention response belong to the same example and snapshot.
Means and marginals can lose exactly that association. No naturally observable
task label should be mistaken for a known correct hidden-layer activation.
Oracle site targets in the current TCD are a deliberately privileged fixture. [R2, A1]

Two hosts can have identical passive data but opposite local influence. The
small example `D_A(z)=z`, `D_B(z)=-z` at z=0 has the same output and target in
both hosts; a positive perturbation reveals which repair sign would help.
This demonstrates a possible information gap, not an autonomous diagnosis
algorithm. The seed needs evidence sufficient to choose a useful action, not
necessarily a complete reconstruction of the hidden defect. [R2, T]

An eventual bounded investigation loop is: collect permitted evidence, apply
fixed probes to one frozen snapshot, form candidates, select using separate
selection data, and retain/request evidence/withdraw under a declared rule.
Start with fixed probes before learned adaptive experiments. Revalidate a
candidate against the current host before installation if the snapshot changed.
Do not combine evidence from changing hosts without versioning it. [R1, R2, A1]

The seed's evidence is a functional input to its model. Correct pairing,
snapshot binding and conditioning/probe-shuffling controls therefore matter
even if dashboards are observationally inert. The meaning of Tamiyo differs
between source projects: Sultai's supplied coarse commissioning concept maps
to Simic HLD Aurelia; Nissa carries direct evidence, Momir forms candidates,
and Simic Tamiyo witnesses. Those Simic domains are design roles, not a
currently implemented runtime. Do not infer a repository rename. [U, R4, V]

## Formation methods and lifecycle boundaries

The research space includes fixed-shape weight generation; typed overcomplete
modules with removable groups/ranks; and later typed graph/cell developmental
rules that can create and delete components. Deterministic hypernetworks,
retrieval/interpolation, stochastic latent models, diffusion, flow-based
builders and learned recurrent update rules are candidates. Graph/cell rules
raise validity, variable-size representation and long-horizon credit problems;
they are proposals, not prerequisites for this TCD. [R1–R3]

Diffusion may propose useful alternatives when several repairs remain plausible.
It cannot recover information absent from all permitted evidence. Sampling
uncertainty is different from identifying which repair is needed. Apparent
weight diversity may be only symmetry; measure functional diversity. A
deterministic predictor trained on task utility or with a small latent need
not suffer the naive mean-of-incompatible-weight-targets failure. Diffusion
must earn its place against strong simpler methods at matched evidence and
candidate-selection budgets. [R2, A0]

Keep these controls separate: evidence acquisition, denoising/update steps,
host influence alpha, task-learning strength, and structural survival. In a
strict generated-formation arm, offline corpus/generator training can use
gradients; new-host formation uses permitted evidence without new-host task
parameter optimization; gradient-guided formation is a separate arm. After
formation, learning and blending can follow a declared schedule. Labels and
forward scoring need not be forbidden merely because backprop is forbidden.
Closed-form fitting and gradient-equivalent features require an explicit
protocol decision; calling no autograd routine is insufficient. [R1–R3, D06]

Evaluate immediate harm and benefit, then a predeclared common-future learning
horizon when studying delayed value. Record failures. Do not reject every
candidate solely for lacking instant gain, or let an immediate-only selector
discard all delayed-value candidates before testing the hypothesis. Which
tensors learn, optimizer-state copies, future data/randomness, alpha schedule,
step units and selection/test trajectories must be explicit. A fully frozen
system at constant alpha cannot demonstrate delayed learning. [U, A1]

John reports good Esper-lite seeds with negligible initial accuracy impact
followed by improvement in 3 or 4 steps, and bad seeds that immediately tanked.
This remains user-reported, with step units unresolved. Local documentation
supports lifecycle machinery, not validation of those particular trajectories.
Do not adopt the number as a calibrated horizon or safety threshold. [U, E]

Abstaining at formation and removing a component after it has shaped learning
are different operations. A presence head or an empty generated object can
prevent an unhelpful insertion, but does not replace the developmental removal
policy. Reports proposing continuous structure or removal of the old slot
ontology do not settle whether Sultai should abandon discrete modules. [H2, U]

## Evidence, controls and interpretation

The first future real-host comparison holds the commissioning instruction
fixed and varies seed evidence: coarse summaries, paired observations, then
paired observations plus fixed probes. Compare no repair, an adequate
conventional fit, retrieval/interpolation, deterministic generation and
diffusion where each evidence condition permits them. Report privileged
gradient/target access separately. Include healthy hosts, diagnostic no-influence
or information-missing fixtures, and shuffled association controls. [R2, A0, A1]

Do not label every failed teacher fit as proof of unrepairability. Possible
statuses include insufficient evidence, no repair found within budget,
insufficient expressivity, no influence through this interface, or genuinely
missing information where that is independently established. Retain failed
episodes for outcome/abstention analysis without presenting failed weight
vectors as successful generative targets. Test null calibration and false
allocation cost, not only mean gain among accepted repairs. [R1–R3, A0]

Split by independent host lineage and held-out bottleneck family, with all
branches/checkpoints of a lineage kept together. A joint held-out study must
prevent sharing either identifier; connected components or disjoint pools
avoid crossed leakage. Keep conditioning, selection and test examples apart.
Freeze candidates before selection feedback; distinguish a single candidate
from best-of-K and charge all queries. Report the number of independent groups
and account for family dependence in uncertainty. [A0, A1, T]

Add matched static capacity **early**, not only at the final allocation stage.
The same-snapshot conventional adapter asks whether generation improves repair
formation. A matched static module trained from step zero asks whether the
whole developmental route earns value beyond supplying capacity earlier.
Those are different controls with different learning histories. A known
defect/site/module is an oracle allocation privilege, not a practical system
that discovers where and when to add structure. [R4, V]

For later developmental claims, compare joint growth/removal, growth-only,
matched late pruning/removal, static final capacity, the discovered final
architecture retrained from scratch, and targeted controls for extra steps,
noise/regularization, inherited weights or teacher effects. Select controls
that distinguish the declared hypothesis, rather than copying a large legacy
campaign. If the final smaller architecture trains just as well from scratch,
the efficiency objective can still succeed; the special trajectory claim is
weaker. [R1, R3, U, A0]

Same-host ablation measures the cost of removal at that moment, including
dependence acquired during training. It cannot by itself establish how much
the component improved the learning trajectory. Matched branches from a common
snapshot with common future learning answer the stronger admission question.
Later removal/replaceability after a chosen adaptation horizon is a different
estimand; a full-tenure no-intervention control costs an additional trajectory.
Declare which question is measured and charge that cost. Esper's counterfactual
source and Simic's claim document explicitly acknowledge this distinction;
no Sultai causal result is implied. [H1, H2, E, V]

The historical Esper review reports misleading telemetry, confounded baselines
and framework-state interactions. Its counts, historical bug diagnoses and
accuracy figures have not been independently reproduced here. Retain the
engineering lesson: define every consumed metric, distinguish unavailable
values from zero, and verify that passive instrumentation preserves outputs,
gradients, RNG and optimizer state. Functional seed evidence is an intentional
input and needs its own integrity tests; a blanket ban on all evidence inside
computation would conflate these roles. No old telemetry platform or reward
formula is adopted wholesale. [H1, H3]

Count all physically retained parameters, trainable and frozen, and any
builder/controller/retrieval assets needed at inference. Report active
parameters, peak temporary capacity, optimizer-state bytes and inference cost
separately. Also report host training, repair-corpus construction, generator
training, probing, sampling/selection and integration costs. Zero masks or
zero blend weights are not evidence of physical removal. [R1–R3, A1]

## What exists and what comes next

The current TCD implements `h + alpha*(W tanh(h)+b)` with 272 fitted parameters,
an identity host, planted/healthy oracle fixtures, ridge optimization,
selection/split checks and the scalar ambiguity test. Its 12 passing unit
checks and near-zero planted error establish software behavior and known
representability only. The identity host's zero learned parameters do not
demonstrate real task development from near-empty. No generator, handover or
physical removal is implemented. [T, A2]

Simic has an implemented bounded CIFAR experiment substrate worth reusing
through a later separately reviewed, pinned interface. It does not yet offer
the HLD's full generative domain runtime or a drop-in 16-channel provider API.
The source audit and limits are in [research/SIMIC_RECONCILIATION.md](research/SIMIC_RECONCILIATION.md).
Keep Sultai's small TCD as an executable specification; do not merge the
projects or edit active experiments based on this research. [V, U]

The four-report Claude handoff is readable through Library; its two requested
file transfers failed with HTTP 403, so neither the bundle nor its expected
file hashes are verified on Nyx. Its historical reports are distinct from the
independent TCD design, which is still pending. [H1–H4]

Claude is independently preparing an adversarial TCD design in a separate
cloud project. It is a pending source, not an instruction to restart work or
a source of settled decisions. Preserve its independence until completion;
then compare assumptions, experiments and failure criteria using the process
in [research/CLAUDE_HANDOFF.md](research/CLAUDE_HANDOFF.md). [U]
