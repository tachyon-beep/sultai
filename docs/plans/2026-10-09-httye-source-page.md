**Emrakul learns how to make a useful local intervention. Tamiyo learns where and when that intervention is worth trying.** The host trainer supplies reproducible host states and carries the benefit through learning, handover and removal. This is the proposed system and evidence route; the implemented result so far is a bounded synthetic demonstrator.

## The system in one pass

**Train the host → observe candidate sites → choose a site and budget → form an edit or no-op → admit and apply safely → train and hand over → taper and remove → measure retained benefit.** Audited outcomes become training examples, with the host, Emrakul, Tamiyo and evaluator versions attached.

- **Host trainer:** owns task learning, checkpoints, replay and state adapters.

- **Tamiyo:** sees a defined, coarser host observation and chooses the site, timing and bounded budget, including the option to wait.

- **Emrakul:** uses richer local telemetry to form an edit or valid no-op, subject to bounded admission and rollback protections.

- **Evaluator and corpus:** record what actually happened, what it cost and what remained after removal.

**Current boundary:** the restricted demonstrator establishes mechanics. General-purpose host integration, meaningful transfer, a learned Tamiyo policy and co-training remain proposed work. Simic and Sultai are still two halves of a joint project with strategy undecided.

## How to Train Your Emrakul

**Sultai technical concept milestones · 9 October 2026.** The hybrid demonstrator has passed its bounded mechanics assay. The next question is whether generated repairs provide a useful, transferable advantage that survives handover and removal. The route below is a proposed minimum evidence plan; its future gates and training programme require owner agreement.

![How to Train Your Emrakul milestone route](visualize:fde1_bGliZmlsZV9WdmVxdmh1Y1dNbF8wdFA1ZnRBMGRR_FileDrive_1d4ac5aec98c819182c7e3b6f1975b6c)

### Evidence needed at each gate

1. **Contract hardening.** Reject malformed model state, incomplete acceptance evidence and invalid ledgers before computing success. Required measurements cannot default to zero; absent generator weights cannot become a valid zero model. Run Ruff, strict mypy with explicit Any rejected, positive and negative checker fixtures, and numerical-parity tests against the preserved baseline. The successor is still in progress.

2. **Instrument truth.** Known-answer cases must recover expected errors, parameter counts, lifecycle transitions and compute totals. Fault injection must expose missing evidence, leakage, invalid state and broken removal. Repeated runs must reproduce declared outputs within an agreed tolerance.

3. **Useful repair headroom.** Hold out whole procedural lineages. Compare the nonlinear repair family with a foldable affine control, retrieval, shuffled/mismatched proposals, random candidates and cheap SGD. Declare label, probe and task-data access by information tier; compare like with like. Demonstrate useful repairs beyond memorising the current four templates, while retaining the nonlinear case as the primary assay.

4. **Honest admission.** Test helpful, harmful, irrelevant and null candidates plus valid no-op cases. Select and tune only on permitted training/development evidence; seal held-out audit results. Report false admissions, unnecessary rejection and abstention separately. A no-op is a valid decision only when required telemetry is present and valid.

5. **Benefit after removal.** Separate frozen-host repair from live task-learning integration. Show actual handover, taper and physical adapter deletion, with retained benefit measured after removal. Compare against no growth, training the final architecture directly, and early static capacity at matched information and compute. The current no-growth result is a serious baseline to beat: it finished with lower error.

6. **Calibrated safety and utility.** Replicate across independent lineages and seeds; report uncertainty, tail failures and the dependence between repeated observations. Predeclare the minimum worthwhile benefit, acceptable degradation and abstention rules before seeing final evaluation results. Count proposal generation, selection, temporary training and retained inference costs.

7. **Training readiness.** Version the corpus, lineage splits, schemas, candidate outcomes and provenance. Audit leakage and coverage, retain rejected/no-op examples, reproduce a small end-to-end run and specify a bounded compute budget. This earns a decision on the next training programme, not an automatic training launch.

### Observed baseline and decision boundary

At **ccdfd52**, the synthetic nonlinear W tanh(h) + b demonstrator uses a restricted four-template formation family with admission/no-op, comparisons, task-learning handover, taper and physical deletion. Source-reported verification records **37 tests and 16 demo gates passed**, with two identical runs of approximately **2.83 seconds**. Held-out formation MSE improved from **0.0872 to 0.000333**; temporary parameters fell from **544 to 272 retained**. These results demonstrate the bounded mechanics. No growth achieved lower final error, so the assay has not established developmental advantage or competitive performance at half the retained size.

The latest hardening handoff records audit commit **f8011ce**, preserving ccdfd52. It describes explicit report schemas and checks, with final verification still pending. [Sultai task evidence](https://chatgpt.com/codex/tasks/01a11f20-bb4c-707d-b721-dd5fc741e0a8).

**Owner choices before the next evidence campaign:** approve or amend the minimum gate set, select the first transfer domain and information tier, agree numerical utility/risk margins and a compute ceiling, and decide which missing result would stop or redirect the programme. No numerical readiness score or arbitrary acceptance threshold is assumed here.

**Emrakul comes in two scales.** The tiny diagnostic generator has already been learned inside the demonstrator. A meaningful learned Emrakul programme needs the audited evidence and agreed scope above; its model family remains an experimental choice, with literal diffusion neither established nor mandatory. The broader half-retained-parameter ambition remains a later research objective and need not block every earlier training experiment.

## How to Train Your Tamiyo

**The prerequisite is an informative, audited Emrakul intervention corpus.** A trained, explicitly versioned Emrakul must generate useful examples across varied host states and sites, including failures, rejections and no-ops. Readiness depends on what those examples establish and cover, rather than an “80% ready” estimate or an arbitrary sample count.

![Tamiyo evidence route](visualize:fde1_bGliZmlsZV9TTnFXa3FvWi1mOFBWYTRVNk93cEd3_FileDrive_1d4ac5aec98c819182c7e3b6f1975b6c)

### The route from outcomes to site selection

1. **Fix the observation and action contract.** Specify the coarser host signals Tamiyo may see, candidate sites, timing, budget and wait action. Keep post-intervention outcomes out of the input. This establishes a decision that can be reproduced and unlocks comparable data collection.

2. **Collect versioned intervention outcomes.** Use a frozen Emrakul version on replayable host states, with declared candidate selection and information access. Record proposals, rejection/no-op reasons, safety events, task benefit, retained benefit and total cost. This establishes what this Emrakul can actually do and unlocks opportunity labels.

3. **Audit the opportunity labels.** Hold out host/procedural lineages, prevent outcome leakage, retain failed and unselected-case context, and account for how sites were chosen. Compare interventions against matched baselines where feasible. A bad outcome with one Emrakul version does not establish that a site is intrinsically unrepairable. This establishes useful, bounded labels and unlocks policy fitting.

4. **Fit a bounded site and budget policy.** Train Tamiyo to predict realized opportunity under the named Emrakul version, using only its permitted pre-decision observation. Compare with random, fixed and simple heuristic policies at matched information and cost. This tests whether choosing where and when adds value and unlocks sealed evaluation.

5. **Evaluate the complete loop.** Freeze policy versions and evaluate on sealed held-out lineages. Include unnecessary interventions, missed opportunities, safety tails, cost, abstention and retained benefit after removal. Passing predeclared utility and risk criteria earns a decision on a bounded live-host trial.

### Co training later

Alternating Emrakul and Tamiyo updates may eventually improve both: Tamiyo finds promising sites, and Emrakul learns from their outcomes. Start with frozen versions and explicit evaluation windows. When Emrakul changes, the meaning of an opportunity label can change; preserve its version, remeasure affected cases and separate old and new policy evaluation. Otherwise the training target moves underneath Tamiyo.

## Building blocks to discuss

**Implemented** below means demonstrated only in the current bounded assay. **To verify** means an interface or capability needs an audit before reuse. **Proposed** means a design or evidence requirement, not an implementation commitment.

### First make the host replayable

- **Host trainer and checkpoint replay — synthetic task-learning loop implemented; checkpoint/replay and broader state adapters to verify.** Capture model, optimizer, RNG, data position and lifecycle state. Prove resume/replay and matched branching; this unlocks causal comparisons between intervention and no-growth runs.

- **Site, probe and telemetry interfaces — restricted inputs implemented; general contract proposed.** Separate Tamiyo's coarse observations from Emrakul's rich local probes. Prove timing, shapes, required fields and permitted information; this unlocks reproducible proposals without leakage.

- **Module and edit executor with rollback — bounded admission/no-op and removal demonstrated; general executor and exact rollback to verify.** Define insertion, application, failure recovery and state restoration. Prove invalid edits cannot corrupt the host; this unlocks controlled intervention trials.

### Then make the evidence trustworthy

- **Held-out data and lineage registry — proposed.** Keep procedural families, host checkpoints and data splits traceable. Prove held-out lineages stay sealed; this unlocks credible transfer claims.

- **Oracle and baseline evaluators — bounded comparisons implemented; broader suite proposed.** Include no growth, final architecture trained directly, early static capacity, retrieval, affine controls, shuffled/random edits and cheap SGD at matched information and compute. Prove evaluator known answers and fault detection; this unlocks meaningful advantage claims.

- **Lifecycle handover, taper and physical removal — implemented in the assay; transfer to other host classes to verify.** Agree what must survive removal and prove both parameter deletion and post-removal performance; this unlocks retained-benefit evaluation.

- **Reward, utility and cost definitions — proposed owner decision.** Balance task gain, retained size, compute, latency, risk and abstention. Count probing, generation, selection, training and inference; this unlocks consistent labels and go/no-go criteria.

- **Safety admission and calibration — bounded checks implemented; general calibration proposed.** Test invalid, harmful, null and helpful cases, false admission and false rejection. Prove the declared risk bounds under held-out evidence; this unlocks a bounded trial.

### Finally make the learning loop useful

- **Corpus generator with provenance and versioning — four-template generation implemented; diverse audited corpus proposed.** Store host and policy versions, observations, selection, every outcome and costs. Prove coverage, reproducibility and leakage controls; this unlocks meaningful Emrakul and Tamiyo training.

- **Emrakul training and evaluation — tiny diagnostic generator learned; wider programme proposed.** Establish transferable useful repairs and calibrated no-op behavior through the evidence gates above; this unlocks the intervention corpus Tamiyo needs.

- **Tamiyo observation and policy training — proposed.** Learn site, timing and budget from the audited corpus, then evaluate the whole loop; this unlocks evidence about allocation rather than edit formation alone.

- **Experiment orchestration, resources and reporting — bounded demo runs implemented; campaign runner to verify.** Version configurations, enforce budgets, resume safely and report distributions and failures. Simic's runner may be reusable after a checkpoint, telemetry, lifecycle and accounting interface audit; its readiness for Sultai is not established.

## Decisions for the next discussion

1. **Which host class first?** Choose the smallest credible host and task with useful intervention headroom.

2. **What counts as formation and removal?** Define the allowed edit family, host learning path, taper and physical deletion boundary.

3. **What gain is worth the cost?** Agree utility, degradation and safety margins, information tiers and the bounded compute ceiling.

4. **Which benchmark can falsify the idea?** Choose held-out lineages, matched baselines and the result that would stop or redirect the programme.

5. **What can be reused?** Audit the host trainer and Simic runner interfaces before choosing reuse or a small dedicated harness.

These are discussion and evidence-design choices. The next training programme, implementation scope and campaign budget still need agreement.

[Flight Report](https://chatgpt.com/space/page_a22a27abde98819188e25dc25695e80c) carries the short operational status. [Sultai task evidence](https://chatgpt.com/codex/tasks/01a11f20-bb4c-707d-b721-dd5fc741e0a8) carries the implementation handoffs.
