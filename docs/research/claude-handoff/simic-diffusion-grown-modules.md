# Simic: diffusion-grown neural modules

*Research report, 9 October 2026. Prepared for John.*

**How to read the evidence markers.** Claims about other people's work carry one of three tags.

- **[V]** The claim was extracted from the primary source and then survived three-way adversarial verification in this run (votes noted where useful).
- **[E]** The claim was extracted from a fetched primary source but was not put through adversarial verification. Treat it as probably right but check before citing externally.
- **[K]** Background knowledge not fetched in this run. Treat it as a pointer to check.

Anything without a tag is my own reasoning, synthesis or proposal, and is labelled as such where it matters.

**Project context inspected.** `tachyon-beep/esper-lite` at commit `2d296b6` (6 July 2026, "product: checkpoint #22"). Two points from it shape this report. First, in that codebase *Simic* names the reward, PPO and accounting subsystem, not the whole system, so "Simic, previously Esper" is a rename that the repo does not yet reflect. Second, esper-lite seeds already receive task gradients while forward-isolated: `kasmina/slot.py` uses a straight-through construction (`host + (seed - seed.detach())`) so the seed learns from the task loss without changing the host's output. The proposed "no task backprop during the diffusion phase" is therefore a genuine departure from the current lifecycle, not a description of it. The blueprint catalogue already contains a zero-parameter `noop` seed, and a decay policy (Emrakul) is planned but not shipped.

---

## Short assessment

**Strongest plausible formulation.** Treat the seed as a *learned prior over local repairs*, sampled from a generator that was trained offline on many examples of "this host, at this site, in this state, was helped by this addition (or by nothing)". At deployment the generator reads a forward-only *site signature* (activation statistics at the insertion site plus a cheap, closed-form summary of what the downstream error looks like from there), proposes several candidate repairs, and keeps at most one. Each candidate is expressed in a canonical, rank-ordered form so that "how much of the repair survives" is literally the number of retained rank components, and "nothing survives" is a valid sample. Overgrowth happens in two places: in the population of sampled candidates and in an over-wide initial rank. Refinement happens through forward-evaluated selection, then gated task learning with a sparsity cost that physically removes rank components. That is "a lot becoming a little" with a mechanism attached.

**Closest demonstrated work.** Conditional generators already produce useful fixed-shape additions to a frozen host without task gradients at generation time: LoRA weights by conditional latent diffusion (COND P-DIFF), conditional recurrent diffusion (ORAL) and a non-diffusion hyper-decoder (Drag-and-Drop LLMs) **[V]**. G.pt generates parameter updates conditioned on the host's current parameters and a target loss **[V]**. GradMax, Firefly and NeST grow structure locally but use task gradients to do it **[V]**. None of these conditions on a diagnosed internal bottleneck, learns a null outcome, or discovers module structure.

**Largest unresolved leap.** Transfer. Every verified generator learns from backprop-produced checkpoints, usually from narrow trajectories, and transfer evidence stops at unseen checkpoints of the same model family or unseen benchmarks reported by the authors themselves **[V]**. One critique argues that prominent weight generators largely memorise their training checkpoints **[E]**. The idea stands or falls on whether a generator can propose a *useful repair for a host state it has not seen*, better than retrieving the nearest training repair. The half-parameter target is a further, separate leap with no direct evidence at any scale in this review.

**First experiment worth doing.** A known insertion site in a deliberately starved small CNN on CIFAR-10, a fixed-maximum-rank residual seed, and a set-structured diffusion generator conditioned on forward-only site signatures, trained on a corpus of oracle repairs *including null cases*, split by training run. The decisive comparisons are a deterministic hypernetwork on the same data, nearest-neighbour retrieval (the memorisation test), the same seed from ordinary initialisation with the same fine-tuning budget, and GradMax-style gradient initialisation. If diffusion beats retrieval on held-out runs and held-out bottleneck placements, and correctly says "nothing" where capacity is not the problem, the idea has earned its next rung.

---

## 1. Conceptual explanation

### 1.1 The idea in plain terms

A small host network is trained until it struggles. A controller (Tamiyo) decides *where* extra computation might help. At that site, instead of picking a module from a menu, the system grows one. Growth starts expansive, with many candidate repairs or a deliberately oversized one, and is then trimmed until only the part that earns its keep remains. Sometimes nothing earns its keep, and the site is left as it was.

The original intuition bundles four separable claims:

1. **Amortised knowledge of repairs.** Something can know, from experience, what kind of addition tends to help a host that looks like this.
2. **Overgrow then refine.** Starting expansive and trimming finds better small solutions than building small directly.
3. **Iterative, generative refinement.** The trimming is best done by a stochastic iterative process like diffusion.
4. **Path value.** The temporary excess can improve the final small result even after it is gone.

These can succeed or fail independently. The experiments below are designed to tell them apart.

### 1.2 What diffusion does and does not give you

A diffusion model learns to reverse a gradual noising process, so that repeated denoising turns random noise into a sample from the distribution it was trained on. Applied to parameters, the "image" is a vector of weights. Three corrections to the original framing matter:

- **Removing noise does not remove parameters.** A denoised weight tensor has exactly the dimensions of the noisy one. Compactness must be *represented* in what is generated (for example, as gates or ordered rank components that can be truncated), then physically enforced by slicing.
- **There is no observable perfect target at an internal layer.** A generator cannot be told "produce the activation that would have been right". What it can learn is the distribution of *additions that helped* in past cases, conditioned on what was observable at the time.
- **The seed knows nothing it has not been told or trained on.** All repair knowledge comes from (a) the offline corpus the generator was trained on, (b) the conditioning signals at deployment, and (c) any task learning after insertion. The design question is how much of the work each one does.

### 1.3 Phase vocabulary

To make "no backprop initially" precise, the lifecycle has four phases.

| Phase | What happens | Backprop allowed? | Task information allowed? |
|---|---|---|---|
| **P0 Offline generator training** | Generator learns from a repair corpus | Yes, to train the generator | Yes; the corpus was built with task gradients |
| **P1 Sampling** (the "diffusion phase") | Generator proposes seed(s) from a site signature | **No** gradients into host or seed | Only via forward passes: activations, outputs, labels, losses |
| **P2 Integration** | Seed blended in; task learning switched on | Yes, into the seed; host optionally | Yes |
| **P3 Consolidation** | Unused structure removed; seed fossilised, absorbed or deleted | Yes | Yes |

Three strictness levels for P1 are worth keeping distinct:

- **P1-strict.** Conditioning uses only unlabelled activation statistics. No labels, no losses.
- **P1-forward.** Conditioning and candidate selection may use labels and losses computed by *forward* passes, including closed-form least-squares probes. No backpropagation through the network. This is the default in this report.
- **Variant G (relaxed).** Gradient guidance during sampling (for example, nudging denoising steps with the gradient of a local loss). This is stronger but is not the proposed mechanism and should be reported separately.

### 1.4 Diagram

```
             ┌───────────────────── offline (P0) ──────────────────────┐
             │ many host runs → snapshots → oracle repairs at sites    │
             │ (incl. failed / null cases) → canonicalise → train G    │
             └──────────────────────────────┬──────────────────────────┘
                                            │ generator G (frozen)
 host training ──► Tamiyo picks site s ──► site signature σ(s) (forward only)
                                            │
                                   P1: sample K candidates
                                   from G(· | σ), each a set of
                                   rank components {(u_i, v_i, g_i)}
                                            │
                       forward-evaluate on calibration batch
                       ┌──────────┴───────────┐
                  none helps               best candidate
                       │                       │
                  null: nothing            truncate ranks with g_i < τ
                  inserted                 (physical slicing)
                                               │
                                    P2: zero-output insertion, α ramp,
                                    task learning + rank-sparsity cost
                                               │
                                    P3: drop dead ranks; fossilise,
                                    absorb into host, or remove
                                               │
                         report (Δloss, retained params, null?) → Tamiyo
```

---

## 2. Closest literature

### 2.1 Comparison table

Columns: what was shown; what was generated or grown; conditioning or feedback; whether architecture can change; whether task gradients are needed at generation time; scale; relationship to Simic.

| Work | Demonstrated | Generated / grown | Conditioning / feedback | Arch. change? | Task grads at generation? | Scale & evaluation | Relationship to Simic |
|---|---|---|---|---|---|---|---|
| **G.pt** — Peebles et al. 2022, [arXiv 2209.12892](https://arxiv.org/abs/2209.12892) **[V]** | One generative step matched baseline optimisers needing thousands of iterations | Parameter *updates* for a fixed network | Current parameters + target loss/error/return | No | No | MNIST, CIFAR-10, Cartpole; one model per architecture, dataset and metric; trained on ~23M checkpoints from 100k+ runs **[E]** | Closest to "conditioned on host state and desired outcome". Whole-network, not additive. No cross-architecture transfer; weak extrapolation beyond training losses **[E]**. |
| **p-diff** — Wang et al. 2024, [arXiv 2402.13144](https://arxiv.org/abs/2402.13144) **[V]** | Generated parameters comparable to their training checkpoints | Last two BatchNorm layers of a fixed ResNet-18 by default; full params only up to ~300K | Unconditional | No | No | Small vision; 300 checkpoints from one converged model | Shows literal latent diffusion over weights works. Mostly memorises when checkpoints are few **[E]**. |
| **D2NWG** — Soro et al., [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f74d79573d71078848973009d0e99bdb-Abstract-Conference.html) **[V]** | Task-specific weights from a model-zoo distribution; selected layers in LLaMA-3.2-1B (+3% maths) | Whole nets or selected layers | Dataset characteristics, task text, architecture spec | No | No | Vision zoos to 1B LLM | Diffusion over weights conditioned on *task*, not on internal need. A claim that unseen-domain transfer works mainly as fine-tuning initialisation was **refuted 0–3**, so do not rely on that characterisation. |
| **COND P-DIFF** — [arXiv 2408.01415](https://arxiv.org/abs/2408.01415) **[V]** | Generated LoRAs roughly match trained ones (style FID 32.94 vs 33.01) **[E]** | LoRA weights, fixed rank and placement | Task text + two-shot examples (NLP); style features (images) — forward only | No | No | BERT-family GLUE; PixArt-α styles; last 150 checkpoints of one run per task | Fixed-template addition to frozen host. Unseen styles tested; no unseen GLUE tasks; authors say data limits generalisation (2–1 vote). |
| **ORAL** — [arXiv 2503.24354](https://arxiv.org/abs/2503.24354) **[V]** | Rank-8 LoRAs from noise, used with no fine-tuning; works on unseen *evolved* host checkpoints | LoRA weights (up to hundreds of millions of params) | Text description of host architecture + task with few-shot examples | No | No | Mistral-7B, SD 2.1; trained on 18 LoRAs | **Closest precedent for "generated addition to a changing host".** Transfer is to unseen checkpoints of the same family, compared only against zero-shot, not against LoRAs trained on the evolved host. |
| **Drag-and-Drop LLMs (DnD)** — [arXiv 2506.16406](https://arxiv.org/abs/2506.16406) **[V]** | Up to 30% average gains over strongest training LoRAs on unseen benchmarks; up to 12,000× less overhead than fine-tuning (self-reported) | Full LoRA matrices in one forward pass | Unlabelled task prompts | No | No | Qwen2.5 0.5B–7B | Deterministic (non-diffusion) comparator. Fails with few training datasets **[E]**. |
| **Text-to-LoRA (T2L)** — [arXiv 2506.06105](https://arxiv.org/abs/2506.06105) **[E]** | Hypernetwork compresses hundreds of LoRAs; zero-shot to unseen tasks | LoRA weights | Natural-language task description | No | No | LLM adapters | Simplest hypernetwork baseline pattern. |
| **MEND** — Mitchell et al., [arXiv 2110.11309](https://arxiv.org/abs/2110.11309) **[E]** | Learned editors make local edits to >10B models; trains on one GPU in under a day | Low-rank weight edits inside existing layers | One input–output pair *plus its fine-tuning gradient* | No | **Yes** (transforms the gradient) | T5, GPT, BERT, BART | A learned local *repair* operator, but gradient-fed. Useful as a Variant-G reference point. |
| **GradMax** — Evci et al. 2022, [arXiv 2201.05125](https://arxiv.org/abs/2201.05125) **[V]** | Adds neurons as a forward no-op; initialises fan-in by SVD to maximise gradient norm | New neurons in known layers | Task gradients at insertion | Width only | **Yes** | Vision | Clean insertion interface (zero fan-out). Natural gradient-informed baseline for the seed's initial state. |
| **Firefly descent** — Wu et al., [NeurIPS 2020](https://papers.nips.cc/paper/2020/hash/fdbe012e2e11314b96402b32c0df26b7-Abstract.html) **[V]** | Joint architecture and parameter descent; smaller and more accurate in continual learning **[E]** | Wider and deeper candidates, greedily selected | Taylor approximation of task loss | Width and depth, predefined vocabulary | **Yes** | Vision NAS and continual learning | **Literally local overgrow-then-select, with gradients.** The key non-diffusion control. |
| **NeST** — Dai, Yin & Jha, [arXiv 1711.02017](https://arxiv.org/abs/1711.02017) **[V]** | Grows from a sparse seed, then prunes; 15.7× (AlexNet) and 30.2× (VGG-16) fewer params at comparable accuracy **[E]** | Connections and neurons | Gradients for growth, magnitude for pruning | Yes (connectivity) | **Yes** | MNIST, ImageNet | Closest "little to lot to little" precedent. Baselines are legacy dense nets, so the ratios overstate gains against strong compact models. |
| **NORTH\*** — Maile et al., [AutoML-Conf 2022](https://proceedings.mlr.press/v188/maile22a.html) **[V]** | Several growth strategies beat a static net of the *same final architecture* trained from scratch | Neurons, zero fan-out at insertion | Orthogonality of activations (forward statistics) **[E]** | Width | Partly (training continues with gradients) | MNIST MLPs, 5 seeds | Direct, small evidence that growth *path* adds value. A claim of beating a larger CIFAR baseline with under half the params was **refuted 0–3**. |
| **RigL** — Evci et al. 2020, [arXiv 1911.11134](https://arxiv.org/abs/1911.11134) **[E]** | Fixed-budget sparse training with drop-and-regrow; strong on ImageNet ResNet-50 | Sparse connectivity | Magnitude (drop) + occasional gradients (grow) | Connectivity | Yes | ImageNet, WikiText-103 | Strong compact-model baseline without a dense phase. |
| **ExpandNets** — Guo et al., [NeurIPS 2020](https://neurips.cc/virtual/2020/poster/17344) **[E]** | Linear over-parameterisation during training, contracted exactly to the compact net afterwards, beats training the compact net directly and beats distillation | Extra linear layers, removed algebraically | Task gradients | No (fixed template) | Yes | Classification, detection, segmentation | **Cleanest evidence that temporary capacity can improve a model whose final size is fixed.** |
| **DiffusionNAG** — An et al., [ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/171c3678c36e39fc0074f3e7332a9a66-Abstract-Conference.html) **[E]** | Graph diffusion generates architectures guided by property predictors; up to 35× search speed-up | Whole-architecture graphs in a predefined space | Separately trained accuracy predictors | **Yes**, within a vocabulary | No (predictor guidance) | NAS benchmarks, MobileNetV3 space | Literal diffusion over *structure*. Whole nets, not site-local additions. |
| **HyperNCA** — Najarro et al. 2022, [arXiv 2204.11674](https://arxiv.org/abs/2204.11674) **[E]** | A 314-param NCA grows policy weights ~5.7× its size; trained by CMA-ES | Weights of a fixed-shape policy | Developmental dynamics; reward via evolution | No | No (black-box) | Small RL | Gradient-free developmental generation. |
| **Neural Developmental Programs** — Najarro et al. 2023, [arXiv 2307.08197](https://arxiv.org/abs/2307.08197) **[E]** | Local rules grow structure and weights from a seed | Nodes, edges, weights | Local node communication; *not* activity-dependent | Yes | Varies (evolution, RL, SL) | CartPole, MNIST 91% | Structural growth from a seed, below baselines. Growing larger than needed hurt. |
| **EAS** — Cai et al., [AAAI 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11709) **[E]** | RL meta-controller applies function-preserving widen/deepen; 4.23% CIFAR-10 error on 5 GPUs | Net2Net-style widen/deepen | Controller over architecture encoding | Yes, predefined ops | Yes (training) | CIFAR-10 | Precedent for Tamiyo-like learned growth control. |
| **Reinforced Continual Learning** — Xu & Zhu 2018, [arXiv 1805.12369](https://arxiv.org/abs/1805.12369) **[E]** | LSTM controller decides filters per layer; reward = accuracy − complexity; 42–53% fewer params than PGN/DEN | Units per layer | Validation accuracy minus complexity penalty | Width | Yes | CIFAR-100 continual | Matches esper-lite's "accuracy minus rent". Much higher search cost than baselines. |

### 2.2 Supporting evidence on scaffolding and pruning

- **Lottery Ticket Hypothesis** — Frankle & Carbin, [arXiv 1803.03635](https://arxiv.org/abs/1803.03635) **[E]**: subnetworks at 10–20% size match dense accuracy at small scale, but only with their original initialisation.
- **Rethinking the Value of Network Pruning** — Liu et al., [arXiv 1810.05270](https://arxiv.org/abs/1810.05270) **[E]**: for structured pruning, training the pruned architecture from scratch matches or beats fine-tuning inherited weights; pruning acts as architecture search. This is the most important counter-evidence to "the overgrown weights carry value".
- **Variance-transfer growth** — Yuan et al., [arXiv 2306.12700](https://arxiv.org/abs/2306.12700) **[E]**: growing to a fixed final architecture saves training compute (54.9% of baseline on CIFAR-10 ResNet-20), with no final-parameter saving. Also: newly grown parts need learning-rate rebalancing, which bears on P2.
- [arXiv 2501.18012](https://arxiv.org/abs/2501.18012) **[E]**: small-start networks grown by a learnable size variable or a controller mask outperform static networks of the same final size on small regression and classification tasks.
- **Memorisation critique** — Zeng et al., [arXiv 2506.07998](https://arxiv.org/abs/2506.07998) **[E, raised by a verifier]**: prominent weight-generation methods are argued to mostly memorise or blend training checkpoints and not beat simple noise or ensemble baselines. Which methods were tested was not confirmed.

### 2.3 Novelty, at the level of mechanisms

Nothing found in this run combines the following. This is absence of evidence, not proof of novelty, and the morphogenetic and NAS-diffusion literatures were only partly covered.

| Mechanism | Nearest precedent | Gap |
|---|---|---|
| Generator conditioned on *site-level forward diagnostics* | ORAL and DnD condition on task text or prompts; G.pt on whole parameter vectors | No conditioning on activations or error structure at an internal insertion site **[low confidence that this is unexplored]** |
| Learned *null* outcome | esper-lite's catalogue `noop`; RCL can add zero units | No generator trained on failed repairs to output "nothing" |
| Generated addition whose *size* is part of the sample | Variable rank appears only via pruning (NeST, RigL) | No generator varies rank, width or connectivity of an addition |
| Forward-only guided refinement of a repair | DiffusionNAG uses predictor guidance for whole architectures | Not applied to local additions |
| Single developing field replacing a sequence of partial scaffolds | ExpandNets (fixed template), NORTH\* path effects | Not tested |

---

## 3. Candidate mechanisms

All three share the same insertion interface: a residual branch at site *s*, with output `y = h_s(x) + α · f_θ(x)`. Insertion is function-preserving because either α = 0 or the output projection starts at zero (the GradMax convention).

### M1. Canonical-set latent diffusion over a rank-ordered seed (literal diffusion)

**Plainly.** The seed is a set of up to *R* rank components. Each component reads one direction from the site's input, applies a nonlinearity, and writes one direction back, scaled by a gain. A diffusion model generates this set, conditioned on a description of the site. Components whose gain comes out near zero are discarded. If every gain is near zero, nothing is inserted.

**1. What is generated.** A set `{(v_i, u_i, g_i)}_{i=1..R}`, with `f(x) = Σ_i g_i · u_i · φ(v_iᵀ x)`. This is a bespoke instance of a fixed template (a rank-≤R bottleneck branch). The template is predefined; the *effective rank* is discovered.

**Expressing it in the host's own basis.** This is the key move for transfer, and is my proposal **[synthesis]**. Compute the principal components `P_in` of the site's input activations and `P_out` of its output activations. Generate `ṽ_i, ũ_i` in those coordinates, then map back with `v_i = P_in ṽ_i` and `u_i = P_out ũ_i`. The generator is then indifferent to how the host happened to permute or rotate its hidden units. That addresses a real obstacle: two hosts with identical function can have different raw weights.

**Symmetries.** For a linear low-rank addition `ΔW = UVᵀ`, any invertible `R×R` matrix M leaves the product unchanged (`UM · M⁻¹Vᵀ`). With a nonlinearity, the residual symmetry is permutation of components plus per-component scaling (exact for ReLU). Two fixes, both simple:
- a **permutation-equivariant denoiser** (a transformer over component tokens with no positional encoding), so the model treats the set as a set; and
- **normalisation**: unit-norm `ṽ_i, ũ_i`, all scale carried by `g_i ≥ 0`, sign fixed by convention.

This leaves the gains as the only scale variable, which is what makes truncation meaningful.

**2. What informs the repair.** The site signature σ(s), computed from a forward pass over a few thousand training examples:
- spectra of input and output activation covariances (how saturated the site is; effective rank);
- in P1-forward: a **closed-form ridge probe** from site activations to the output-error residual `(onehot(y) − softmax(z))`, expressed in the `P_in` basis. This tells the generator which directions of the site's input are predictive of what the network currently gets wrong. It uses labels and least squares, not backprop;
- class-conditional activation means for correct versus incorrect examples;
- coarse host descriptors: depth of site, width, training step, current loss.

**Detection versus location.** The signature can show that the site's representation is saturated and that predictive information is not reaching the output. It cannot by itself show that *extra capacity at this site* is the fix rather than optimisation trouble or lack of data. That distinction must be learned from the corpus, which is why null cases matter.

**3. How it is trained.** Corpus of tuples `(σ, R*, Δ)`: for many host snapshots, train an oracle seed at rank R with task gradients and a rank-sparsity penalty (host frozen), canonicalise it, and record the gain Δ in held-out loss. Where Δ is below a threshold, the target is the **empty set** (all gains zero). Failed repairs therefore appear as explicit null targets. Train with standard diffusion or flow-matching loss, with classifier-free guidance (randomly dropping σ during training). All the backprop cost sits here and must be reported as generator cost.

**4. What "no backprop initially" permits.** P0: full backprop to train the generator. P1: sampling and forward evaluation only; the host and seed receive no gradients. P1-strict drops the ridge probe and the label-based selection. P2: task gradients into the seed. Variant G (relaxed, separate): gradient guidance during denoising.

**5. Integration.** Four separate schedules: denoising steps (finished before insertion in the default design), the blend coefficient α, the rank-sparsity pressure, and the task learning rate. The default is *generate in isolation, then insert*. Progressive blending during denoising is possible (insert partial samples with small α) but entangles a moving host with an unfinished sample. Keep it as a later ablation. For a changing host, re-sampling is cheap, so the seed can be re-conditioned if the site signature drifts beyond a threshold before fossilisation.

**6. Compactness and absence.** Retained parameters = `r′ · (d_in + d_out + 1)`, where r′ is the number of components with `g_i > τ`. The kept components are *physically sliced* into a smaller module, not zero-masked inside a dense allocation. Null is a sampled outcome, not a failure. During P2, a group-sparsity penalty on the gains (or hard-concrete L0 gates) lets further components die.

**Which symmetry problems matter now.** In the simplest version (fixed site, one host family): permutation and scale symmetry of components (handled above) and host-basis variation (handled by PCA coordinates). In the ambitious version: variable input and output dimensions across sites (needs dimension-agnostic tokenisation, e.g. one token per principal direction), alignment across different host architectures, and structural vocabularies.

### M2. Amortised deterministic seed plus gradient overgrow-and-prune (no diffusion)

**Plainly.** A hypernetwork reads the same site signature and outputs one seed at full rank R. The seed is then trained with task gradients and a sparsity cost that trims it down. Overgrowth is in parameters; refinement is ordinary sparse training.

1. Same template and canonical form as M1.
2. Same signature.
3. Same corpus, trained by regression to the canonical oracle seed (plus a classifier head for null).
4. P1 is a single forward pass.
5. Integration is identical to M1.
6. Compactness comes entirely from the P2 sparsity cost.

**What it preserves.** Amortised knowledge and overgrow-then-refine. **What it drops.** Multimodality and population-level selection. When several quite different repairs work equally well, a regression model tends to average them into something that works worse than either. Whether that happens here is an empirical question, and M2 is the control that answers it.

### M3. Structural seed: graph generation within a small operation vocabulary

**Plainly.** The generator first proposes a small computation graph for the seed (for example, choices among 1×1 conv, depthwise 3×3, a squeeze-excitation gate, or nothing, with widths), then weights are produced by M1 or M2 for that graph.

1. Operations and connectivity from a predefined vocabulary. This is genuine structural discovery *within* the vocabulary, not open-ended invention. The vocabulary is the catalogue moved inside the generator.
2. Same signature, plus descriptors of nearby host structure.
3. Corpus requires oracle searches over structures at each snapshot, which multiplies cost. A property-predictor approach (as in DiffusionNAG) is cheaper: train a predictor of Δ from (σ, graph) and guide generation with it.
4–6. As M1, with "null" as an explicit graph.

**Relationship to esper-lite.** M3 is the generative version of Tamiyo's blueprint choice. It moves the "what" decision from the controller into the seed, which is the stated ambition.

### Recommendation

Build **M1 and M2 together on the same corpus, signature and template**. They differ only in the generator, which makes the comparison clean, and M2 is the strongest simple explanation of any M1 success. Defer M3 until M1 or M2 shows transfer to held-out host runs. If M2 matches M1, that says the valuable part of the idea is the *learned prior over repairs and the site signature*, not the stochastic process. That would still be a successful realisation of "Tamiyo chooses where; the seed knows what".

**What literal diffusion could add that M2 cannot** **[synthesis]**:
- **Candidate populations for free.** Sampling K repairs and keeping the best one, judged by forward passes, is overgrowth in *candidate space*. That is a faithful reading of "a lot becoming a little" that costs inference rather than parameters.
- **Calibrated doubt.** Wide disagreement between samples is a signal that the site is ambiguous, which can feed the null decision and Tamiyo's credit assignment.
- **Forward-only guided refinement.** Sequential Monte Carlo over partially denoised samples, reweighted by forward-evaluated loss, steers generation without gradients. This keeps P1 within the forward-only rule while still using task feedback.

---

## 4. Temporary scaffolding, learning trajectories and parameter efficiency

### 4.1 What could produce a smaller capable model

Four candidate sources, each needing its own test:

| Source | Mechanism | Evidence nearby |
|---|---|---|
| **Better architecture** | Growth puts capacity where it is needed, so less is wasted elsewhere | NeST, Firefly, RCL report fewer params **[V/E]**; Liu et al. say architecture is what pruning really finds **[E]** |
| **Better weights** | Generated or overgrown weights carry information a fresh initialisation lacks | Lottery tickets **[E]**, but fragile to learning-rate tuning per Liu et al. **[E]** |
| **Better trajectory** | Temporary capacity changes the optimisation path, and the benefit survives removal | ExpandNets **[E]**; NORTH\* beat same-architecture static on MNIST **[V]**; RigL's topology change escapes minima **[E]** |
| **Amortisation** | The generator transfers repair knowledge across runs | ORAL, DnD within narrow bounds **[V]** |

The half-parameter target needs the first source, helped by the third. The generator mainly saves *search and training effort per repair*; it is not obviously a source of parameter efficiency by itself.

### 4.2 Can a single developing field replace several partly blended seeds?

**Plausible reading of the esper-lite observation [synthesis; the observation itself is reported, not verified].** Early partly blended seeds may have acted as *transient teachers or optimisation aids*: they shaped the host's representation so that later seeds found better solutions. Three mechanisms are consistent with that:

1. **Linear over-parameterisation effect** (as in ExpandNets): extra paths change the effective learning dynamics even when they add no final expressivity.
2. **Representation shaping**: early seeds pull host features towards a form later seeds can use, then become redundant.
3. **Regularisation or noise**: partial blends perturb the host and act like dropout or noise injection.

A single field could reproduce (1) through an over-wide rank that is gradually pruned, and (2) if its early high-rank phase pushes the host while its late low-rank phase keeps only what is needed. It cannot reproduce effects that depend on *sequential diversity* (different seeds trying genuinely different things) unless it samples several candidates over time.

### 4.3 Tests that separate the explanations

Each of these is a control in the experiment progression below.

| Explanation | Test that distinguishes it |
|---|---|
| Extra optimisation | Give the static compact model the same total steps and FLOPs |
| Better architecture only | Retrain the discovered final architecture from scratch, properly tuned (Liu et al.) |
| Better initialisation | Retrain the final architecture from the seed's inherited weights, rewound (lottery style) |
| Regularisation or noise | Replace the scaffold with a frozen random branch of matched output variance, blended on the same schedule |
| Teacher effect | Distil from the scaffold-equipped model into the compact one; if that matches, the scaffold is a teacher |
| Genuine path dependence | Benefit survives all of the above and persists after scaffold removal, across seeds |

### 4.4 Division of labour

**What the proposal removes from Tamiyo.** Blueprint choice, blend-curve and speed choice for multiple scaffold seeds, and some stopping decisions.

**What it relocates into the seed.** Knowledge of what a good repair looks like (now in the generator), the null decision, and rank selection.

**What still has to travel.**
- Tamiyo → seed: the site and its signature (or the seed computes the signature locally).
- Seed → Tamiyo: predicted gain before insertion, realised gain after, retained parameters, null or not, and sample disagreement. The esper-lite "accuracy minus rent" economy needs these to assign credit.
- Host → both: drift in the site signature, which may trigger re-sampling or removal.

**What is not removed.** The hardest diagnostic problem stays with Tamiyo: telling a *capacity* bottleneck from an optimisation or data problem. One promising, unverified route **[synthesis]**: let the generator double as a probe. It can predict gain for candidate sites from their signatures, giving Tamiyo a forward-only "value of growth" map. The site signature then also serves as the compact learned summary Tamiyo is meant to observe.

### 4.5 Scale

Everything verified here sits between MNIST and 7B-parameter hosts with fixed adapter templates. No evidence in this review bears on trillion-parameter models. Hierarchical control (the planned Narset and Esika layers) and CTDE (centralised training with decentralised execution, where a shared critic trains many local controllers that act on local observations) are reasonable architectural directions. They are not supported or refuted by anything found here.

---

## 5. The minimal experiment and two extensions

### 5.1 Rung 1: forward-only generated repair at a known site

**What it tests.** Whether a generator, conditioned only on forward-pass information, can propose a useful, correctly sized or null addition for a *host state from a run it has never seen*, better than retrieval and better than ordinary initialisation.

**What it leaves open.** Site selection (the site is given, so this is an oracle-site setting), structural discovery beyond rank, self-removal over long horizons, scaffolding, and the half-parameter claim.

**Host and task.** CIFAR-10. A small CNN in the esper-lite `cifar_baseline` style, with one deliberately starved block: a 1×1 channel bottleneck of width *c* ∈ {4, 8, 16, 32} after block *b* ∈ {1, 2, 3}. Narrow *c* creates real capacity bottlenecks; wide *c* creates sites where extra capacity should not help.

**Null-by-design conditions** (extra capacity should not help):
- wide bottlenecks (*c* = 32 or no bottleneck);
- hosts trained with heavy label noise or a too-high learning rate (failure for non-capacity reasons);
- hosts trained on 10% of the data (data-limited).

**Seed.** M1 template at rank R = 32, residual at the bottleneck output, nonlinearity GELU, input and output in PCA coordinates of the site.

**Corpus.**
- ~400 host runs = 12 bottleneck configs × ~33 seeds and hyperparameter jitters, plus the null-by-design runs.
- Snapshots at 5 points per run ≈ 2,000 host states.
- For each: oracle seed trained with task gradients (host frozen), 3 epochs, group-lasso on gains, 3 restarts; keep best; record Δ on a held-out validation slice; label null if Δ < 0.3 percentage points.

**Splits (leakage control).** Split by *run*, never by snapshot. Test sets:
- **T1** held-out runs, seen configurations;
- **T2** held-out bottleneck width (train on *c* ∈ {4, 16, 32}, test on 8);
- **T3** held-out block position (train on b ∈ {1, 2}, test on 3);
- **T4** held-out null types (train with noise and LR failures, test on data-limited hosts).

**Pseudocode: generator training (P0)**

```python
corpus = []
for run in host_runs:                       # each run: config + seed
    for snap in run.snapshots:
        sig   = site_signature(snap, site, X_probe, y_probe)   # forward only
        seed, gain = train_oracle_seed(snap, site, R=32,
                                       steps=3*epoch, group_lasso=λ)  # backprop
        target = canonicalise(seed, P_in(sig), P_out(sig))     # sort, normalise
        if gain < GAIN_MIN:
            target = EMPTY_SET                                 # null example
        corpus.append((sig, target, gain, run.id))

train, test = split_by_run(corpus)          # no shared runs across splits
G_diff = train_set_diffusion(train, cond_dropout=0.1)          # M1
G_det  = train_regressor(train)                                # M2 control
retrieve = nearest_neighbour_index(train)                      # memorisation control
```

**Pseudocode: seed deployment (P1 then P2)**

```python
def deploy(host, site, G, K=16, mode="forward"):
    sig = site_signature(host, site, X_probe, y_probe if mode=="forward" else None)
    # P1: no gradients into host or seed
    with torch.no_grad():
        cands = [G.sample(sig) for _ in range(K)]
        cands = [truncate(c, tau) for c in cands]               # drop g_i < tau
        if mode == "forward":
            scores = [calib_loss(host, site, c, alpha=1.0) for c in cands]
            best = cands[argmin(scores)]
            if min(scores) > calib_loss(host, site, None) - margin:
                return NULL                                     # nothing inserted
        else:                                                   # P1-strict
            best = cands[0]
            if best.is_empty(): return NULL
    seed = slice_to_module(best)                                # physical rank r'
    insert(host, site, seed, zero_output=True)
    # P2: task learning switched on
    for t in range(T_integrate):
        alpha = ramp(t)                                         # blend schedule
        loss  = task_loss(host) + λ * group_lasso(seed.gains)   # sparsity schedule
        loss.backward(); step(seed); maybe_step(host)
    prune_dead_components(seed, tau)                            # physical removal
    return seed
```

**Decisive controls for rung 1** (all share the oracle site, so all are labelled *oracle-site*):

| Control | What it rules out |
|---|---|
| **C1** M2 deterministic hypernetwork, same data | That diffusion adds nothing beyond amortisation |
| **C2** Nearest-neighbour retrieval of the closest training repair (and the mean of the k nearest) | Memorisation |
| **C3** Same rank-R seed from ordinary initialisation, same P2 budget | That generation adds nothing beyond extra training |
| **C4** GradMax-style SVD gradient initialisation, same P2 budget | That forward-only conditioning is worse than a one-shot gradient look |
| **C5** Oracle seed for that exact snapshot | Upper bound |
| **C6** Random-direction branch with matched gains, frozen | Noise and perturbation effects |

**Metrics.**
- Gain at zero P2 steps (pure generation quality).
- P2 steps to reach 90% of oracle gain.
- Final gain after a fixed P2 budget.
- Retained rank r′ and retained parameters versus oracle r′.
- Null precision and recall on T4.
- Per-split results T1–T4, with mean and spread over at least 3 generator training seeds.

**Resource estimate.** Assumptions: one modern data-centre GPU; CIFAR-10 held in GPU memory; small CNN at ~5–10 s per epoch.
- Host runs: ~400 runs × ~30 epochs × ~7 s ≈ 25 GPU-hours.
- Oracle repairs: ~2,000 snapshots × 3 restarts × 3 epochs × ~7 s ≈ 35 GPU-hours.
- Generators: small set transformer over ≤32 tokens, ~1–5 GPU-hours each.
- Evaluation: minor.

Total ≈ 60–80 GPU-hours, dominated by corpus building. Report it as generator cost, separately from per-deployment cost.

**How to read the outcomes.**

| Result | What it supports |
|---|---|
| M1 > C2 and C3 on T2/T3 at zero P2 steps, null recall is high on T4 | The specific mechanism: forward-only generative repair with learned null |
| M1 ≈ C1 > C2, C3 | Amortised repair prior works; diffusion is not needed at this rung |
| M1 ≈ C2 on held-out splits | Memorisation; the generator does not transfer |
| M1 and C1 lose to C3 after short P2 | Generation is no better than initialisation plus training |
| C4 ≫ M1 | Gradients carry information the signature lacks; revisit P1 or adopt Variant G |
| Null recall low | The signature cannot tell capacity bottlenecks from other failures; Tamiyo's diagnosis problem remains open |

A failure of M1 does not falsify growth-based parameter efficiency. A success does not establish it.

### 5.2 Rung 2: structure, removal and scaffolding at a known site

- Add M3: a small vocabulary (none, 1×1 branch, depthwise 3×3 branch, squeeze-excitation gate) with widths, generated with weights.
- Add a **scaffold mode**: the seed acts for *T* epochs with high rank, then decays α to zero while the host continues training, then is removed.
- Measure whether the host's improvement survives removal, using the section 4.3 controls (extra steps, frozen random branch, distillation, retrained final architecture).
- Add the esper-lite catalogue as a comparator: Tamiyo picking a blueprint at the same site.

This rung tests "seed discovers *what*" and "temporary structure can matter to the path without remaining".

### 5.3 Rung 3: learned site selection and the parameter-efficiency curve

- Start from a host at roughly 25–40% of a credible static model's size. Let Tamiyo choose sites using site signatures plus the generator's predicted gain. Grow until a parameter budget is reached.
- Report a **quality versus final-parameter curve**, not a single point.

**Comparators.**
- A family of strong static compact models (width and depth scaled), each trained with the same total compute *and* with distillation from the large model.
- RigL at matched final sparsity.
- Firefly-style gradient growth.
- The esper-lite catalogue policy.
- Each discovered final architecture retrained from scratch.
- A conventionally trained larger model.

**Comparable capability.** Same test accuracy (or loss) within the seed-to-seed standard deviation of the static model, at the same numerical precision and evaluation protocol.

**Accounting, reported separately.**
- Final stored parameters (host + retained seeds).
- Any generator or controller needed at inference (none in this design).
- Active parameters per example.
- Temporary peak parameters.
- Inference FLOPs.
- Host training compute.
- Search and controller compute.
- Generator corpus and training compute.

"Half" is supported only if the grown model matches a static model of twice its final parameter count *and* the retrained-from-scratch and compute-matched controls fall short.

---

## 6. Conclusion

### Demonstrated nearby

- **Diffusion and hypernetworks can generate useful weights for fixed-shape additions to a frozen host, with no task gradients at generation time.** COND P-DIFF, ORAL, DnD, T2L. *High confidence.* Evidence that would change this: replications showing these results reduce to checkpoint memorisation.
- **Generation conditioned on the host's state and a target outcome works within a single architecture and task.** G.pt. *High confidence.*
- **Local grow-and-prune and overgrow-then-select produce compact networks using task gradients.** NeST, Firefly, GradMax, RigL. *High confidence*, though ratios against legacy dense baselines overstate gains against strong compact models.
- **Temporary capacity can improve a model whose final size is fixed.** ExpandNets; NORTH\* on MNIST. *Moderate confidence*: ExpandNets was not adversarially verified here, and NORTH\* is small-scale, with its CIFAR "under half the parameters" claim refuted in verification.

### Plausible synthesis

- **A generator conditioned on forward-only site signatures, expressed in the host's activation basis, can propose useful repairs for held-out runs of the same host family.** *Moderate-to-low confidence.* ORAL's unseen-checkpoint result is the nearest support; the memorisation critique is the nearest threat. Rung 1 T1–T3 decides it.
- **Rank-ordered canonical sets make compactness and null outcomes native to the generated object.** *Moderate confidence* that it works mechanically; unknown whether nulls are predicted accurately.
- **Diffusion's value lies in candidate populations, doubt estimates and forward-only guidance, not in denoising per se.** *Moderate confidence* as an argument; untested.
- **One developing field can replace several partly blended scaffold seeds when the benefit is optimisation-path shaping.** *Low confidence.* Rung 2 scaffold controls decide it.

### Still speculative

- **Structural discovery beyond a predefined vocabulary.** *Low confidence.* No generative precedent for local additions; NDPs grow structure but below baselines.
- **Learned site selection that separates capacity bottlenecks from other failures.** *Low confidence.* Null recall on T4 is the first measurable signal.
- **Half the final parameters at comparable capability against strong compact baselines.** *Low confidence; no direct evidence at any scale here.* Rung 3's Pareto curve, with retrain-from-scratch and compute-matched controls, is the evidence that would move it.
- **Hierarchical or CTDE control of growth in very large models.** *Speculative.* Nothing found bears on it.

**Answer to the question.** Local generative overgrowth and refinement is a coherent way to frame "discover what a small model needs", provided overgrowth is located in candidate populations and over-wide rank rather than in denoising itself, compactness is built into what is generated, and nulls are trained for explicitly. The part that can be tested today is rung 1, a single experiment costing on the order of tens of GPU-hours. It decides whether a forward-only repair prior transfers beyond memorisation. Everything beyond that depends on its answer.

---

### Source notes and limits of this review

- 24 primary sources were fetched. 118 claims were extracted and the top 25 put through three-vote adversarial verification: 22 confirmed, 3 refuted.
- Refuted claims:
  - D2NWG unseen-domain characterisation (0–3);
  - NORTH\* trigger and initialisation description (0–3);
  - NORTH\* CIFAR result with fewer than half the parameters (0–3).
- Morphogenetic, NCA, NAS-diffusion and scaffolding sources were extracted but not adversarially verified ([E]).
- The literature is moving quickly (2024–2026), so newer site-conditioned generators may exist that this review missed.
