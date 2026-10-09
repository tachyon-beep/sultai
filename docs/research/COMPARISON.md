# Research comparison and bounded synthesis

Prepared 2026-10-09. All three Library items returned complete API text
(`has_more=false`: 996, 314 and 468 lines respectively). Selected sections
were inspected for this synthesis. The API text enabled reading, **not an
exact local original import**. Original-file sizes/hashes remain unverified.
Citation tokens embedded in the reports are report content, not verified
citations here. The primary-source table below identifies checks actually made.

## Reports: useful proposals and limits

| Source | Contribution adopted | Qualification/correction |
| --- | --- | --- |
| OpenAI initial, sections on abstraction, schedules, symmetry, temporary scaffolding and first experiment | Separate formation, blending, task learning and structural survival; compare generated repair with adequate simple controls; test final architecture from scratch | Its initial affine 1×1 proposal can be too weak; use fixed nonlinear features. Assumed common host coordinates do not establish transfer across independent hosts. No demonstrated half-parameter result |
| Local telemetry follow-up, sections 1–4 and 6–7 | Controller summary and seed evidence are distinct; preserve example pairing; fixed shadow probes can resolve ambiguities passive evidence cannot; hold controller instruction fixed | Rich task feedback is not a known correct hidden activation. The initial frozen assay cannot detect co-adaptation-only value. No improvement found is not proof of no repair |
| Claude, M1/M2 and staged experiments | Keep deterministic generator/retrieval controls, explicit null outcomes, physical slicing, and staged structure/removal work | GELU input-scale canonicalization is invalid; PCA does not align hosts automatically; unsuccessful teacher runs cannot be definitive null labels; proposed ~400-host corpus is outside current authorization |

The first TCD uses `W tanh(h)+b`, a fixed nonlinear family with 272 fitted
parameters. It deliberately avoids an internal hidden-layer permutation
problem. It does not remove host-coordinate variation: independently trained
hosts may still represent different functions in their 16 coordinates.
Parameter interpolation/retrieval requires compatible coordinates or measured
functional alignment. Do not assume a PCA transform supplies that alignment.

## Mathematical checks

For GELU(x) = x Φ(x), `GELU(2) = 1.9544997361`, whereas
`2 * GELU(1) = 1.6826894921`. These values were recomputed with Python
`math.erf` on Nyx. Thus input norm cannot generally be absorbed into an output
gain while preserving the GELU computation. ReLU's positive homogeneity does
not transfer to GELU. Unit-norm canonicalization must be activation-specific
or keep a separate input scale. This is a mathematical counterexample, not
an experimental result.

PCA eigenvectors admit sign changes; repeated eigenvalues permit rotations
within the eigenspace. Even distinct eigenvalues do not make semantically
different host representations interchangeable. Validate functional response
to paired probes and include explicit alignment controls before mixing weights.

The chosen adapter is nonlinear in activation but linear in W and b. Ridge
regression on fixed tanh features is an adequate site-target baseline. A
strong fit on planted targets verifies the instrument; it offers no evidence
that diffusion is needed and no test of real task-gradient-free formation.

## Primary references checked

These checks read primary abstract/proceedings pages on 2026-10-09. They
confirm nearby mechanisms and bibliographic identities, not reproduce their
results or establish a combined Sultai system.

| Primary source | Supported nearby mechanism | Gap to Sultai |
| --- | --- | --- |
| [G.pt, Peebles et al., 2022](https://arxiv.org/abs/2209.12892) | Conditional diffusion over network checkpoints predicts parameter updates from a starting vector and requested metric | Does not establish local insertion from a coarse instruction or physical removal |
| [Neural Network Diffusion / p-diff, Wang et al., 2024](https://arxiv.org/abs/2402.13144) | Autoencoded subsets of trained parameters can be generated through latent diffusion | Weight generation alone does not identify the repair a host needs |
| [Conditional Neural Processes, Garnelo et al., ICML 2018](https://proceedings.mlr.press/v80/garnelo18a.html) | Learned predictors adapt predictions to observed context examples | Context adaptation is precedent, not demonstrated host repair |
| [Deep Adaptive Design, Foster et al., ICML 2021](https://proceedings.mlr.press/v139/foster21a.html) | An offline-trained policy maps previous observations to the next experiment with a forward pass | Supports the proposed investigation pattern, not compact neural growth |
| [ExpandNets, Guo et al., arXiv v5 2021](https://arxiv.org/abs/1811.10495) | Linear factor expansions aid training and can contract algebraically for inference | Contraction is different from teaching a host through a nonlinear temporary branch and deleting it |

Other leads in the reports (including ORAL, D2NWG, graph generation and recent
developmental preprints) remain leads; this checkpoint does not claim their
detailed findings were independently verified. Nothing above demonstrates the
complete near-empty-to-half-size Sultai objective or proves novelty.

## Esper-lite lifecycle lessons, read-only

Archive inspected at commit `5ad6f9a657ceffba04946f68ef2c9b9a21f040b4`:
`/mnt/data/archive/esper-lite/README.md` and
`docs/reference/tamiyo-lifecycle-and-action-masking.md`.
The documentation describes isolated germination/training, blending, holding,
commitment and pruning/cleanup. It permits newer germination while older
seeds blend or hold. Section 5 corrects an earlier full-amplitude shorthand:
holding can occur at the chosen target alpha. These are documentation
observations; the implementation was not validated or imported here.

Retain only the small ideas needed for a future demonstrator: distinct
existence/influence/learning state, frozen shadow formation, gradual blend,
measured hold period, explicit withdrawal and evidence after withdrawal.
Do not copy policy training, reward machinery, dashboards or broad legacy
state machines into this first TCD.

John reports that good Esper-lite seeds had negligible immediate impact and
improved after 3 or 4 steps, while bad seeds immediately tanked. That is
**user-reported and not independently verified**. The archive README defines
step/epoch vocabulary for its documented configuration, but this does not
identify the units or exact runs behind John's observation. Keep those units
unresolved; do not calibrate a new horizon from that number.

Simic was read at commit `454c7314cc2728ceee17513928541d60616bd081`.
Its current-state document records partial graft capture in later work and
an ongoing timing/horizon fleet; the older AGENTS summary of no gain at the
screen scale is not the whole current result. Sultai must not disturb those
runs. Current Simic Tamiyo is an observability role; Sultai's supplied coarse
controller concept follows the request and is not a change to Simic's locked
naming constitution.

## Sequence after this checkpoint

1. Restore exact source imports through the supported route.
2. Keep the CPU instrument as a reproducible software/identifiability check.
3. Pre-register a feasible real-host task and true observable target contract;
   evaluate coarse, paired and paired-plus-probe evidence with leakage guards.
4. Compare retrieval/interpolation, deterministic generation and diffusion
   only after a useful conventional repair reference is established. A failed
   reference remains a bounded failure to find a repair, not unrepairability.
5. Add a predeclared learning horizon, physically removable structure and
   post-withdrawal assessment before trajectory claims. Count all retained
   inference assets; compare the final architecture trained from scratch.
