# 00 · Index, provenance and conflicts note

Delivered to `docs/research/claude-handoff/` on 2026-10-09 by the Claude project thread "Learn from the esper-lite attempt", through the Claude Code session running on Nyx. Each file in this directory is a byte-exact copy of the project file listed below; the SHA-256 is of that file. The text after the rule below is the header of the project's handoff bundle (`/mnt/project-files/handoff/sultai-claude-handoff-bundle.md`). Its "where this bundle lives" section describes the situation before this delivery.

| Delivered file | Project source path | SHA-256 |
|---|---|---|
| `esper-lite-review.md` | `notes/esper-lite-review.md` | `4673c4f08ff80d46d463079c83e5fdf4514d5e2e8c8a5155654f1d7369eda306` |
| `simic-seed-comparison.md` | `notes/simic-seed-comparison.md` | `d31ca487ad0050e8280a4736157e2b0444cde01acdb17a84d3e5eb242113c702` |
| `openai-report-vs-repos.md` | `notes/openai-report-vs-repos.md` | `8c86478d7a68c95c217ee7b29a5af45895ee9f3b2b47e2059f2d235d0a019564` |
| `simic-diffusion-grown-modules.md` | `research/simic-diffusion-grown-modules.md` | `4d3f5aa845766c2a34406213e2c4d39b65113a3d640d8614e5da76eb29e1d69d` |
| `sultai-concept-demonstrator-design.md` | `sultai-tcd/reports/Sultai concept demonstrator design.md` | `e376250015bad71675f2194636bf82e85de9638c1d1f820b97e7ef1921ad38cf` |
| `emrakul-tamiyo-revision.md` | `sultai-tcd/emrakul-tamiyo-revision.md` | `64b622478453f394eac55d258d1bdc77f0082c10b37968d24665e31f9e199493` |

---

# Sultai: Claude project handoff bundle

Assembled 2026-10-09 for the Sultai owner. These are source reports and supplementary notes, not a synthesis or a design.

## Where this bundle lives

This bundle was **not** written to Nyx or to `/home/john/sultai`. The session that built it only had the Claude project's cloud filesystem. Environment checks, all read-only:

- Container hostname: `vm`, user `root`, Linux 6.18.44.
- `/home/john/sultai` does not exist in the container: `ls: cannot access '/home/john/sultai': No such file or directory`.
- The project files are an rclone FUSE mount of the project file store: `/mnt/project-files` resolves to `/mnt/attach/project-files`, mounted from `rclone-filestore:pjfs_01VEohiiXXqiV8tyKnotvaT1`.
- The linked desktop device is `velma` (Windows, `win32`). It has **no connected folders** and is not Nyx. No new access was requested.
- No git remote applies, because no Sultai repo is reachable.

To place these files in `/home/john/sultai/docs/research/claude-handoff/`, copy this bundle, or split it at the `BEGIN FILE` / `END FILE` markers.

## Provenance index

Each file below is included verbatim. The SHA-256 is of the source file at bundle time.

| # | Source path (project files) | Produced by | Contents | Bytes | SHA-256 |
|---|---|---|---|---|---|
| 1 | `notes/esper-lite-review.md` | Claude thread 'Learn from the esper-lite attempt' (this thread), 2026-10-09 | esper-lite @ 2d296b6c read-only review: architecture, telemetry collapse, what carries into Sultai | 6825 | `4673c4f08ff80d46d463079c83e5fdf4514d5e2e8c8a5155654f1d7369eda306` |
| 2 | `notes/simic-seed-comparison.md` | Same thread, 2026-10-09 (correction note added after the OpenAI report) | simic @ 16bcf70 vs esper-lite seed design: what simic kept, what Sultai would replace | 6253 | `d31ca487ad0050e8280a4736157e2b0444cde01acdb17a84d3e5eb242113c702` |
| 3 | `notes/openai-report-vs-repos.md` | Same thread, 2026-10-09 | Three-way comparison: John's OpenAI report vs the project research report vs esper-lite/simic | 6391 | `8c86478d7a68c95c217ee7b29a5af45895ee9f3b2b47e2059f2d235d0a019564` |
| 4 | `research/simic-diffusion-grown-modules.md` | Claude thread '/deep-research' (separate thread), 2026-10-09 | Deep-research report on diffusion-grown modules; evidence tags [V]/[E]/[K] | 48427 | `4d3f5aa845766c2a34406213e2c4d39b65113a3d640d8614e5da76eb29e1d69d` |
| 5 | `sultai-tcd/reports/Sultai concept demonstrator design.md` | Claude thread 'Sultai demonstrator design', 2026-10-09. **A design that may still be revised.** It now opens with a pointer to revision 2, which overrides it where the two differ | TCD design: rung 1a frozen-host generated repair and rung 1b live capture, built in simic's conventions | 60395 | `e376250015bad71675f2194636bf82e85de9638c1d1f820b97e7ef1921ad38cf` |
| 6 | `sultai-tcd/emrakul-tamiyo-revision.md` | Claude thread 'Sultai demonstrator design', 2026-10-09, after John set the roles. **A design revision that may still change** | TCD revision 2: Emrakul is the generator and owns each site, including no-op and site safety; edits can be additive or reductive; Tamiyo forecasts opportunity; Tamiyo's training signal depends on the status Emrakul returns; later sections cover the crisis start, coordinated interventions, Emrakul sizing and scope for the first build | 23027 | `64b622478453f394eac55d258d1bdc77f0082c10b37968d24665e31f9e199493` |

**Referenced but not included:**
- **OpenAI deep-research report**, "Simic: Local Generative Overgrowth and Diffusion-Grown Neural Modules", plus John's follow-up Q&A. John uploaded it on 2026-10-09; it sits in project files at `uploads/hearth/29f23cee-10a9-4b2a-ae47-52624c81c33b` (93437 bytes, SHA-256 `22a4b45ee0df903be456b82c60288a61ef5713121149aa9e44693dc0524c94d6`). The owner already has it, so it is left out here.
- **TCD research notes.** These sit next to the TCD design, under `sultai-tcd/research_notes/Sultai concept demonstrator design/`: controls_ablation_accounting, forward_only_contract, generator_design, host_and_corpus and simic_harness_reuse. They are not included; the design (file 5) is the deliverable.

**Repositories the reports were read from**, read-only and never modified:
- `github.com/tachyon-beep/esper-lite` @ `2d296b6c` (2026-07-06)
- `github.com/foundryside-dev/simic` @ `16bcf70` (2026-10-09)

## Supplementary note: unresolved conflicts for the owner

These are open points between the sources. They are not decisions.

1. **What counts as "no backprop".** The OpenAI report rules out activation-times-error statistics as gradient-equivalent. The project research report's default site signature includes such a feature: a ridge probe from site activations to `onehot − softmax`. It drops this only in its "P1-strict" variant. One of them has to be the headline protocol.
2. **How "nothing" is represented.** The OpenAI report uses a separate presence/abstention head. The research report makes the empty set a sampled outcome through rank-ordered gains.
3. **The rung-1 module.**
   - OpenAI report: a 272-parameter full-rank 1×1 conv, with no hidden-unit symmetry and hosts branched from a common anchor.
   - Research report: a rank-32 GELU bottleneck in the site's PCA coordinates, using independent runs.
4. **The seed's information contract.** The OpenAI follow-up gives the seed paired, example-level telemetry plus forward-only probes, and tests evidence richness while Tamiyo's signal stays fixed. The research report uses one aggregate site signature with no probes.
5. **Controls each report has that the other lacks.**
   - Only the research report has nearest-neighbour retrieval (the memorisation test) and GradMax gradient initialisation.
   - Only the OpenAI report has shuffled-conditioning and shuffled-probe controls.
6. **A baseline neither report has at rung 1.** Neither tests against static added capacity, which simic's bounded ladder found its graft loses to (31–60% capture of static's gain).
7. **Where rung 1 should run.** Neither report inspected simic. The research report proposes esper-lite's `cifar_baseline`. simic's bounded CIFAR harness already has synthetic host pathologies, paired multi-seed screens, pre-registration and `verify_run`.
8. **What "Tamiyo" means.** Both reports and esper-lite use Tamiyo as the controller. In simic's HLD, Tamiyo is a non-steering witness, and the commissioning role is Aurelia's. simic's routing rule already matches the OpenAI follow-up's split: Aurelia sends only a brief, and Nissa publishes evidence directly to Momir (INV-07/09).
9. **What "ablation" means in "simic + ablation".** No source defines ablation as a mechanism; the reports use it only for controls. simic `01-claim.md:132` and esper-lite's `counterfactual.py` both warn that same-host ablation mixes intrinsic value with acquired dependence.
10. **Telemetry risk.** Telemetry that feeds a learned generator fails quietly, by weakening the model rather than misleading a person. That differs from the esper-lite failure. simic's "observability is inert" rule does not cover it.


### What the TCD design (file 5) proposes on each conflict

These are the design's proposals; the owner has not confirmed them, and the design may still be revised.

- **1. What counts as "no backprop"**: *addressed.* The design says paired activation-times-error telemetry *is* gradient information, though not a backward pass. It runs graded telemetry tiers (T0, T0+P, T1, T1+P) with a gradient-recoverability audit, so the claim is named per tier.
- **2. How "nothing" is represented**: *addressed.* The design uses both: a presence head, and the share of samples that truncate to rank 0. Both are calibrated with Learn-then-Test on held-out lineages, and abstentions are analysed intention-to-treat.
- **3. The rung-1 module**: *addressed.* The 272-parameter 16→16 1×1 conv, compacted by SVD truncation.
- **4. The seed's information contract**: *partly addressed.* The design keeps probe tiers, but leaves loss-observing probes out of the TCD.
- **5. Controls each report lacks**: *addressed.* Retrieval, retrieval plus noise and k-NN are in, as are shuffled conditions and separately shuffled probes.
- **6. Static-capacity baseline**: *addressed.* It is in rung 1b as arm BS, plus a compute-matched variant BS+.
- **7. Where rung 1 should run**: *addressed.* The design uses simic's data split, RNG, common-future pairing, statistics, pre-registration and verification pattern. It needs a new runner, host and telemetry, and edits none of simic's bounded modules.
- **8. What "Tamiyo" means**: *addressed.* The design follows simic Namespec 2.0: Aurelia commissions, Nissa publishes to Momir, Momir generates, Isperia admits against the no-op, and Tamiyo witnesses.
- **9. What "ablation" means**: *proposed, awaiting John.* The design proposes subtraction at formation, plus a retention-ablation contrast against matched no-op branches, with intrinsic value and acquired dependence reported separately.
- **10. Telemetry risk**: *not addressed by name.* The shuffled-condition controls and the gradient-recoverability audit bear on it.
- **11. Role names: Sultai now differs from simic.** In revision 2 (file 6), Emrakul is the generator. It owns each site's outcome and safety, including returning a no-op. Tamiyo forecasts opportunity, which can mean adding or removing. In simic's HLD, Emrakul lyses committed structure only after commit, and Tamiyo is a witness that must not steer. Conflict 8's "addressed" status above refers to the original design (file 5), which used simic's naming; revision 2 supersedes that mapping.




