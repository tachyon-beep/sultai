# Research pack and evidence ledger

Start with [the canonical concept](../CONCEPT.md),
[decision register](../DECISIONS.md), and [source map](SOURCE_MAP.md).
The synthesis incorporates the original reports, telemetry follow-up, Astra
reviews, current Simic source audit, historical Claude source bundle and
existing Sultai work. It preserves source disagreements and evidence status.
The independent Claude TCD and later revision are now delivered and compared;
[handoff status](CLAUDE_HANDOFF.md) distinguishes verified local sources from
failed Library byte transfers and still-missing original pasted attachments.

Read [COMPARISON.md](COMPARISON.md) for the grounded report synthesis, checked
primary references, mathematical corrections and minimal lifecycle lessons.
The follow-up [Simic reconciliation](SIMIC_RECONCILIATION.md) checks Claude's
59-line comparison against current source, separates HLD from implementation,
and records the separate-interface reuse recommendation and early static
capacity comparison. It does not change the runnable TCD.

## Original attachment inventory

Originals belong in `originals/`, under their exact supplied names. They have
**not been imported** at this first checkpoint. Current Library preparation
was used with a Nyx-local destination. The supported transfer failed, and the
single supported retry returned HTTP 403 for all three files. No substitute
text, invented URL, cloud-path claim, or reconstructed original was created.

| Exact original name | Library identity | Reported byte count | Attribution |
| --- | --- | ---: | --- |
| Pasted text.txt | libfile_8a7b298915588191b512a92d77443579 | 68158 | OpenAI initial report |
| Pasted text(1).txt | libfile_d643f55842e88191b58ec69df6ee006a | 26238 | Local telemetry follow-up |
| Pasted text(2).txt | libfile_7de516b203408191905c86eb469c63da | 48895 | Claude report |

Reported sizes are Library metadata, not verified local byte counts. A future
successful import must verify readable bytes, exact sizes and SHA-256 hashes
and preserve Library identity. Do not mark the original-import requirement
complete until those checks pass. `sources.json` records the machine-readable
inventory. Do not put signed URLs or transfer credentials in this repository.

## Evidence status at first checkpoint

| Claim | Status | Consequence |
| --- | --- | --- |
| Joint addition/removal and half retained parameters are desired | User-specified objective | Not an achieved result |
| Rich paired seed telemetry and frozen shadow probes | Proposed interface, supplied in delegation | Keep diagnosis out of Tamiyo instruction |
| Diffusion can generate useful local repairs | Research hypothesis | Must beat adequate deterministic/retrieval/optimization controls |
| One 16-channel nonlinear 272-parameter probe is useful | Supplied Astra proposal | Tests repair formation only |
| Esper-lite good seeds hand over with negligible initial impact then improve after 3–4 steps; bad seeds tank | User-reported, unverified; unit of step unresolved | Measure delayed benefit and early harm; no invented step unit |
| Original reports' detailed claims/citations | Complete API text available; selected sections inspected after first commit | Reading succeeded; exact local originals still absent; only listed primary references checked |

## Corrections carried from the delegated comparison

- GELU is not positively homogeneous: absorbing arbitrary input scale into
  output weights is not a valid general canonicalization. Check the function,
  not merely the parameter shape.
- PCA does not automatically align independently trained hosts: signs,
  permutations, rotations and degenerate subspaces remain possible.
- Failed null/teacher repair is not proof that no repair exists.
- Denoising a parameter vector is not physical parameter removal.
- A final half-size architecture that trains equally well from scratch still
  meets the efficiency goal; it weakens a special developmental-path claim.

## Local sources (read-only)

Simic's constitution and current-state document and Esper-lite's README,
ROADMAP and lifecycle reference were read without modification. The newer
Simic current-state document reports partial graft capture after the initial
screen; see COMPARISON.md for context and commit identities. None is a Sultai
measurement. No implementation or private runtime log was copied.

## Verified Claude source delivery

Commit `72f9eeb` delivered six source files plus their provenance index.
All six hashes match the index. The independent original's expected hash is
also reproduced by removing only the delivered report's revision pointer in
memory. The subsequent Emrakul/Tamiyo revision is a separate document. See
`sources.json` and [source-design provenance](../designs/README.md).

The older bundle/manifest downloads failed. Those failures remain in the
ledger and do not negate the verified local delivery. Related delivered
reports are not assumed to be byte-identical to the three missing originals.
The hybrid design incorporates the completed comparison without rewriting
source proposals as facts or adopting their compute estimates.
