# Sultai hybrid TCD — independent review handoff

Please independently review the attached Sultai technical concept demonstrator.
Review the delivered implementation and experimental meaning, not whether it
agrees with an earlier reviewer. Do not modify source or start larger training.
Return prioritized actionable findings with file/line, impact, a concrete
counterexample or reproducer, and a regression test. Distinguish observed
defects, scientific limitations, and optional improvements. A clean review is
also useful; do not invent findings to meet a quota.

## Exact target and access

- Repository on Nyx: `/home/john/sultai`.
- Delivered commit: `ccdfd5225ee5575f365aa6a4a0698bed0f652958`.
- Git tree: `274d65d0b0b958a952cc4be4046b4fb077b50dfa`.
- Implementation milestone: `6b03f89`.
- Local branch: `main`; no remote is configured. There is no GitHub URL to use.
- Attach `sultai-ccdfd52-review.zip` alongside this prompt. It contains 37
  allowlisted tracked files, each byte-exact from that commit, and
  `SOURCE_MANIFEST.json` with file sizes and SHA-256 hashes. It excludes Git
  metadata, tracker databases, credentials, private logs and prior review reports.
- Archive SHA-256:
  `572fad1389c6607653f4d4bce6de471d6e3c490121235158061f0ee980a396b3`.

Claude web may have no access to Nyx or its local paths. The attachment, rather
than a path mention, provides source access. If archive extraction is unavailable,
request the individual files listed below; do not claim to have inspected code
that was not supplied. The archive is a source snapshot, not a Git checkout;
its provenance comes from the manifest. Runtime and snapshot integrity claims
should be checked where your environment permits. Do not run repository
maintenance or tracking commands merely because AGENTS.md describes them;
this is an external read-only review.

## Design names and intent

**emmy design:** the original Sultai research/design and conventional synthetic
repair instrument at `c720bcef55b58bbce6025265ec14a46e5fdb5922`; runtime began at
`6d655e4`. Its design snapshot is `docs/designs/emmy-DESIGN-c720bce.md`.

**Claude design:** an independently authored concept-demonstrator proposal,
followed by the Emrakul/Tamiyo role revision. Read
`docs/research/claude-handoff/sultai-concept-demonstrator-design.md` and
`docs/research/claude-handoff/emrakul-tamiyo-revision.md`. Other imported reports
and the provenance index are included. Proposals are not measured results.

**Hybrid design:** the implementation contract in `docs/HYBRID_CONTRACT.md`,
combining a nonlinear primary repair with labeled affine diagnostics, bounded
learned formation and separate lifecycle mechanics. `docs/designs/README.md`
explains provenance; `docs/DESIGN.md` separates active and historical scope.

Long-term Sultai intent is addition AND deliberate removal jointly driving
development from near-empty toward equal capability with roughly half the
final retained parameters. Training/search cost is secondary and reported
separately. Temporary structure may shape later learning and disappear.
Diffusion is a candidate, not a requirement; denoising is not parameter removal.

In the hybrid, Tamiyo's coarse opportunity nomination is fixed. Sultai Emrakul
forms and admits an edit or chooses no-op using local information. The seed may
collect paired telemetry and probe a frozen shadow host; Tamiyo need not
diagnose the full defect. No trained Tamiyo policy is implemented here.

## Implemented scope

The standard-library Python runner joins two assays:

1. **Formation/admission:** nonlinear residual adapter
   `h + alpha * (W tanh(h) + b)` at one 16-channel site, with 272 stored values.
   Offline conventional ridge teachers supervise a tiny deterministic generator.
   It forms repairs from permitted conditioning evidence on procedural held-out
   lineages. Coarse, paired, paired-plus-identity-probe evidence, ridge, affine,
   short SGD, retrieval/interpolation, shuffled conditioning, residual mispairing
   and no-op comparisons are included. Selection and final audit examples are
   separate; each method offers one candidate plus no-op.
2. **Lifecycle:** a 272-value trainable host receives a temporary 272-value
   repair from the frozen paired generator. Task-gradient learning, fixed hold,
   taper, actual deletion and further learning are compared with no-growth,
   cold/random addition and early-static controls. Separate affine-folding and
   signed nonlinear probe diagnostics are included.

The corpus has 32 training, 8 development and 8 test procedural lineages from
one shared public four-template family, with 64 conditioning, 24 selection and
64 test examples each. Eight healthy counterparts are also evaluated. Adapter
output has four effective degrees of freedom despite 272 stored values.
Direct oracle residuals are privileged; paired moments contain gradient
information. Identity-host probes are executed but add no identifying features.

Lifecycle begins with 272 allocated host values, shares the target's feature
class, and uses a scheduled insertion/taper/removal rather than a learned
removal policy. Its insertion is raw; empirical admission is tested separately.
Host/temporary weights could be algebraically combined, but the experiment
claims actual task-gradient learning rather than teacher-weight copying.

These fixtures do not establish unseen-family or real-host transfer,
population safety, near-empty growth, diffusion, a developmental advantage,
or the final half-parameter goal. A 544-to-272 temporary-peak reduction is not
a comparison against a tuned half-size reference. Read all control results,
including unfavorable ones, rather than interpreting overall pass as a
comparative win.

## Files and commands

Start with:
- `docs/HYBRID_CONTRACT.md`, `docs/HYBRID_DEMO.md`, `docs/TEST_PLAN.md`.
- `src/sultai/repair.py`, `formation.py`, `lifecycle.py`, `hybrid.py`.
- `tests/test_repair.py`, `test_formation.py`, `test_lifecycle.py`,
  `test_hybrid_cli.py`.
- `scripts/hybrid.sh` and the two files in `docs/results/`.
- The source design files named above for intent and provenance.

Python 3.10+ and the standard library suffice. No pip install, model download,
service credential, GPU or network is required. On a Linux environment with
`nice` and `timeout`, from the extracted `sultai-ccdfd52/` directory:

```sh
sh scripts/hybrid.sh --summary
sh scripts/hybrid.sh > /tmp/sultai-independent-review.json
nice -n 10 timeout 120 env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests -v
```

For an environment lacking those shell utilities, use a separately bounded
Python invocation with `src` on PYTHONPATH:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m sultai.hybrid --summary
```

The author reports 37 tests passing and two byte-identical final demo runs
with 16 passing gates. Treat these as reported evidence to inspect, not proof
of correctness. Source identities and deterministic metrics are in the full
JSON; load-dependent timing is stored separately. Isolate temporary tests and
outputs outside the source snapshot. Do not increase the 120-second demo
budget, run a training campaign, install services, publish code, or touch
Simic/Esper/ELSPETH repositories.

## Questions for independent review

- Do the implementation and metrics support the bounded formation and lifecycle
  claims? Are any controls misleading or comparisons confounded beyond the
  declared limits?
- Can missing, invalid or non-finite required state become a plausible value,
  an accepted repair, an apparent no-op, or a successful report? Are defaults
  justified for genuinely optional fields?
- Do partitioning, conditioning, frozen candidate banks, selection and audit
  preserve the declared information boundaries, including adversarial cases?
- Are rejection/no-op/acted outcomes, denominators and resource/parameter costs
  counted consistently? Are training and retained inference assets separated?
- Do lifecycle transitions and serialization establish actual removal, rather
  than zero influence alone? Do reported learning trajectories justify their
  interpretation?
- Which tests could pass despite a violated semantic contract, and what focused
  regression would detect it? What would falsify the current interpretation?

Please keep recommendations bounded. A future real-host or larger experiment
requires a separate task, protocol and compute authorization.

## Source import status

The three original pasted Library attachments requested early in the project
are not present as exact local originals: supported transfers failed after
bounded retries. Related Claude source reports were separately delivered and
hash-verified; they do not substitute for those originals. The independent
Claude TCD local copy includes a separately identified revision-pointer
paragraph. Consult `docs/research/sources.json` and the source index for exact
provenance; do not silently reconstruct or relabel absent originals.
