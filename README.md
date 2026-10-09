# Sultai

Sultai studies neural development through **addition and deliberate removal**.
The long-term target is equal capability with roughly half the final retained
parameters, starting near-empty. Training and search cost are secondary but
must be reported separately. No result here establishes that target.

TCD means **technical concept demonstrator**, as clarified by John. This
repository begins as a restart-safe research and design checkpoint on
Nyx, created on 2026-10-09. ELSPETH landing takes priority. No GPU work, long
training, paid service, remote publication, or changes to Simic/Esper are part
of this checkpoint.

- [Resume brief](docs/RESUME.md): exact state, commands, blockers, next action.
- [Canonical concept](docs/CONCEPT.md): consolidated intent, mechanism and evidence.
- [Decision register](docs/DECISIONS.md): settled corrections and open choices.
- [Scope and decisions](docs/SCOPE.md): bounded TCD and later research gates.
- [Bounded design](docs/DESIGN.md): one-site local repair proposal.
- [Test plan](docs/TEST_PLAN.md): validity checks and staged comparisons.
- [Research pack](docs/research/README.md): sources, evidence status, corrections.
- [Runnable TCD](docs/SMOKE.md): implemented behavior, commands and limits.

Run the CPU instrument without installing dependencies:

```sh
cd /home/john/sultai
sh scripts/smoke.sh
```

The first TCD checks a nonlinear 272-parameter adapter, conventional ridge
repair on planted targets, split/selection boundaries, and a fixed shadow-probe
ambiguity example. It is not yet the full learned local repair comparison,
growth/removal system, or a test of half-size final models. The three exact
original research imports and newer Claude source bundle remain blocked;
readable Library text informed the annotated research pack. See the
[source map](docs/research/SOURCE_MAP.md) and
[handoff status](docs/research/CLAUDE_HANDOFF.md). Claude's independent
adversarial TCD design is still pending and has not been steered by this task.
