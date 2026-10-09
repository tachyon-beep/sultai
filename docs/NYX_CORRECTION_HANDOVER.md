# Nyx correction confirmation: owner handover

Parent owns safe synchronization and launch after confirming PR #1 merged.
These commands have not run on Nyx. They require existing Python 3, Git,
`nice`, GNU `timeout` and `/usr/bin/time`; the experiment has no third-party
runtime dependency, network call, paid service or GPU use. Preserve local work,
read Nyx's own AGENTS/tracker state and inspect memory/CPU/ELSPETH streaming
load first. Do not change Simic, its sessions, schedules or experiments.

## Exact scope and resource envelope

Run the reviewed experiment commit
`64439d80bdd7897aa94f8aa819ab23c17fcaffba` after proving it is merged and its
executable inputs equal merged main. Keeping this exact commit makes source
identity directly comparable to the development record. Later integration
commits contain docs/evidence only. Never substitute the current tip if it
changes executable inputs or the frozen protocol.

Host: fixed 16-channel same-feature synthetic system. Action: public four-template
nonlinear residual materialized as 272 parameters. Information: privileged
paired conditioning residuals; selection separate; clean final audit sealed.
Formation offline training 1000..1031 is intentionally shared with development;
confirmation evaluation 5000..5007, clean healthy 15000..15007, noisy healthy
25000..25007. Trajectory roots 9000, 9010, 9020, 9030, each with offsets
+1 training / +2 conditioning / +3 audit / +4 random. All other settings and
success/stop meanings are in the immutable protocol. This is confirmation of
an instrument, not a broader transfer burn or readiness pass.

One process at a time, nice 10, 120 seconds maximum per invocation. Run one
confirmation and one exact replay, sequentially; no tuning/retries. Cloud
development (two lifecycle streams) measured 16.13 seconds wall, 16.12 CPU,
62.7 MiB peak RSS. Four-stream confirmation is estimated at 25–40 seconds and
below 256 MiB on comparable hardware; Nyx performance is unmeasured. No GPU.
Reserve less than 10 MiB for JSON/text evidence. Stop if host load is unsuitable;
these estimates do not authorize a new campaign or extend the time ceiling.

## Safe source check and isolated checkout

Run from the existing Sultai repository after inspecting its local status.
The final parent report supplies the verified PR merge SHA; verify origin/main
includes that commit before proceeding. The following checks preserve the
original checkout and refuse changed executable inputs:

```sh
set -eu
cd /home/john/sultai
git status --short
git worktree list
git remote -v
git fetch origin main
sultai_source=64439d80bdd7897aa94f8aa819ab23c17fcaffba
git merge-base --is-ancestor "$sultai_source" origin/main
git diff --exit-code "$sultai_source" origin/main -- src scripts tests pyproject.toml docs/plans/2026-10-09-correction-protocol.md
sultai_work=/tmp/sultai-confirmation-64439d8
test ! -e "$sultai_work"
sultai_evidence=$(mktemp -d /tmp/sultai-confirmation-evidence.XXXXXX)
git show origin/main:docs/results/correction-2026-10-09/development.json > "$sultai_evidence/development.json"
git worktree add --detach "$sultai_work" "$sultai_source"
cd "$sultai_work"
test "$(git rev-parse HEAD)" = "$sultai_source"
test -z "$(git status --porcelain)"
python3 --version
sha256sum docs/plans/2026-10-09-correction-protocol.md
```

Expected plan SHA-256:
`abbf3f52d1ef4eb148536b3cc0fc5bb13a24962a5353f55d739c541486bda486`.
If source availability, ancestry, diff, status or identity checks fail, stop.
Do not reset/stash another owner's work or silently run a different revision.
The temporary worktree/path must not already exist; a prior attempt requires
explicit adjudication, not a fresh budget under a new name.

## Fixed execution and replay

Continue in the same shell so the task-specific variables remain defined.
This wrapper fixes the CPU priority, threads and 120-second timeout. `/usr/bin/time`
records exit/resource data independently of the scientific JSON. An existing
JSON path is refused and only a complete report is atomically published.

```sh
/usr/bin/time -v -o "$sultai_evidence/confirmation-time.txt" sh scripts/correction.sh --phase confirmation --output "$sultai_evidence/confirmation.json" > "$sultai_evidence/confirmation.stdout" 2> "$sultai_evidence/confirmation.stderr"
/usr/bin/time -v -o "$sultai_evidence/replay-time.txt" sh scripts/correction.sh --phase confirmation --output "$sultai_evidence/replay.json" > "$sultai_evidence/replay.stdout" 2> "$sultai_evidence/replay.stderr"
cmp "$sultai_evidence/confirmation.json" "$sultai_evidence/replay.json"
sha256sum "$sultai_evidence"/*
```

The earlier `set -e` stops after a nonzero invocation or byte mismatch. Exit 0
means instrument validity, including a scientific negative; exit 2 means invalid
checks. Timeout/nonfinite/provenance/code faults are failed attempts. No automatic
retry, parameter change, repair suppression, budget reset or evidence overwrite.
Record and retain stderr, time/exit, partial files and worktree state. Restart
only after the parent explicitly links and dispositions the failed attempt.

## Provenance and adjudication

This bounded post-run check reads saved evidence; it executes no cohorts.
Use normal Python (not `-O`, which disables assertions):

```sh
python3 - "$sultai_evidence" <<'PY'
import hashlib, json, pathlib, sys
p = pathlib.Path(sys.argv[1])
d = json.loads((p / 'development.json').read_text())
c = json.loads((p / 'confirmation.json').read_text())
assert hashlib.sha256((p / 'development.json').read_bytes()).hexdigest() == '0fab1abb37b5d6a15c4082f154978e6c8b86f62503e6ed71cc6232f5dc9de97e'
assert (p / 'confirmation.json').read_bytes() == (p / 'replay.json').read_bytes()
assert d['phase'] == 'development' and c['phase'] == 'confirmation'
assert c['identity']['source_commit'] == '64439d80bdd7897aa94f8aa819ab23c17fcaffba'
assert c['identity']['plan_sha256'] == d['identity']['plan_sha256']
assert c['source_identity'] == d['source_identity']
assert c['software_valid'] and not c['training_ready']
dp = {x['name']: x for x in d['formation']['provenance']}
cp = {x['name']: x for x in c['formation']['provenance']}
assert dp['train'] == cp['train']  # shared frozen offline training is deliberate
held = ('evaluation', 'clean_healthy', 'noisy_healthy')
dev_ids = {x for n in held for x in dp[n]['lineages']}
confirm_ids = {x for n in held for x in cp[n]['lineages']}
assert not dev_ids.intersection(confirm_ids)
for name in held:
    assert len(cp[name]['lineages']) == 8
    assert cp[name]['sample_count'] == 1216
    assert cp[name]['sample_ids_sha256'] != dp[name]['sample_ids_sha256']
    assert cp[name]['pairs_sha256'] != dp[name]['pairs_sha256']
assert [s['root'] for s in c['trajectory']['streams']] == [9000, 9010, 9020, 9030]
assert not {s['root'] for s in c['trajectory']['streams']}.intersection(s['root'] for s in d['trajectory']['streams'])
print('Replay and report provenance checks passed; scientific verdicts follow:')
print(json.dumps({'gates': c['gates'], 'specificity': c['diagnostic_specificity'], 'contrasts': c['trajectory']['contrast_summary']}, indent=2))
PY
```

Different digests alone do not prove disjoint sets. The checked source constructs
formation sample IDs from seed/partition/index and lifecycle IDs from explicit
stream/partition names; fixed disjoint seeds/names plus the runtime global
isolation guards supply that argument. Inspect expected seed/partition records
and all harmful admissions, false rejections, contrasts and removal metrics.
Shared training data are intentional; held-out development and confirmation
lineages must not overlap. Record runtime differences from cloud without
rewriting the original result.

Parent should preserve confirmation/replay JSON, resource logs, Git commit/tree,
source/plan hashes and independent adjudication together. Update the certificate
and Page with pass/fail/stop outcomes and the evidence links. Gate 2 can close
only after all checks and replay genuinely succeed; gates 3/5/6/7 do not inherit
that pass. Development's specificity failure remains regardless of confirmation.

The recommended separate transfer screen, proposed utility/risk margins and
resource alternatives are in `plans/2026-10-09-nyx-transfer-proposal.md`.
John's choice of domain/information/margins/ceiling remains required before
implementing or launching it. The present commands do not launch that work.
