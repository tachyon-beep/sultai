# Claude source handoff and independent design boundary

Status supplied by the parent on 2026-10-09: John separately commissioned a
TCD design as an **independent adversarial review** in the
[Claude Sultai project](https://claude.ai/code/project/chan_01Lkr1WDXsFdduxjV1A2mf41).
The browser confirmed a cloud `/mnt/project-files` workspace, not direct Nyx
access. This repository has not received that independent design yet.

## Historical source bundle: readable, byte transfer blocked

The parent supplied `sultai-claude-handoff-bundle.md`
(`libfile_c1a641abe658819187e616c9dffbf020`) and `verified-manifest.json`
(`libfile_44706a3c3ff48191883e4bce66894fa1`). Library returned all 719 and
22 text lines respectively. A fresh supported consumer-side materialization
for each requested file on Nyx failed with the exact error:

```text
library file transfer failed: download failed with HTTP status 403
```

No source files landed in the reserved directory; there are no Nyx byte counts
or verified checksum matches. [sources.json](sources.json) records identities,
reported sizes, and the four embedded reports' expected checksums from the
readable manifest. These are reported checksums, not local verification.
The source bundle contains historical Esper/Simic/report comparisons and a
Claude research report; it explicitly excludes the independent TCD still in
progress. Its insights are attributed as H1–H4 in [SOURCE_MAP.md](SOURCE_MAP.md).
Do not treat this bundle as a verified replacement for the three original
attachments or reconstruct it from API text. No further transfer retry is
planned without a working supported handoff.

## Independent TCD: still pending

`docs/research/claude-handoff/` is reserved for a possible additive contribution.
It was created empty; only the authorized supported import may populate it.
Core-doc consolidation must not add, edit or remove contributions there.
Empty directories are not preserved by Git; after recovery it may
need to be recreated when the transfer owner confirms readiness. Core synthesis,
design and resume files remain owned by the Nyx Sultai coordinator.

Preserve independence: do not send this TCD design, its proposed conclusions
or leading comparison prompts to Claude before its independent work finishes.
No such message has been sent by this Sultai task. The separate work is not
an instruction to restart/duplicate implementation, and does not authorize
new features, a project merge, compute or experiments.

After the parent confirms completion and a supported transfer:

1. Preserve the received source separately, with reported title/revision,
   source path/identity, exact local byte status and checksum when available.
   Cloud paths are not local files. Do not invent an exact import from read text.
2. Record the design's assumptions, hypothesis, minimal experiment, information
   access, controls, failure/success criteria, lifecycle and cost accounting
   before judging it against our design.
3. Compare it with CONCEPT.md, DESIGN.md and DECISIONS.md. Separate agreement,
   factual/mathematical corrections, different scope and genuine tradeoffs.
4. Add material conflicts to DECISIONS.md with evidence and owner/action;
   do not silently rewrite either original or automatically expand code.

The next comparison has not happened. Pending transfer is not a reason to
hold the completed local documentation or CPU instrument uncommitted.
