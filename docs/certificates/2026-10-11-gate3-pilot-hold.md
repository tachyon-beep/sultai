# Gate 3 pilot execution hold — 2026-10-11

**HOLD. Execution invalid; no scientific result. No retry or confirmation.**

The sole authorized pilot was admitted at source/evidence commit
`2e128065eb512596fd70b006ef0435d8d6ce8090`, with independently accepted
implementation `4936d28c90e9ab3a4e7358c772ae0b7acbaffa54`, aggregate
`74f1c403562d13fa37c724ab6ca358642dbaf025c265a82598303c60f4c086a8`.
[Draft PR #3](https://github.com/tachyon-beep/sultai/pull/3) remains unmerged.
The prior CPU evidence (20 quality checks, 137 runtime tests) and specialist
acceptance are engineering evidence, not a successful GPU runtime result.

Fresh host admission at 09:22 UTC / 20:22 AEDT verified both exact GPU UUIDs
idle, no compute/screen process, released Simic terminals, exclusive owner,
no existing attempt, clean committed source and bound review certificate.
Both initial children then exited 1 at the provenance wheel guard. Supervisor,
GNU time and outer execution exits are 2. No dataset decoding, host training,
checkpoint, loss stream, scientific gain or sizing result was reached.

The guard compared `importlib.metadata.version("torch")` to `2.9.1+cu128`.
Actual host values are distribution metadata **2.9.1**, runtime build
**2.9.1+cu128**, CUDA build **12.8**; numpy distribution/runtime are **2.3.5**.
This is a software contract-field mismatch, not GPU unavailability. The owner
missed this distinction in preflight. The early provenance guard sits outside
the worker's structured failure block, so no child `failure.json` exists;
traceback signatures and raw-log hashes are retained without fabricating it.

The ledger conservatively charges **2.370860553928651 aggregate GPU
process-seconds** (0.000658572376 hours), within the unchanged 10,800-second
ceiling. GNU time records **2.40 seconds wall**, **664,400 KiB peak RSS**.
Both child return codes and charged times remain in the ledger. Host post-stop
verification at 09:25:12 UTC / 20:25:12 AEDT shows both GPUs 0% / 15 MiB,
no compute process and no Sultai screen process. No other workload was stopped.

[Published evidence](../results/gate3-pilot-2026-10-11-01a125af/manifest.json)
byte-preserves admission, ledger, process receipts, attempt custody, preflight,
post-stop and GNU-time records. Private raw logs remain in ignored local
runtime storage; their hashes are published. Model weights were never created.
The original `.weft/gate3-pilot-attempt.json` remains untouched. Output-path
changes, receipt deletion, automatic restart or budget reset are forbidden.

## Concrete disposition requested from John

The frozen protocol §9 stop rule and D-P hold-on-deviation condition require
operator disposition before another attempt. No source repair is applied after
the failed attempt in this checkpoint. Proposed bounded repair:

1. Validate separate pinned fields: distribution torch `2.9.1`, runtime
   torch `2.9.1+cu128`, CUDA build `12.8`, numpy distribution/runtime `2.3.5`.
   Align the optional dependency declaration with distribution metadata.
   No installation, environment change or weaker build check is proposed.
2. Run these environment checks in parent admission **before** consuming
   attempt custody or starting children; retain explicit mismatch evidence.
3. Add CPU positive/negative controls for the actual metadata/runtime split,
   re-run affected quality/runtime gates, and obtain a targeted independent
   review of the exact repair commit and updated admission certificate.
4. Only with John's explicit disposition, create a linked new-attempt record
   retaining this failed receipt and charge. The **same overall 3 GPU-hour
   ceiling** leaves **10,797.629139446071 process-seconds**; no budget reset,
   scientific threshold change, new environment or automatic extension.
   Recheck actual resources and run only the development pilot.

Confirmation remains blocked regardless of this disposition. A clean pilot,
allowed fills, reviewed/committed freeze and the existing conditional terms
are still required. Missing evidence remains missing.
