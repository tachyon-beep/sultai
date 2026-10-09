"""Complete report fixtures derived from the immutable delivered numerical record."""

import copy
import json
from functools import lru_cache
from pathlib import Path

from sultai import hybrid
from sultai.formation import FIXED, guard_lineage_splits, make_episode


@lru_cache(maxsize=1)
def validated_splits():
    return guard_lineage_splits(
        {
            "train": tuple(make_episode(seed) for seed in FIXED.train_seeds),
            "development": tuple(make_episode(seed) for seed in FIXED.dev_seeds),
            "test": tuple(make_episode(seed) for seed in FIXED.test_seeds),
        }
    )


def reference_report():
    path = Path(__file__).resolve().parents[1] / "docs/results/hybrid-cpu-2026-10-09.json"
    report = json.loads(path.read_text())
    isolation = validated_splits()
    if isolation is None:
        raise AssertionError("fixture provenance missing")
    report["formation"]["isolation"] = isolation.report()
    report["protocol"] = hybrid.PROTOCOL
    report["source_identity"] = hybrid.source_identity()
    return copy.deepcopy(report)
