"""Run existing strict tools and prove their configured gates reject bad input."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Check:
    name: str
    command: tuple[str, ...]
    expected_exit: int
    actual_exit: int
    output: str


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mypy", default="mypy", help="Existing mypy executable; no automatic installation.")
    parser.add_argument("--ruff", default="ruff", help="Existing Ruff executable; no automatic installation.")
    parser.add_argument("--output", type=Path, help="Optional JSON evidence destination outside source inputs.")
    args = parser.parse_args()
    mypy = shutil.which(args.mypy)
    ruff = shutil.which(args.ruff)
    if mypy is None or ruff is None:
        parser.error("Both configured mypy and Ruff executables must exist; install the dev extras or supply paths.")
    root = Path(__file__).resolve().parents[1]
    sources = sorted((root / "src" / "sultai").glob("*.py"))
    if not sources:
        parser.error("No Sultai production modules found; refusing an empty gate.")
    source_hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    checks: list[Check] = []

    def check(name: str, command: list[str], expected: int) -> None:
        completed = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=60, check=False)
        checks.append(Check(name, tuple(command), expected, completed.returncode, completed.stdout + completed.stderr))

    with tempfile.TemporaryDirectory(prefix="sultai-quality-") as temporary:
        cache = Path(temporary)
        config = str(root / "pyproject.toml")
        mypy_base = [mypy, "--config-file", config, "--cache-dir", str(cache / "mypy")]
        ruff_base = [ruff, "check", "--no-cache", "--config", config]
        check("mypy_version", [mypy, "--version"], 0)
        check("ruff_version", [ruff, "--version"], 0)
        check("strict_source_types", [*mypy_base, "src/sultai"], 0)
        check("source_and_test_lint", [*ruff_base, "src", "tests"], 0)
        check(
            "source_and_test_format", [ruff, "format", "--check", "--no-cache", "--config", config, "src", "tests"], 0
        )

        fixtures = {
            "good.py": 'from typing import TypedDict\n\n\nclass Measurement(TypedDict):\n    value: float\n\n\ndef read(item: Measurement) -> float:\n    return item["value"]\n',
            "missing_measurement.py": "from typing import TypedDict\n\n\nclass Measurement(TypedDict):\n    value: float\n\n\nitem: Measurement = {}\n",
            "wrong_type.py": 'def measured() -> float:\n    return "missing"\n',
            "untyped.py": "def missing_contract(value):\n    return value\n",
            "explicit_any.py": "from typing import Any\n\n\ndef hidden(value: Any) -> Any:\n    return value\n",
            "lint_bad.py": "import os\n\nvalue = undefined_measurement\n",
            "format_bad.py": "value=1\n",
        }
        for name, source in fixtures.items():
            (cache / name).write_text(source)
        check("type_positive_control", [*mypy_base, str(cache / "good.py")], 0)
        check("lint_positive_control", [*ruff_base, str(cache / "good.py")], 0)
        check(
            "format_positive_control",
            [ruff, "format", "--check", "--no-cache", "--config", config, str(cache / "good.py")],
            0,
        )
        for name in ("missing_measurement.py", "wrong_type.py", "untyped.py", "explicit_any.py"):
            check(f"type_negative_{name}", [*mypy_base, str(cache / name)], 1)
        check("lint_negative_control", [*ruff_base, str(cache / "lint_bad.py")], 1)
        check(
            "format_negative_control",
            [ruff, "format", "--check", "--no-cache", "--config", config, str(cache / "format_bad.py")],
            1,
        )
        # Mutate a disposable copy of a real scanned module, not the checkout.
        copied_source = cache / "mutation" / "src"
        shutil.copytree(root / "src", copied_source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        mutated = copied_source / "sultai" / "repair.py"
        with mutated.open("a") as stream:
            stream.write('\n\ndef _quality_type_mutation() -> int:\n    return "invalid measurement"\n')
        check("real_source_type_mutation", [*mypy_base, str(copied_source / "sultai")], 1)
        if "[return-value]" not in checks[-1].output or "repair.py" not in checks[-1].output:
            raise RuntimeError("Type mutation failed for the wrong reason; expected repair.py return-value diagnostic.")
        with mutated.open("a") as stream:
            stream.write("\n_quality_lint_mutation = MISSING_MEASUREMENT_FOR_GATE_CONTROL\n")
        check("real_source_lint_mutation", [*ruff_base, str(copied_source)], 1)
        if "F821" not in checks[-1].output or "MISSING_MEASUREMENT_FOR_GATE_CONTROL" not in checks[-1].output:
            raise RuntimeError(
                "Lint mutation failed for the wrong reason; expected the undefined measurement diagnostic."
            )
    after = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / "src" / "sultai").glob("*.py"))
    }
    if source_hashes != after:
        raise RuntimeError("Source changed during quality checks; results do not bind a stable tree.")
    passed = bool(checks) and all(item.actual_exit == item.expected_exit for item in checks)
    report = {
        "passed": passed,
        "source_files": source_hashes,
        "source_unchanged": True,
        "checks": [asdict(item) for item in checks],
    }
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.write_text(serialized)
    print(serialized, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
