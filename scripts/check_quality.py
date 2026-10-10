"""Run existing strict tools and prove their configured gates reject bad input."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from check_policy import SOURCE_ROOTS, PolicyInputError, production_sources

# Cold mypy on a torch-importing module measured 13.7 s wall on Nyx (2026-10-11,
# torch feasibility review); a real screen package is expected at 20-40 s. The
# subprocess ceiling is set well above that so a slow gate fails loudly on its
# own merits rather than on the clock.
CHECK_TIMEOUT_SECONDS = 300


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
    try:
        sources = production_sources(root)
    except PolicyInputError as error:
        parser.error(f"Refusing an empty gate: {error}")
    source_hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    checks: list[Check] = []

    def check(name: str, command: list[str], expected: int) -> None:
        completed = subprocess.run(
            command, cwd=root, text=True, capture_output=True, timeout=CHECK_TIMEOUT_SECONDS, check=False
        )
        checks.append(Check(name, tuple(command), expected, completed.returncode, completed.stdout + completed.stderr))

    with tempfile.TemporaryDirectory(prefix="sultai-quality-") as temporary:
        cache = Path(temporary)
        config = str(root / "pyproject.toml")
        mypy_base = [mypy, "--config-file", config, "--cache-dir", str(cache / "mypy")]
        ruff_base = [ruff, "check", "--no-cache", "--config", config]
        check("mypy_version", [mypy, "--version"], 0)
        check("ruff_version", [ruff, "--version"], 0)
        check("operation_trust_policy", [sys.executable, "scripts/check_policy.py"], 0)
        check("strict_source_types", [*mypy_base, *SOURCE_ROOTS], 0)
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
        # Mutate a disposable copy of one real scanned module per declared root, not the checkout.
        mutation_root = cache / "mutation"
        copied_source = mutation_root / "src"
        shutil.copytree(root / "src", copied_source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copytree(root / "tests", mutation_root / "tests", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy(root / "pyproject.toml", mutation_root / "pyproject.toml")
        targets = {"src/sultai": "repair.py", "src/sultai_screen": "__init__.py"}
        if set(targets) != set(SOURCE_ROOTS):
            raise RuntimeError("Every declared source root needs exactly one mutation target.")
        for relative, filename in targets.items():
            package = Path(relative).name
            mutated = copied_source / package / filename
            marker = f"MISSING_MEASUREMENT_FOR_GATE_CONTROL_{package.upper()}"
            with mutated.open("a") as stream:
                stream.write('\n\ndef _quality_type_mutation() -> int:\n    return "invalid measurement"\n')
            check(f"real_source_type_mutation_{package}", [*mypy_base, str(copied_source / package)], 1)
            if "[return-value]" not in checks[-1].output or filename not in checks[-1].output:
                raise RuntimeError(f"Type mutation failed for the wrong reason; expected {filename} return-value.")
            with mutated.open("a") as stream:
                stream.write(f"\n_quality_lint_mutation = {marker}\n")
            check(f"real_source_lint_mutation_{package}", [*ruff_base, str(copied_source / package)], 1)
            if "F821" not in checks[-1].output or marker not in checks[-1].output:
                raise RuntimeError(f"Lint mutation failed for the wrong reason; expected {marker} undefined.")
        # The policy gate must see the widened root: plant a prohibited Any in the new package only.
        policy_target = copied_source / "sultai_screen" / "__init__.py"
        with policy_target.open("a") as stream:
            stream.write("\nfrom typing import Any as _quality_policy_mutation  # planted by the gate control\n")
        check(
            "real_source_policy_mutation_sultai_screen",
            [sys.executable, "scripts/check_policy.py", "--root", str(mutation_root)],
            1,
        )
        if "sultai_screen/__init__.py" not in checks[-1].output or "prohibited Any" not in checks[-1].output:
            raise RuntimeError(
                "Policy mutation failed for the wrong reason; expected a prohibited Any in sultai_screen."
            )
    after = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in production_sources(root)
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
