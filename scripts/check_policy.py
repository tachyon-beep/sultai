"""Sultai-local trust seam checks, not a general type/value analyzer."""

from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path


class PolicyInputError(ValueError):
    """The policy reference cannot name one test function inside the checkout."""


def ast_fingerprint(node: ast.AST) -> str:
    normalized = ast.dump(node, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(normalized.encode()).hexdigest()


def test_fingerprint(root: Path, reference: str) -> str:
    parts = reference.split("::")
    if len(parts) != 2:
        raise PolicyInputError("paired test must use path::Class.test_method")
    path_text, qualified = parts
    path = (root / path_text).resolve()
    if not path.is_relative_to((root / "tests").resolve()) or path.suffix != ".py" or not path.is_file():
        raise PolicyInputError("paired test must be an existing test file under tests/")
    node: ast.AST = ast.parse(path.read_text(), filename=str(path))
    names = qualified.split(".")
    for name in names:
        if not isinstance(node, (ast.Module, ast.ClassDef)):
            raise PolicyInputError("paired test is not a test function")
        matches = [
            child
            for child in node.body
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == name
        ]
        if len(matches) != 1:
            raise PolicyInputError(f"paired test not found uniquely: {reference}")
        node = matches[0]
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not node.name.startswith("test_"):
        raise PolicyInputError("paired test must name a test function")
    return ast_fingerprint(node)


def call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


def has_unknown_input(annotation: ast.expr | None) -> bool:
    if annotation is None:
        return False
    for part in ast.walk(annotation):
        if isinstance(part, ast.Name) and part.id == "object":
            return True
        if isinstance(part, ast.Constant) and isinstance(part.value, str):
            try:
                parsed = ast.parse(part.value, mode="eval")
            except SyntaxError:
                continue
            if has_unknown_input(parsed.body):
                return True
    return False


def check_policy(root: Path) -> list[str]:
    sources = sorted((root / "src/sultai").rglob("*.py"))
    if not sources:
        return ["empty production surface"]
    errors: list[str] = []
    for path in sources:
        text = path.read_text()
        location = str(path.relative_to(root))
        tree = ast.parse(text, filename=str(path))
        for line, content in enumerate(text.splitlines(), 1):
            if "#" in content and any(
                token in content.partition("#")[2] for token in ("type: ignore", "noqa", "mypy: ignore")
            ):
                errors.append(f"{location}:{line}: prohibited ignore")
        for node in ast.walk(tree):
            at = f"{location}:{getattr(node, 'lineno', 1)}"
            if isinstance(node, ast.Attribute) and node.attr == "get":
                errors.append(f"{at}: prohibited .get")
            if isinstance(node, ast.Name) and node.id in ("Any", "cast"):
                errors.append(f"{at}: prohibited {node.id}")
            if isinstance(node, ast.Attribute) and node.attr in ("Any", "cast"):
                errors.append(f"{at}: prohibited {node.attr}")
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    if alias.name in ("Any", "cast"):
                        errors.append(f"{at}: prohibited {alias.name}")
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
            if node.args.vararg is not None:
                arguments.append(node.args.vararg)
            if node.args.kwarg is not None:
                arguments.append(node.args.kwarg)
            unknown = any(has_unknown_input(argument.annotation) for argument in arguments)
            boundary = not node.name.startswith("_") and (
                unknown or node.name.startswith("parse_") or node.name in ("from_json", "from_dict")
            )
            decorators = [item for item in node.decorator_list if isinstance(item, ast.Call)]
            marked = [item for item in decorators if call_name(item) == "t3_boundary"]
            if boundary and len(marked) != 1:
                errors.append(f"{at}: {node.name} missing T3 boundary decorator")
            for marker in marked:
                keywords = {keyword.arg: keyword.value for keyword in marker.keywords}
                if set(keywords) != {"test", "fingerprint"} or any(
                    not isinstance(value, ast.Constant) or not isinstance(value.value, str)
                    for value in keywords.values()
                ):
                    errors.append(f"{at}: T3 requires literal paired test and fingerprint")
                    continue
                test_node, fingerprint_node = keywords["test"], keywords["fingerprint"]
                if not isinstance(test_node, ast.Constant) or not isinstance(fingerprint_node, ast.Constant):
                    raise RuntimeError("policy narrowing contradicted")
                reference, fingerprint = test_node.value, fingerprint_node.value
                if not isinstance(reference, str) or not isinstance(fingerprint, str):
                    raise RuntimeError("policy literal narrowing contradicted")
                try:
                    current = test_fingerprint(root, reference)
                except (PolicyInputError, SyntaxError) as error:
                    errors.append(f"{at}: invalid paired test: {error}")
                    continue
                if fingerprint != current:
                    errors.append(f"{at}: stale test fingerprint for {reference}")
            for marker in decorators:
                if call_name(marker) != "t2_operation":
                    continue
                keywords = {keyword.arg: keyword.value for keyword in marker.keywords}
                if set(keywords) != {"invariants", "failures"}:
                    errors.append(f"{at}: T2 requires invariants and failures")
                    continue
                invariants, failures = keywords["invariants"], keywords["failures"]
                if (
                    not isinstance(invariants, ast.Constant)
                    or not isinstance(invariants.value, str)
                    or not invariants.value
                ):
                    errors.append(f"{at}: T2 requires documented operation invariants")
                if (
                    not isinstance(failures, ast.Tuple)
                    or not failures.elts
                    or any(not isinstance(item, (ast.Name, ast.Attribute)) for item in failures.elts)
                ):
                    errors.append(f"{at}: T2 requires explicit domain failure classes")
                elif any(
                    (item.id if isinstance(item, ast.Name) else item.attr if isinstance(item, ast.Attribute) else "")
                    in ("Exception", "BaseException", "ValueError", "ContractViolation", "InputDataError")
                    for item in failures.elts
                ):
                    errors.append(
                        f"{at}: T2 requires specific recoverable domain failures, separate from contract/data faults"
                    )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--fingerprint", help="Print a specific test AST fingerprint; never updates source markers.")
    arguments = parser.parse_args()
    if arguments.fingerprint is not None:
        print(test_fingerprint(arguments.root, arguments.fingerprint))
        return 0
    errors = check_policy(arguments.root)
    for error in errors:
        print(error)
    if not errors:
        print("Sultai trust policy passed")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
