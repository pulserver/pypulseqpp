"""Consistency checks for defaults in NumPy-style callable docstrings."""

import ast
import importlib
import re
from pathlib import Path

import numpy
import pytest

ROOT = Path(__file__).parents[1]
SOURCES = [
    *ROOT.joinpath("src/pypulseqpp").rglob("*.py"),
    *ROOT.joinpath("examples/sequence").glob("*.py"),
]
HEADER = re.compile(
    r"^(\*{0,2}[A-Za-z_]\w*(?:\s*,\s*\*{0,2}[A-Za-z_]\w*)*)\s*:\s*(.+)$"
)
DEFAULT = re.compile(r"(?:^|,\s*)default\s*=\s*(.+)$")


def _callables(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        target = node
        if isinstance(node, ast.ClassDef):
            target = next(
                (
                    item
                    for item in node.body
                    if isinstance(item, ast.FunctionDef)
                    and item.name in {"__init__", "init_module"}
                ),
                None,
            )
        if not isinstance(target, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        doc = ast.get_docstring(node, clean=True)
        if doc and (
            "Parameters\n----------" in doc
            or "Other Parameters\n----------------" in doc
        ):
            yield node, target, doc


def _documented_parameters(doc):
    """Parameter headers of the Parameters and Other Parameters sections.

    ``ast.get_docstring(clean=True)`` dedents a header to column 0 and leaves
    its description indented, so a section ends at the next section title, a
    line followed by its dashes underline.
    """
    lines = doc.splitlines()
    out = {}
    inside = False
    for i, line in enumerate(lines):
        underlined = i + 1 < len(lines) and re.fullmatch(r"-{3,}", lines[i + 1])
        if underlined:
            inside = line in {"Parameters", "Other Parameters"}
            continue
        if not inside or not line or line.startswith(" ") or re.fullmatch(r"-+", line):
            continue
        match = HEADER.match(line)
        if match:
            for name in map(str.strip, match.group(1).split(",")):
                out[name.lstrip("*")] = match.group(2)
    return out


def _module_namespace(path):
    """The namespace of the module at ``path``, where a default's names resolve."""
    if path.is_relative_to(ROOT / "examples/sequence"):
        name = f"pypulseqpp.sequences.sequence.{path.stem}"
    else:
        parts = path.relative_to(ROOT / "src").with_suffix("").parts
        name = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
    return {"numpy": numpy, "np": numpy, **vars(importlib.import_module(name))}


def _value(expression, namespace):
    """A literal, or a dotted name resolved in the module's namespace."""
    try:
        return ast.literal_eval(expression)
    except (ValueError, SyntaxError):
        head, *rest = expression.split(".")
        value = namespace[head]
        for attribute in rest:
            value = getattr(value, attribute)
        return value


def _same_value(stated, expected, path):
    """Whether two default expressions are equal, a named constant to its value."""
    if stated == expected:
        return True
    namespace = _module_namespace(path)
    try:
        return bool(_value(stated, namespace) == _value(expected, namespace))
    except (KeyError, AttributeError):
        return False


def _signature_defaults(node):
    positional = node.args.posonlyargs + node.args.args
    args = positional + node.args.kwonlyargs
    defaults = (
        [None] * (len(positional) - len(node.args.defaults))
        + list(node.args.defaults)
        + list(node.args.kw_defaults)
    )
    return {
        arg.arg: None if default is None else ast.unparse(default)
        for arg, default in zip(args, defaults, strict=True)
    }


CASES = [
    (path, owner, function, doc)
    for path in SOURCES
    for owner, function, doc in _callables(path)
]


@pytest.mark.parametrize(
    "path,owner,function,doc",
    CASES,
    ids=lambda value: str(value) if isinstance(value, Path) else None,
)
def test_documented_defaults_match_python_signatures(path, owner, function, doc):
    """Documented callable parameters state exactly the Python signature default."""
    documented = _documented_parameters(doc)
    defaults = _signature_defaults(function)
    errors = []
    for name, type_spec in documented.items():
        if name not in defaults or name in {"self", "cls"}:
            continue
        match = DEFAULT.search(type_spec)
        stated = match.group(1).strip() if match else None
        expected = defaults[name]
        if expected is None and stated is not None:
            errors.append(f"{name} is required but documents default={stated}")
        elif expected is not None and (
            stated is None or not _same_value(stated, expected, path)
        ):
            errors.append(
                f"{name} documents {stated!r}; signature default is {expected!r}"
            )
        if "optional" in type_spec:
            errors.append(f"{name} uses 'optional' instead of an explicit default")
    assert not errors, f"{path}:{owner.lineno}: " + "; ".join(errors)
