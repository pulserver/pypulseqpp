"""Scalar annotations and NumPy ``Parameters`` sections, read for the command line and protocol editors."""

from __future__ import annotations

import inspect
import itertools
import types
import typing

#: The annotations a command-line flag or a protocol entry can be made from.
SCALARS = (bool, int, float, str)


def _members(annotation: typing.Any) -> tuple:
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        return typing.get_args(annotation)
    return (annotation,)


def scalar(annotation: typing.Any) -> type | None:
    """Return the first supported scalar type in an annotation or union, if any."""
    return next((member for member in _members(annotation) if member in SCALARS), None)


def accepts_none(annotation: typing.Any) -> bool:
    """Check whether an annotation admits None, alone or in a union."""
    return any(member in (None, type(None)) for member in _members(annotation))


def _is_heading(lines: list[str], i: int) -> bool:
    """Check whether line ``i`` is a NumPy section heading: a name underlined with dashes."""
    return (
        i + 1 < len(lines)
        and bool(lines[i].strip())
        and not lines[i][0].isspace()
        and set(lines[i + 1].strip()) == {"-"}
    )


def _section(doc: str | None, name: str) -> list[str]:
    """Return the lines of a NumPy section's body, up to the next heading."""
    lines = inspect.cleandoc(doc or "").splitlines()
    starts = [i for i in range(len(lines)) if _is_heading(lines, i)]
    for at, following in itertools.pairwise([*starts, len(lines)]):
        if lines[at].strip() == name:
            return lines[at + 2 : following]
    return []


def documented(doc: str | None) -> dict[str, tuple[str, str]]:
    """Return the type and the description of each parameter a docstring documents.

    A NumPy ``Parameters`` block states each name, then its type after a colon,
    then its description indented under it. Several names sharing a description
    are comma separated, and a long list of them wraps with a trailing
    backslash. The description is returned on one line; a name described twice
    keeps its first description, and a name with no description is omitted.
    """
    described: dict[str, tuple[str, list[str]]] = {}
    current: list[str] | None = None
    pending: list[str] = []
    for line in _section(doc, "Parameters"):
        if not line.strip():
            continue
        if line[0].isspace():
            if current is not None:
                current.append(line.strip())
            continue
        listed, _, kind = line.removesuffix("\\").partition(":")
        names = pending + [n.strip() for n in listed.split(",") if n.strip()]
        if line.endswith("\\"):
            pending = names
            continue
        pending = []
        current = None if names[0] in described else []
        for name in names if current is not None else ():
            described.setdefault(name, (kind.strip(), current))
    return {
        name: (kind, " ".join(text)) for name, (kind, text) in described.items() if text
    }
