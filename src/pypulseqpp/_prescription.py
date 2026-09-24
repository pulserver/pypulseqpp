"""Scalar annotations and NumPy ``Parameters`` sections, read for the command line and protocol editors."""

from __future__ import annotations

import inspect
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


def documented(doc: str | None) -> dict[str, tuple[str, str]]:
    """Return the type and the description of each parameter a docstring documents.

    A NumPy ``Parameters`` block states each name, then its type after a colon,
    then its description indented under it. Several names sharing a description
    are comma separated, and a long list of them wraps with a trailing
    backslash. The description is returned on one line; a name described twice
    keeps its first description, and a name with no description is omitted.
    """
    lines = inspect.cleandoc(doc or "").splitlines()
    try:
        start = next(
            i + 2
            for i, line in enumerate(lines[:-1])
            if line.strip() == "Parameters" and set(lines[i + 1].strip()) == {"-"}
        )
    except StopIteration:
        return {}

    described: dict[str, tuple[str, list[str]]] = {}
    current: list[str] | None = None
    pending: list[str] = []
    heading = ""
    for line in lines[start:]:
        if not line.strip():
            continue
        if line[0].isspace():
            if current is not None:
                current.append(line.strip())
            continue
        if heading and set(line.strip()) == {"-"}:
            break  # the next section's underline
        heading = line
        header = line[:-1] if line.endswith("\\") else line
        listed, _, kind = header.partition(":")
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
