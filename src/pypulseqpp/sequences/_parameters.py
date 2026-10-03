"""Protocol parameters of a sequence function, read from its signature and docstring."""

from __future__ import annotations

import ast
import inspect
import re
import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pypulseqpp._prescription import accepts_none, documented, scalar

__all__ = ["ProtocolParameter", "parameters"]


@dataclass(frozen=True)
class ProtocolParameter:
    """One protocol parameter, as a sequence function declares and documents it."""

    #: The keyword the sequence function takes.
    name: str
    #: The scalar the annotation names, bool, int, float or str; None for any
    #: other annotation.
    type: type | None
    #: The default; `inspect.Parameter.empty` for a parameter without one.
    default: Any
    #: Whether the annotation admits None, which leaves the value to the design.
    optional: bool
    #: The first parenthesised group of the description's first sentence, the
    #: form the shipped sequences state units in, as in ``Echo time (s).``;
    #: empty where there is none.
    unit: str
    #: The values the documented type lists in braces, as in
    #: ``{'slab', 'nonselective'}``, in the order listed; empty otherwise.
    choices: tuple[Any, ...]
    #: The description in the Parameters section, on one line.
    description: str


def _unit(description: str) -> str:
    first = re.split(r"(?<=\.)\s", description, maxsplit=1)[0]
    group = re.search(r"\(([^()]+)\)", first)
    return group.group(1) if group else ""


def _choices(kind: str) -> tuple[Any, ...]:
    listed = re.match(r"\{(.*?)\}", kind)
    try:
        return tuple(ast.literal_eval(f"[{listed.group(1)}]")) if listed else ()
    except (ValueError, SyntaxError):  # braces around names, not values
        return ()


def _prescribed(function: Callable[..., Any]) -> dict[str, inspect.Parameter]:
    """Return the parameters a protocol names: those after the first, which is ``system``."""
    return {
        name: p
        for name, p in list(inspect.signature(function).parameters.items())[1:]
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    }


def parameters(function: Callable[..., Any]) -> dict[str, ProtocolParameter]:
    """Return each protocol parameter of a sequence function with its type, default, unit and description.

    The protocol is every parameter after the first, which is ``system``. Its
    type, default and whether it admits None are read from the annotation and
    the default; its unit, choices and description from the NumPy
    ``Parameters`` section of the docstring.

    Parameters
    ----------
    function : callable
        A sequence function ``function(system, **protocol)``.

    Returns
    -------
    dict[str, ProtocolParameter]
        One entry per protocol parameter, in signature order.

    Examples
    --------
    >>> from pypulseqpp import sequences
    >>> def gre(system, *, te: float | None = None, n: int = 64):
    ...     '''Gradient echo.
    ...
    ...     Parameters
    ...     ----------
    ...     te : float | None, default=None
    ...         Echo time (s). ``None`` is as short as the readout admits.
    ...     n : int, default=64
    ...         Matrix size.
    ...     '''
    >>> protocol = sequences.parameters(gre)
    >>> list(protocol), protocol["te"].unit, protocol["te"].optional
    (['te', 'n'], 's', True)
    """
    hints = typing.get_type_hints(function)
    documentation = documented(inspect.getdoc(function))
    entries = {}
    for name, parameter in _prescribed(function).items():
        annotation = hints.get(name, parameter.annotation)
        kind, description = documentation.get(name, ("", ""))
        entries[name] = ProtocolParameter(
            name=name,
            type=scalar(annotation),
            default=parameter.default,
            optional=accepts_none(annotation),
            unit=_unit(description),
            choices=_choices(kind),
            description=description,
        )
    return entries
