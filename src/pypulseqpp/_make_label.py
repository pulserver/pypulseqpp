"""Pulseq label events, including custom names registered on first use."""

from __future__ import annotations

from . import _ext as _cxx

__all__ = ["make_label"]

#: What a label event does with its value.
_KINDS = {"SET": "labelset", "INC": "labelinc"}


def make_label(label: str, type: str, value):  # noqa: A002
    """Construct a label event.

    Parameters
    ----------
    label : str
        Label name. Pulseq's builtin names keep their reference numbering;
        any other name is registered as a custom label on first use.
    type : {'SET', 'INC'}
        ``'SET'`` assigns ``value`` to the label, ``'INC'`` adds it to the
        label's running value.
    value : int or bool or float
        Value to assign or add, converted to an integer.

    Returns
    -------
    LabelEvent
        The compiled label event.

    Raises
    ------
    ValueError
        If ``type`` is neither ``'SET'`` nor ``'INC'``, or ``label`` is empty.
        The name is not checked against Pulseq's builtin labels.

    Examples
    --------
    >>> import pypulseqpp as pp
    >>> event = pp.make_label("LIN", "SET", 12)
    >>> event.type, event.label, event.value
    ('labelset', 'LIN', 12)

    A name Pulseq does not define is a custom label, carried by name:

    >>> pp.make_label("SPARKLE", "INC", 1).label
    'SPARKLE'
    """
    kind = _KINDS.get(str(type).upper())
    if kind is None:
        raise ValueError(f"make_label(): type has to be 'SET' or 'INC', not {type!r}")
    if not isinstance(label, str) or not label:
        raise ValueError("make_label(): a label needs a name")

    from types import SimpleNamespace

    return _cxx._label_from(SimpleNamespace(type=kind, label=label, value=int(value)))
