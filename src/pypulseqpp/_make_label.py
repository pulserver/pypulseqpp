"""Label events, for a vocabulary that is not fixed in advance.

Upstream's `make_label` checks the name against the list Pulseq defines and
refuses anything else, so a sequence carrying its own bookkeeping -- a bin
index, a preparation state -- has to abuse a counter that means something
else. This one accepts any name.

That costs nothing, because a label is named rather than numbered: the file
writes the name, the reader reads it back, and a name the builtin table does
not carry is listed in `[DEFINITIONS]` so its number means something too.
Recognising it is the interpreter's business, and one it does not recognise
it ignores.
"""

from __future__ import annotations

from . import _ext as _cxx

__all__ = ["make_label"]

#: What a label event does with its value.
_KINDS = {"SET": "labelset", "INC": "labelinc"}


def make_label(label: str, type: str, value):  # noqa: A002
    """Create a label event.

    Parameters
    ----------
    label : str
        The label's name, in any spelling. Pulseq's own names keep their
        numbering; anything else is minted on first use.
    type : {'SET', 'INC'}
        ``'SET'`` assigns ``value``, ``'INC'`` adds it to the running counter.
    value : int or bool or float
        The counter or flag value, taken as an integer.

    Returns
    -------
    LabelEvent
        The event, with its fields in slots.

    Raises
    ------
    ValueError
        If ``type`` is neither ``'SET'`` nor ``'INC'``. The name is not
        checked, which is the point.

    Examples
    --------
    >>> import pypulseqpp as pp
    >>> event = pp.make_label("LIN", "SET", 12)
    >>> event.type, event.label, event.value
    ('labelset', 'LIN', 12)

    A name Pulseq does not define is accepted and travels by name:

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
