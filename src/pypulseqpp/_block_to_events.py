"""A block as the events it plays."""

from __future__ import annotations

__all__ = ["block_to_events"]

from typing import Any


def block_to_events(*args: Any) -> tuple:
    """Split a block into the events it plays, or pass events through.

    A block is a namespace with an ``rf`` field, and its events are its
    fields in the order they are stored -- a field holding several, as a
    label chain does, contributing each of them. Anything else is already a
    list of events and comes back as a tuple of the same events, so a caller
    may hand over either without knowing which it has.

    Parameters
    ----------
    *args : SimpleNamespace or list
        One block, or the events themselves.

    Returns
    -------
    tuple
        The events.

    Raises
    ------
    ValueError
        When more than one block is passed.

    Examples
    --------
    >>> import pypulseqpp as pp
    >>> gx = pp.make_trapezoid("x", area=1000, duration=1e-3)
    >>> adc = pp.make_adc(num_samples=128, duration=1e-3)
    >>> len(pp.block_to_events([gx, adc]))
    2
    """
    items = tuple(args)

    # A block handed over as a one-element list of a one-element list is the
    # same block; MATLAB's cell arrays nest that way.
    while (
        len(items) == 1
        and isinstance(items[0], (list, tuple))
        and len(items[0]) == 1
        and isinstance(items[0][0], (list, tuple))
    ):
        items = tuple(items[0])

    if not items:
        return ()

    first = items[0]
    if hasattr(first, "rf"):
        if len(items) != 1:
            raise ValueError("block_to_events(): only one block at a time")
        events: list[Any] = []
        for value in vars(first).values():
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                events.extend(value)
            else:
                events.append(value)
        return tuple(events)

    if isinstance(first, (list, tuple)):
        return tuple(first)

    return items
