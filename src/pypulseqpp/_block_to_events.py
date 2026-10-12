"""Decomposition of a block into its constituent events."""

from __future__ import annotations

__all__ = ["block_to_events"]

from typing import Any


def block_to_events(*args: Any) -> tuple:
    """Return the events of a block, or of a list of events, as a tuple.

    A block is a namespace with an ``rf`` field; its events are its non-None
    fields in storage order, a field holding a list, such as a label chain,
    contributing each element. A list of events is returned as a tuple of the
    same events.

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
