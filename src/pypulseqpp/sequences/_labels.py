"""Writer of the label events a block changes."""

from __future__ import annotations

import pypulseqpp as pp

__all__ = ["Labels"]


class Labels:
    """Writer of label events for the sticky label state of a Pulseq sequence.

    A label keeps its value across blocks until an event changes it, so a block
    carries the events of the labels it changes and no others. Calling an
    instance with each label's new value, by its Pulseq name, returns the
    events to add to the block, empty when nothing changed. An unchanged value
    writes nothing, a change equal to the label's previous change is an INC,
    which a scan repeats as one event, and any other change is a SET. A change
    of ``ONCE`` calls :meth:`restart` first, so every label passed with it is
    written again as a SET.

    Examples
    --------
    An unchanged value writes nothing, and a change equal to the label's
    previous change is an INC:

    >>> from pypulseqpp import sequences
    >>> labels = sequences.Labels()
    >>> [[(e.type, e.value) for e in labels(LIN=line)] for line in (0, 2, 4, 4, 3)]
    [[('labelset', 0)], [('labelset', 2)], [('labelinc', 2)], [], [('labelset', 3)]]
    """

    def __init__(self) -> None:
        self._state: dict[str, int] = {}
        self._steps: dict[str, int | None] = {}

    def __call__(self, **values: int) -> list:
        """Return the label events that set each label to its new value.

        Parameters
        ----------
        **values : int
            The new value of each label, by its Pulseq name.

        Returns
        -------
        list
            The label events to add to the block, empty when nothing changed.

        Examples
        --------
        >>> from pypulseqpp import sequences
        >>> labels = sequences.Labels()
        >>> [(e.label, e.type, e.value) for e in labels(LIN=0, SLC=1)]
        [('LIN', 'labelset', 0), ('SLC', 'labelset', 1)]
        >>> [(e.label, e.type, e.value) for e in labels(LIN=0, SLC=2, ONCE=1)]
        [('LIN', 'labelset', 0), ('SLC', 'labelset', 2), ('ONCE', 'labelset', 1)]
        """
        once = values.get("ONCE")
        if once is not None and int(once) != self._state.get("ONCE"):
            self.restart()
        events = []
        for name, value in values.items():
            value = int(value)
            last = self._state.get(name)
            if last == value:
                continue
            step = None if last is None else value - last
            if step is not None and step == self._steps.get(name):
                events.append(pp.make_label(name, "INC", step))
            else:
                events.append(pp.make_label(name, "SET", value))
            self._state[name] = value
            self._steps[name] = step
        return events

    def restart(self) -> None:
        """Write every label's next value as a SET, regardless of what was written before.

        An interpreter repeating the scan plays ``ONCE`` blocks only once, so a
        value set inside them, or counted from one set there, is not there on
        the repeats. Calling the instance restarts at every ``ONCE`` it
        writes; call this where a module writes ``ONCE`` itself.

        Examples
        --------
        >>> from pypulseqpp import sequences
        >>> labels = sequences.Labels()
        >>> [e.type for e in labels(LIN=3)], [e.type for e in labels(LIN=3)]
        (['labelset'], [])
        >>> labels.restart()
        >>> [e.type for e in labels(LIN=3)]
        ['labelset']
        """
        self._state.clear()
        self._steps.clear()
