"""Writer of the label events a block changes."""

from __future__ import annotations

import pypulseqpp as pp
from pypulseqpp import _ext

__all__ = ["Labels"]


class Labels(_ext.LabelWriter):
    """Writer of label events for the sticky label state of a Pulseq sequence.

    A label keeps its value across blocks until an event changes it, so a block
    carries the events of the labels it changes and no others. Calling an
    instance with each label's new value, by its Pulseq name, returns the
    events to add to the block, empty when nothing changed. An unchanged value
    writes nothing, a change equal to the label's previous change is an INC,
    which a scan repeats as one event, and any other change is a SET. A change
    of ``ONCE`` calls :meth:`restart` first, so every label passed with it is
    written again as a SET. An instance returns the same event object for
    equal statements, so the events are added to blocks as they are, not
    modified.

    Examples
    --------
    An unchanged value writes nothing, and a change equal to the label's
    previous change is an INC:

    >>> from pypulseqpp import sequences
    >>> labels = sequences.Labels()
    >>> [[(e.type, e.value) for e in labels(LIN=line)] for line in (0, 2, 4, 4, 3)]
    [[('labelset', 0)], [('labelset', 2)], [('labelinc', 2)], [], [('labelset', 3)]]

    Labels passed together are written in the order given:

    >>> labels = sequences.Labels()
    >>> [(e.label, e.type, e.value) for e in labels(LIN=0, SLC=1)]
    [('LIN', 'labelset', 0), ('SLC', 'labelset', 1)]
    >>> [(e.label, e.type, e.value) for e in labels(LIN=0, SLC=2, ONCE=1)]
    [('LIN', 'labelset', 0), ('SLC', 'labelset', 2), ('ONCE', 'labelset', 1)]
    """

    def __init__(self) -> None:
        super().__init__(pp.make_label)

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
        self._restart()
