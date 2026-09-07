"""Loading an upstream sequence into the compiled core.

The parity tests need one sequence held two ways: as the upstream toolbox
built it, and as pypulseqpp's core holds it. This module is the bridge, and it
exists only for the tests -- the package itself never imports upstream.

Nothing here interprets a row. Upstream's libraries and the core's are the
same libraries, in the same order, with the same columns, because both are the
file format's; so the rows cross as they stand and their ids cross with them.
Every registration is checked to have landed on the id it came from, since a
renumbering would show up as a confusing diff much later.
"""

from __future__ import annotations

import numpy as np

from pypulseqpp import _ext

#: Columns of an ADC row that the file format carries. Upstream appends the
#: system's dead time as a ninth, which comes from the system rather than the
#: sequence and is never written back.
_ADC_COLUMNS = 8


def _rows(library) -> list[tuple[int, np.ndarray]]:
    """A library's rows as (id, values), in id order."""
    return [
        (key, np.asarray(library.data[key], dtype=float))
        for key in sorted(library.data)
    ]


def _check(registered: int, expected: int, what: str) -> None:
    if registered != expected:
        raise AssertionError(
            f"{what} registered as id {registered} but upstream numbered it {expected}"
        )


def to_core(seq) -> _ext.Sequence:
    """Return the compiled-core sequence holding what ``seq`` holds.

    Parameters
    ----------
    seq
        An upstream :class:`pypulseq.Sequence`.

    Returns
    -------
    _ext.Sequence
        The same libraries, block table and definitions.
    """
    core = _ext.Sequence()
    core.set_rasters(
        seq.rf_raster_time,
        seq.grad_raster_time,
        seq.adc_raster_time,
        seq.block_duration_raster,
    )

    for key, value in seq.definitions.items():
        core.set_definition(key, value)

    for identifier, row in _rows(seq.shape_library):
        _check(core.register_shape(int(row[0]), row[1:]), identifier, "shape")

    for identifier, row in _rows(seq.rf_library):
        use = seq.rf_library.type.get(identifier, "u")
        _check(core.register_rf(row, use), identifier, "RF event")

    # Trapezoids and arbitrary gradients share one numbering in the file, so
    # they are registered in that shared order rather than table by table.
    for identifier, row in _rows(seq.grad_library):
        if seq.grad_library.type[identifier] == "t":
            _check(core.register_trap(row), identifier, "trapezoid")
        else:
            _check(core.register_arbitrary(row), identifier, "arbitrary gradient")

    for identifier, row in _rows(seq.adc_library):
        _check(core.register_adc(row[:_ADC_COLUMNS]), identifier, "ADC event")

    for identifier, row in _rows(seq.trigger_library):
        _check(core.register_trigger(row), identifier, "trigger")

    for identifier, row in _rows(seq.label_set_library):
        _check(
            core.register_label_set(int(row[0]), int(row[1])), identifier, "LABELSET"
        )

    for identifier, row in _rows(seq.label_inc_library):
        _check(
            core.register_label_inc(int(row[0]), int(row[1])), identifier, "LABELINC"
        )

    for identifier in sorted(seq.soft_delay_library.data):
        number, offset, factor, hint = seq.soft_delay_library.data[identifier]
        delay = _ext.SoftDelay(
            num=int(number), offset=float(offset), factor=float(factor), hint=str(hint)
        )
        _check(core.register_soft_delay(delay), identifier, "soft delay")

    # The extension type ids are pinned before the chain is built, so a name
    # keeps the number upstream gave it and the written file agrees.
    for identifier, name in zip(
        seq.extension_numeric_idx, seq.extension_string_idx, strict=True
    ):
        core.set_extension_type_id(name, identifier)
    for identifier, row in _rows(seq.extensions_library):
        _check(
            core.chain_extension(int(row[0]), int(row[1]), int(row[2])),
            identifier,
            "extension",
        )

    add_block = core.add_block
    for identifier in sorted(seq.block_events):
        events = seq.block_events[identifier]
        add_block(
            int(events[1]),
            int(events[2]),
            int(events[3]),
            int(events[4]),
            int(events[5]),
            int(events[6]),
            float(seq.block_durations[identifier]),
        )

    return core
