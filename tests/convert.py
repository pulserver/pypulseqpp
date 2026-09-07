"""Loading an upstream sequence into the compiled core.

The parity tests need one sequence held two ways: as the upstream toolbox
built it, and as pypulseqpp's core holds it. This module is the bridge, and it
exists only for the tests -- the package itself never imports upstream.

Nothing here interprets a row. Upstream's libraries and the core's are the
same libraries, in the same order, with the same columns, because both are the
file format's; so the rows cross as they stand and their ids cross with them.

There are two ways across, and the tests use both. :func:`to_core` registers
one row at a time and checks each landed on the id it came from, which is what
catches a renumbering. :func:`to_core_bulk` hands each library over as one
array, which is the path a real protocol takes: registering millions of rows
one at a time costs more than everything else put together. The two must
produce the same file.
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


def _gradient_tables(seq):
    """Upstream's one gradient library as the core's two, plus the slot map.

    A trapezoid and an arbitrary gradient share one numbering in the file, so
    the core keeps two fixed-width tables and a map from the shared id onto
    whichever row is real: positive indexes the trapezoids, negative the
    arbitrary waveforms, both 1-based.
    """
    traps: list[np.ndarray] = []
    arbitrary: list[np.ndarray] = []
    slots: list[int] = []
    for identifier in sorted(seq.grad_library.data):
        row = np.asarray(seq.grad_library.data[identifier], dtype=float)
        if seq.grad_library.type[identifier] == "t":
            traps.append(row)
            slots.append(len(traps))
        else:
            arbitrary.append(row)
            slots.append(-len(arbitrary))
    return (
        np.asarray(traps, dtype=float).reshape(len(traps), 5),
        np.asarray(arbitrary, dtype=float).reshape(len(arbitrary), 6),
        np.asarray(slots, dtype=np.int32),
    )


def _shape_arrays(seq):
    """The shape library as sample counts, offsets and one flat sample array."""
    counts: list[int] = []
    samples: list[np.ndarray] = []
    starts = [0]
    for identifier in sorted(seq.shape_library.data):
        row = np.asarray(seq.shape_library.data[identifier], dtype=float)
        counts.append(int(row[0]))
        samples.append(row[1:])
        starts.append(starts[-1] + len(row) - 1)
    flat = np.concatenate(samples) if samples else np.zeros(0)
    return np.asarray(counts, dtype=np.int32), np.asarray(starts, dtype=np.int32), flat


def _int_rows(library, width):
    """A whole-number library as one (N, width) int32 array, in id order."""
    rows = [
        np.asarray(library.data[key], dtype=np.int32) for key in sorted(library.data)
    ]
    return np.asarray(rows, dtype=np.int32).reshape(len(rows), width)


def _float_rows(library, width):
    """A real-valued library as one (N, width) float array, in id order."""
    rows = [
        np.asarray(library.data[key], dtype=float)[:width]
        for key in sorted(library.data)
    ]
    return np.asarray(rows, dtype=float).reshape(len(rows), width)


def to_core_bulk(seq) -> _ext.Sequence:
    """The same sequence, loaded one library at a time rather than one row.

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

    core.set_shapes(*_shape_arrays(seq))

    uses = "".join(
        seq.rf_library.type.get(key, "u") for key in sorted(seq.rf_library.data)
    )
    core.set_rf(_float_rows(seq.rf_library, 10), uses)
    core.set_gradients(*_gradient_tables(seq))
    core.set_adc(_float_rows(seq.adc_library, _ADC_COLUMNS))
    core.set_triggers(_float_rows(seq.trigger_library, 4))
    core.set_label_set(_int_rows(seq.label_set_library, 2))
    core.set_label_inc(_int_rows(seq.label_inc_library, 2))

    delays = [
        seq.soft_delay_library.data[key] for key in sorted(seq.soft_delay_library.data)
    ]
    core.set_soft_delays(
        [int(row[0]) for row in delays],
        [float(row[1]) for row in delays],
        [float(row[2]) for row in delays],
        [str(row[3]) for row in delays],
    )

    for identifier, name in zip(
        seq.extension_numeric_idx, seq.extension_string_idx, strict=True
    ):
        core.set_extension_type_id(name, identifier)
    core.set_extensions(_int_rows(seq.extensions_library, 3))

    order = sorted(seq.block_events)
    events = np.asarray([seq.block_events[key] for key in order], dtype=np.int32)
    durations = np.asarray([seq.block_durations[key] for key in order], dtype=float)
    core.set_blocks(events.reshape(len(order), -1), durations)
    return core
