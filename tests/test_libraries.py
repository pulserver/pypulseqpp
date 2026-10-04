"""A sequence's libraries, exported: the rows its file holds, under the ids its blocks name."""

from pathlib import Path

import numpy as np
import pytest
from pypulseq.decompress_shape import decompress_shape

import pypulseqpp as pp

SEQ = Path(__file__).parent / "seq"

#: The corpus files of revision 1.5, which a reader holds as they are written.
#: Older ones are converted as they are read, so their tables are not the file's.
CORPUS = sorted(
    path.name
    for path in SEQ.glob("*.seq")
    if "\nminor 5\n" in path.read_text(errors="replace")
)

#: The file's initial for each use an RF event can have.
USES = {
    "e": "excitation",
    "r": "refocusing",
    "i": "inversion",
    "s": "saturation",
    "p": "preparation",
    "o": "other",
    "u": "undefined",
}

#: A file writes six significant digits of a floating-point column.
RTOL = 1e-5

MICRO = 1e-6


def _sections(text):
    """Each section's rows, split into fields, and the extension types declared."""
    sections, types, current = {}, {}, None
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            current = line.strip("[]")
            sections[current] = []
        elif line.startswith("extension "):
            _, current, number = line.split()
            types[current] = int(number)
            sections[current] = []
        else:
            sections[current].append(line.split())
    return sections, types


def _numbers(rows, width=None):
    table = np.array([[float(field) for field in row] for row in rows], dtype=float)
    return table.reshape(-1, width) if width is not None else table


def _shapes(rows):
    shapes, current = {}, None
    for row in rows:
        if row[0] == "shape_id":
            current = int(row[1])
            shapes[current] = [0, []]
        elif row[0] == "num_samples":
            shapes[current][0] = int(row[1])
        else:
            shapes[current][1].append(float(row[0]))
    return shapes


def _ids(table, count):
    np.testing.assert_array_equal(table[:, 0], np.arange(1, count + 1))


def _assert_holds(tables, text, block_raster):
    """Assert that ``tables`` is, section by section, what ``text`` writes."""
    sections, types = _sections(text)
    assert tables.layout == 1
    assert tables.extension_types == types

    blocks = _numbers(sections.get("BLOCKS", []), 8)
    np.testing.assert_array_equal(tables.blocks, blocks[:, 2:])
    np.testing.assert_allclose(tables.block_durations, blocks[:, 1] * block_raster)

    rf_rows = sections.get("RF", [])
    rf = _numbers([row[:-1] for row in rf_rows], 11)
    _ids(rf, len(tables.rf))
    expected = rf[:, 1:].copy()
    expected[:, 4:6] *= MICRO
    np.testing.assert_allclose(tables.rf, expected, rtol=RTOL, atol=1e-9)
    assert tables.rf_use == tuple(USES[row[-1]] for row in rf_rows)

    arbitrary = _numbers(sections.get("GRADIENTS", []), 7)
    np.testing.assert_array_equal(tables.arbitrary_gradient_ids, arbitrary[:, 0])
    expected = arbitrary[:, 1:].copy()
    expected[:, 5] *= MICRO
    np.testing.assert_allclose(tables.arbitrary_gradients, expected, rtol=RTOL)

    trapezoids = _numbers(sections.get("TRAP", []), 6)
    np.testing.assert_array_equal(tables.trapezoid_ids, trapezoids[:, 0])
    np.testing.assert_allclose(
        tables.trapezoids,
        trapezoids[:, 1:] * [1, MICRO, MICRO, MICRO, MICRO],
        rtol=RTOL,
    )

    adc = _numbers(sections.get("ADC", []), 9)
    _ids(adc, len(tables.adc))
    expected = adc[:, 1:] * [1, 1e-9, MICRO, 1, 1, 1, 1, 1]
    np.testing.assert_allclose(tables.adc, expected, rtol=RTOL, atol=1e-12)

    chains = _numbers(sections.get("EXTENSIONS", []), 4)
    _ids(chains, len(tables.extensions))
    np.testing.assert_array_equal(tables.extensions, chains[:, 1:])

    triggers = _numbers(sections.get("TRIGGERS", []), 5)
    _ids(triggers, len(tables.triggers))
    np.testing.assert_allclose(
        tables.triggers, triggers[:, 1:] * [1, 1, MICRO, MICRO], rtol=RTOL
    )

    rotations = _numbers(sections.get("ROTATIONS", []), 5)
    _ids(rotations, len(tables.rotations))
    np.testing.assert_allclose(tables.rotations, rotations[:, 1:], rtol=RTOL, atol=1e-6)

    for name, values, labels in (
        ("LABELSET", tables.label_set_values, tables.label_set_labels),
        ("LABELINC", tables.label_inc_values, tables.label_inc_labels),
    ):
        rows = sections.get(name, [])
        assert [int(row[0]) for row in rows] == list(range(1, len(values) + 1))
        np.testing.assert_array_equal(values, [int(row[1]) for row in rows])
        assert labels == tuple(row[2] for row in rows)

    shims = sections.get("RF_SHIMS", [])
    assert [int(row[0]) for row in shims] == list(range(1, len(tables.rf_shims) + 1))
    for row, shim in zip(shims, tables.rf_shims, strict=True):
        assert int(row[1]) == shim.size // 2
        np.testing.assert_allclose(shim, [float(value) for value in row[2:]], rtol=RTOL)

    delays = sections.get("DELAYS", [])
    assert [int(row[0]) for row in delays] == list(
        range(1, len(tables.soft_delays) + 1)
    )
    np.testing.assert_allclose(
        tables.soft_delays,
        _numbers([row[1:4] for row in delays], 3) * [1, MICRO, 1],
        rtol=RTOL,
    )
    assert tables.soft_delay_hints == tuple(row[4] for row in delays)

    shapes = _shapes(sections.get("SHAPES", []))
    assert sorted(shapes) == list(range(1, len(tables.shapes) + 1))
    for identifier, shape in enumerate(tables.shapes, start=1):
        count, values = shapes[identifier]
        assert shape.num_samples == count
        written = pp.io.Shape(count, np.array(values))
        np.testing.assert_allclose(
            shape.decompressed(), written.decompressed(), rtol=RTOL, atol=1e-9
        )


def _every_library():
    """A sequence with a row in every library a file can hold."""
    system = pp.Opts(
        max_grad=32, grad_unit="mT/m", max_slew=130, slew_unit="T/m/s", B0=3.0
    )
    seq = pp.Sequence(system)
    rf, gz, _ = pp.make_sinc_pulse(
        flip_angle=np.pi / 2,
        duration=2e-3,
        slice_thickness=5e-3,
        return_gz=True,
        system=system,
        use="excitation",
        freq_ppm=-3.3,
    )
    seq.add_block(
        rf,
        gz,
        pp.make_label("SLC", "SET", 2),
        pp.make_label("SPARKLE", "SET", 5),
        pp.make_rf_shim(np.array([1 + 0j, 0.5 + 0.5j])),
    )
    ramp = pp.make_extended_trapezoid(
        "y", times=[0, 1e-4, 3e-4, 4e-4], amplitudes=[0, 2e5, 2e5, 0], system=system
    )
    seq.add_block(
        pp.make_arbitrary_grad("x", np.linspace(0, 1e5, 20), system=system),
        ramp,
        pp.make_rotation(np.pi / 4),
    )
    seq.add_block(
        pp.make_adc(
            num_samples=64,
            duration=3.2e-3,
            system=system,
            phase_modulation=np.linspace(0, 1, 64),
        ),
        pp.make_label("LIN", "INC", 1),
        pp.make_trigger("physio1", duration=1e-3),
    )
    seq.add_block(pp.make_digital_output_pulse("osc0", duration=1e-4))
    seq.add_block(pp.make_soft_delay("TE", default_duration=5e-3))
    return seq


def test_a_designed_sequence_exports_every_row_its_file_writes(tmp_path):
    seq = _every_library()
    path = tmp_path / "every.seq"
    seq.write(path, remove_duplicates=False)
    tables = seq.libraries()

    _assert_holds(tables, path.read_text(), seq.block_duration_raster)
    for name in (
        "rf",
        "trapezoids",
        "arbitrary_gradients",
        "adc",
        "shapes",
        "extensions",
        "triggers",
        "rotations",
        "label_set_values",
        "label_inc_values",
        "rf_shims",
        "soft_delays",
    ):
        assert len(getattr(tables, name)), f"no {name} to compare"
    assert "SPARKLE" in tables.label_set_labels


@pytest.mark.parametrize("name", CORPUS)
def test_a_file_read_exports_its_rows_under_its_own_ids(name):
    seq = pp.Sequence()
    seq.read(SEQ / name, remove_duplicates=False)

    _assert_holds(seq.libraries(), (SEQ / name).read_text(), seq.block_duration_raster)


def test_every_gradient_id_is_one_trapezoid_or_one_arbitrary_gradient():
    tables = _every_library().libraries()
    ids = np.concatenate([tables.trapezoid_ids, tables.arbitrary_gradient_ids])
    played = tables.blocks[:, 1:4]

    assert sorted(ids) == list(range(1, ids.size + 1))
    assert set(played[played > 0]) <= set(ids)


def test_taking_the_tables_changes_nothing_and_no_later_edit_reaches_them():
    seq = _every_library()
    edits = seq._native.edits()
    tables = seq.libraries()
    held = tables.blocks.copy(), tables.rf.copy(), len(tables.shapes)

    assert seq._native.edits() == edits
    seq.add_block(pp.make_block_pulse(np.pi, duration=1e-3, use="refocusing"))
    seq.mod_grad_axis("x", -1)

    np.testing.assert_array_equal(tables.blocks, held[0])
    np.testing.assert_array_equal(tables.rf, held[1])
    assert len(tables.shapes) == held[2]
    for array in (
        tables.blocks,
        tables.block_durations,
        tables.rf,
        tables.shapes[0].data,
    ):
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 0


def test_an_unencoded_shape_is_exported_as_its_samples_and_decodes_as_upstream_does(
    tmp_path,
):
    seq = _every_library()
    before = seq.libraries().shapes
    seq.write(tmp_path / "every.seq", remove_duplicates=False)
    after = seq.libraries().shapes

    assert any(shape.data.size == shape.num_samples for shape in before)
    assert any(shape.data.size < shape.num_samples for shape in after)
    for raw, encoded in zip(before, after, strict=True):
        np.testing.assert_allclose(
            raw.decompressed(), encoded.decompressed(), atol=1e-6
        )
        np.testing.assert_array_equal(decompress_shape(encoded), encoded.decompressed())


def test_a_gradient_waveform_shape_is_normalised_to_its_amplitude():
    seq = _every_library()
    tables = seq.libraries()
    row = tables.arbitrary_gradients[
        list(tables.arbitrary_gradient_ids).index(tables.blocks[1, 1])
    ]

    waveform = tables.shapes[int(row[3]) - 1].decompressed() * row[0]
    np.testing.assert_allclose(waveform, seq.get_block(2).gx.waveform, rtol=1e-12)


def _definition_of(ids, table):
    """The definition each id names, 0 where a block carries no such event."""
    found = np.zeros(len(ids), dtype=np.int64)
    present = ids > 0
    found[present] = table[ids[present] - 1]
    return found


def test_gradients_differing_only_in_amplitude_share_a_definition():
    seq = pp.Sequence(pp.Opts())
    for amplitude in (1e5, -1e5, 5e4):
        seq.add_block(
            pp.make_trapezoid("x", amplitude=amplitude, rise_time=1e-4, flat_time=1e-3)
        )
    seq.add_block(pp.make_trapezoid("x", amplitude=1e5, rise_time=1e-4, flat_time=2e-3))

    played = seq.event_definitions().gradient

    assert played.tolist() == [1, 1, 1, 2]


@pytest.mark.parametrize("name", CORPUS)
def test_a_blocks_definition_follows_from_its_events_definitions(name):
    """What pypulseqpp interned a block as is a function of what it publishes.

    A consumer composing the block table with the per-event definitions
    partitions the blocks exactly as pypulseqpp's own block definitions do:
    the correspondence runs both ways, so the composed key neither splits a
    definition nor merges two. That is what lets the consumer take the
    partition rather than intern the events a second time. Pure delays are
    left out, being one definition whatever they last.
    """
    seq = pp.Sequence()
    seq.read(SEQ / name, remove_duplicates=False)

    tables = seq.libraries()
    definitions = seq.event_definitions()
    blocks = tables.blocks
    composed = zip(
        _definition_of(blocks[:, 0], definitions.rf),
        _definition_of(blocks[:, 1], definitions.gradient),
        _definition_of(blocks[:, 2], definitions.gradient),
        _definition_of(blocks[:, 3], definitions.gradient),
        np.rint(tables.block_durations * 1e9).astype(np.int64),
        strict=True,
    )

    by_interned: dict[int, set] = {}
    by_key: dict[tuple, set] = {}
    for key, interned in zip(composed, seq._native.instance_definitions(), strict=True):
        if not any(key[:4]):
            continue
        by_interned.setdefault(int(interned), set()).add(key)
        by_key.setdefault(key, set()).add(int(interned))

    if not by_interned:
        pytest.skip("plays no RF or gradient, so there is no key to compose")
    assert all(len(keys) == 1 for keys in by_interned.values())
    assert all(len(interned) == 1 for interned in by_key.values())


def test_arbitrary_gradients_of_different_lengths_are_different_definitions():
    """A waveform belongs to the playout; how many samples it has does not.

    With no time shape an arbitrary gradient lasts as long as its waveform,
    so two of different lengths are different events however alike the rest
    of the row is.
    """
    seq = pp.Sequence(pp.Opts())
    for samples in (64, 128, 64):
        seq.add_block(
            pp.make_arbitrary_grad(
                "x", np.sin(np.linspace(0.0, 3.0, samples)) * 1e5, system=seq.system
            )
        )

    assert seq.event_definitions().gradient.tolist() == [1, 2, 1]


def test_a_delay_carrying_a_trigger_is_not_the_definition_a_plain_wait_is():
    """A block plays something when it fires a trigger, however empty it is.

    Every plain wait is one definition, its length being set at run time. One
    that fires a trigger is not among them.
    """
    seq = pp.Sequence(pp.Opts())
    seq.add_block(pp.make_delay(1e-3))
    seq.add_block(pp.make_delay(5e-3))
    seq.add_block(
        pp.make_delay(1e-3), pp.make_digital_output_pulse("osc0", duration=1e-4)
    )

    played = seq.block_definitions().tolist()

    assert played[0] == played[1]
    assert played[2] != played[0]


def test_a_library_collapsed_across_threads_is_numbered_in_order_of_first_appearance():
    # More rows than one thread collapses, with each distinct gradient first
    # appearing in every range and repeated within rounding of the file.
    seq = pp.Sequence(pp.Opts())
    for n in range(70000):
        level = n // 25 + 1
        seq.add_block(
            pp.make_trapezoid(
                "x",
                amplitude=100 * level * (1 + 1e-9 * (n % 3)),
                flat_time=1e-4,
                rise_time=1e-4,
            )
        )
    seq.remove_duplicates(in_place=True)

    ids = np.array([row[2] for row in seq.block_events.values()])
    np.testing.assert_array_equal(ids, np.arange(70000) // 25 + 1)
    for block in (1, 34999, 70000):
        gradient = seq.get_block(block).gx
        assert gradient.amplitude == pytest.approx(
            100 * ((block - 1) // 25 + 1), rel=1e-6
        )
