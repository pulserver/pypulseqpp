"""Built-in label IDs and CustomLabels names across text and binary files."""

import numpy as np
import pytest

from pypulseqpp import _ext


def labelled(*names):
    """A sequence setting each of ``names`` on a block of its own."""
    sequence = _ext.Sequence()
    gradient = sequence.register_trap(np.array([2000.0, 1e-4, 2e-3, 1e-4, 0.0]))
    for name in names:
        label = sequence.register_label_set(1, sequence.label_id(name))
        chain = sequence.chain_extension(
            sequence.extension_type_id("LABELSET"), label, 0
        )
        sequence.add_block(0, gradient, 0, 0, 0, chain, 2.2e-3)
    return sequence


def label_names(sequence):
    """The names in the `LABELSET` rows of what ``sequence`` writes."""
    written = _ext.write_text(sequence, True).decode()
    inside = False
    out = []
    for line in written.splitlines():
        if line.startswith("extension "):
            inside = line.startswith("extension LABELSET")
        elif line.startswith("["):
            inside = False
        elif inside and line and not line.startswith("#"):
            out.append(line.split()[-1])
    return out


def custom_labels(sequence):
    """The `CustomLabels` definition of what ``sequence`` writes, or None."""
    return _ext.read(_ext.write_text(sequence, True)).definitions().get("CustomLabels")


def test_a_builtin_label_needs_nothing_said_about_it():
    """Every toolbox numbers the builtin table the same way."""
    assert custom_labels(labelled("LIN", "SLC", "NOISE")) is None


def test_a_label_the_table_does_not_carry_is_named_in_the_definitions():
    assert custom_labels(labelled("LIN", "SPARKLE")) == "SPARKLE"


def test_only_the_names_past_the_table_are_listed():
    """The builtins are shared, so listing them would be overhead for nothing."""
    assert custom_labels(labelled("SPARKLE", "LIN", "GLITTER")) == "SPARKLE GLITTER"


@pytest.mark.parametrize("form", ["text", "binary"])
def test_a_custom_label_comes_back_by_its_name(form):
    """Through the binary form too, where only the number crosses."""
    sequence = labelled("LIN", "SPARKLE", "GLITTER")
    written = (
        _ext.write_text(sequence, True)
        if form == "text"
        else _ext.write_binary(sequence)
    )

    assert label_names(_ext.read(written)) == ["LIN", "SPARKLE", "GLITTER"]


def test_the_names_survive_a_second_trip_through_the_binary_form():
    """Reading registers them in order, so they land on the ids they had."""
    once = _ext.read(_ext.write_binary(labelled("SPARKLE", "LIN", "GLITTER")))

    twice = _ext.read(_ext.write_binary(once))

    assert label_names(twice) == ["SPARKLE", "LIN", "GLITTER"]
    assert custom_labels(twice) == "SPARKLE GLITTER"


def test_a_file_naming_no_custom_label_still_reads():
    """The definition is optional; a file without one is not broken."""
    contents = _ext.write_text(labelled("LIN"), True)

    assert b"CustomLabels" not in contents
    assert label_names(_ext.read(contents)) == ["LIN"]


def test_a_repeated_label_statement_is_registered_once_and_its_chain_shared():
    import pypulseqpp as pp

    seq = pp.Sequence(pp.Opts())
    delay = pp.make_delay(1e-3)
    for line in range(40):
        seq.add_block(delay, pp.make_label("REV", "SET", line % 2))
    assert len(seq.libraries().label_set_values) == 2
    assert len(seq.block_events) == 40
    chains = {int(seq.block_events[i][6]) for i in range(1, 41)}
    assert len(chains) == 2
    values = [seq.get_block(i).label[0].value for i in range(1, 41)]
    assert values == [line % 2 for line in range(40)]


def test_a_label_beside_a_trigger_still_reads_back_with_its_block():
    import pypulseqpp as pp

    seq = pp.Sequence(pp.Opts())
    for _ in range(3):
        seq.add_block(
            pp.make_delay(1e-3),
            pp.make_label("LIN", "SET", 5),
            pp.make_trigger("physio1", duration=1e-4),
        )
    for i in range(1, 4):
        block = seq.get_block(i)
        assert block.label[0].value == 5
        assert block.trig is not None
