"""A scan played more than once, written into its block table by ``Sequence.expand_repeats``."""

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import sequences


def built(sections):
    """Pure delays, each with an optional ``ONCE`` value; a block's duration names it."""
    seq = pp.Sequence(pp.Opts())
    for duration, once in sections:
        events = [pp.make_delay(duration)]
        if once is not None:
            events.append(pp.make_label("ONCE", "SET", once))
        seq.add_block(*events)
    return seq


def durations(seq):
    return list(np.round(seq.libraries().block_durations * 1e3, 6))


def labels(seq, name):
    """A label's value in force at each block."""
    return list(np.atleast_1d(seq.evaluate_labels(evolution="blocks").get(name, 0)))


def test_the_body_plays_every_time_the_preparation_first_and_the_cooldown_last():
    seq = built([(1e-3, 1), (2e-3, 0), (3e-3, None), (4e-3, 2)])

    counts = seq.expand_repeats(3)

    assert counts == {
        "repeats": 3,
        "blocks_before": 4,
        "blocks_after": 8,
        "prep_blocks": 1,
        "body_blocks": 2,
        "cooldown_blocks": 1,
    }
    assert durations(seq) == [1, 2, 3, 2, 3, 2, 3, 4]


def test_once_is_read_per_block_where_the_preparation_is_not_one_run_at_the_front():
    seq = built([(1e-3, None), (2e-3, 1), (3e-3, 0), (4e-3, None)])

    seq.expand_repeats(2)

    assert durations(seq) == [1, 2, 3, 4, 1, 3, 4]


def test_each_repetition_past_the_first_numbers_itself_once_on_its_first_block():
    seq = built([(1e-3, 1), (2e-3, 0), (3e-3, None), (4e-3, 2)])

    seq.expand_repeats(3)

    assert labels(seq, "AVG") == [0, 0, 0, 1, 1, 2, 2, 2]
    tables = seq.libraries()
    avg = sum(1 for name in tables.label_set_labels if name == "AVG")
    assert avg == 2


def test_one_repetition_leaves_the_scan_as_it_is():
    seq = built([(1e-3, 1), (2e-3, 0), (4e-3, 2)])

    counts = seq.expand_repeats(1)

    assert counts["blocks_after"] == 3
    assert durations(seq) == [1, 2, 4]
    assert labels(seq, "ONCE") == [1, 0, 2]
    assert "AVG" not in seq.evaluate_labels(evolution="blocks")


def test_an_incremented_counter_restarts_with_every_repetition():
    seq = pp.Sequence(pp.Opts())
    for _ in range(3):
        seq.add_block(pp.make_delay(1e-3), pp.make_label("LIN", "INC", 1))

    seq.expand_repeats(3)

    assert labels(seq, "LIN") == [1, 2, 3] * 3
    assert labels(seq, "AVG") == [0, 0, 0, 1, 1, 1, 2, 2, 2]


def test_a_counter_set_only_in_the_preparation_starts_later_repetitions_at_zero():
    seq = pp.Sequence(pp.Opts())
    seq.add_block(
        pp.make_delay(1e-3),
        pp.make_label("ONCE", "SET", 1),
        pp.make_label("SLC", "SET", 3),
    )
    seq.add_block(pp.make_delay(2e-3), pp.make_label("ONCE", "SET", 0))
    seq.add_block(pp.make_delay(2e-3))

    seq.expand_repeats(2)

    assert labels(seq, "SLC") == [3, 3, 3, 0, 0]


def test_a_counter_the_first_block_of_a_repetition_sets_itself_is_left_to_it():
    seq = pp.Sequence(pp.Opts())
    seq.add_block(pp.make_delay(1e-3), pp.make_label("LIN", "SET", 5))
    seq.add_block(pp.make_delay(1e-3), pp.make_label("LIN", "INC", 1))

    seq.expand_repeats(2)

    assert labels(seq, "LIN") == [5, 6, 5, 6]


def test_a_block_sets_its_once_before_it_increments_it():
    seq = pp.Sequence(pp.Opts())
    seq.add_block(
        pp.make_delay(1e-3),
        pp.make_label("ONCE", "INC", 1),
        pp.make_label("ONCE", "SET", 0),
    )
    seq.add_block(pp.make_delay(2e-3), pp.make_label("ONCE", "SET", 0))

    seq.expand_repeats(2)

    assert durations(seq) == [1, 2, 2]


def test_flags_kept_still_mark_the_preparation_of_the_expanded_scan():
    seq = built([(1e-3, 1), (2e-3, 0), (3e-3, None)])

    seq.expand_repeats(2, strip_once=False)

    assert labels(seq, "ONCE") == [1, 0, 0, 0, 0]


def test_a_scan_without_flags_repeats_whole():
    seq = built([(1e-3, None), (2e-3, None)])

    seq.expand_repeats(4)

    assert durations(seq) == [1, 2] * 4


def test_a_chain_keeps_its_other_extensions_when_its_flag_is_removed():
    seq = pp.Sequence(pp.Opts())
    turn = pp.make_rotation(
        np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    )
    seq.add_block(
        pp.make_delay(1e-3),
        pp.make_label("ONCE", "SET", 0),
        pp.make_label("LIN", "SET", 5),
        turn,
    )

    seq.expand_repeats(2, label="")

    assert labels(seq, "LIN") == [5, 5]
    assert "ONCE" not in seq.evaluate_labels(evolution="blocks")
    assert all(seq.get_block(k).rotation is not None for k in (1, 2))


def test_a_counter_the_scan_already_writes_is_refused():
    seq = pp.Sequence(pp.Opts())
    seq.add_block(pp.make_delay(1e-3), pp.make_label("AVG", "SET", 1))

    with pytest.raises(RuntimeError, match="already writes AVG"):
        seq.expand_repeats(2)


def test_an_empty_label_numbers_nothing_and_leaves_the_scans_counters_alone():
    seq = pp.Sequence(pp.Opts())
    seq.add_block(pp.make_delay(1e-3), pp.make_label("AVG", "SET", 4))

    seq.expand_repeats(3, label="")

    assert labels(seq, "AVG") == [4, 4, 4]


@pytest.mark.parametrize(
    ("sections", "repeats", "error", "match"),
    [
        ([(1e-3, None)], 0, ValueError, "at least 1"),
        ([(1e-3, 1), (2e-3, 2)], 2, RuntimeError, "nothing plays on every repetition"),
        ([(1e-3, 3)], 2, RuntimeError, "ONCE is 0, 1 or 2"),
    ],
)
def test_what_cannot_be_played_more_than_once_is_refused(
    sections, repeats, error, match
):
    with pytest.raises(error, match=match):
        built(sections).expand_repeats(repeats)


def test_the_expanded_scan_reads_back_as_written(tmp_path):
    seq = built([(1e-3, 1), (2e-3, 0), (4e-3, 2)])
    seq.expand_repeats(3)
    path = tmp_path / "expanded.seq"

    seq.write(str(path))
    back = pp.Sequence(pp.Opts())
    back.read(str(path))

    assert durations(back) == [1, 2, 2, 2, 4]
    assert labels(back, "AVG") == [0, 0, 1, 2, 2]


def test_a_shipped_scan_grows_only_its_block_table_and_keeps_its_definitions():
    seq = sequences.bssfp2D_sequence(
        pp.Opts(), n_x=64, n_y=32, n_slices=1, readout_bandwidth_hz=50e3
    )
    seq = seq[-1] if isinstance(seq, list) else seq
    before = seq.libraries()
    definitions = list(seq._native.instance_definitions())

    counts = seq.expand_repeats(2)

    after = seq.libraries()
    prep, body, cool = (
        counts["prep_blocks"],
        counts["body_blocks"],
        counts["cooldown_blocks"],
    )
    assert prep > 0 and cool > 0 and prep + body + cool == counts["blocks_before"]
    for name in ("rf", "trapezoids", "arbitrary_gradients", "adc", "shapes"):
        assert len(getattr(after, name)) == len(getattr(before, name)), name
    body_definitions = definitions[prep : prep + body]
    assert list(seq._native.instance_definitions()) == (
        definitions[:prep] + body_definitions * 2 + definitions[prep + body :]
    )


def test_a_recorded_total_duration_states_the_expanded_scan(tmp_path):
    seq = built([(1e-3, 1), (2e-3, 0)])
    assert seq.check_timing()[0]

    seq.expand_repeats(3)
    path = tmp_path / "expanded.bin"
    pp.io.write(seq, str(path), binary=True)
    back = pp.io.read(str(path))

    assert float(back.get_definition("TotalDuration")[0]) == pytest.approx(7e-3)
    assert back.check_timing()[0]
