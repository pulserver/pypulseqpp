"""Where a label is stated, as opposed to the value it holds."""

import pytest

import pypulseqpp as pp


def repetitions(count=3):
    """Each repetition sets TRID to the same group at its head and counts LIN."""
    seq = pp.Sequence(pp.Opts())
    for _ in range(count):
        seq.add_block(pp.make_label("TRID", "SET", 1), pp.make_delay(1e-3))
        seq.add_block(pp.make_label("LIN", "INC", 1), pp.make_delay(2e-3))
    return seq


def test_every_block_that_sets_a_label_is_listed_once():
    seq = repetitions()
    seq.add_block(
        pp.make_label("TRID", "SET", 2),
        pp.make_label("TRID", "SET", 3),
        pp.make_delay(1e-3),
    )

    assert seq.label_blocks("TRID").tolist() == [1, 3, 5, 7]


def test_a_label_set_to_the_value_it_holds_is_still_stated_where_it_is_set():
    """The value alone cannot say where each repetition starts."""
    seq = repetitions()

    assert set(seq.evaluate_labels(evolution="blocks")["TRID"].tolist()) == {1}
    assert seq.label_blocks("TRID").tolist() == [1, 3, 5]


def test_increments_are_found_apart_from_settings():
    seq = repetitions()

    assert seq.label_blocks("LIN", "INC").tolist() == [2, 4, 6]
    assert seq.label_blocks("LIN").tolist() == []
    assert seq.label_blocks("TRID", "INC").tolist() == []


def test_a_label_no_block_states_is_stated_nowhere():
    assert repetitions().label_blocks("SLC").tolist() == []
    assert pp.Sequence(pp.Opts()).label_blocks("TRID").tolist() == []


def test_a_kind_that_is_neither_a_setting_nor_an_increment_is_refused():
    with pytest.raises(ValueError, match="'SET' or 'INC'"):
        repetitions().label_blocks("TRID", "set")
