"""Block definitions folded into nested loops."""

import math

import pytest

import pypulseqpp as pp


def played(loops):
    """The block definitions the loops play, in order."""
    blocks = []
    for item in loops:
        if isinstance(item, int):
            blocks.append(item)
        else:
            body, count = item
            blocks.extend(played(body) * count)
    return blocks


@pytest.fixture
def events():
    system = pp.Opts()
    return {
        "inversion": pp.make_block_pulse(math.pi, duration=2e-3, system=system),
        "excitation": pp.make_block_pulse(math.pi / 18, duration=0.5e-3, system=system),
        "refocusing": pp.make_block_pulse(math.pi, duration=1e-3, system=system),
        "readout": pp.make_trapezoid("x", area=1000, duration=1.5e-3, system=system),
        "adc": pp.make_adc(64, duration=1e-3, delay=0.25e-3, system=system),
        "encode": pp.make_trapezoid("y", area=500, duration=1e-3, system=system),
    }


def prepared_trains(events, shots, trains=1, views=4, acquire=lambda shot: True):
    """Per shot an inversion, then trains of excitation, encode and readout, each after a wait."""
    sequence = pp.Sequence(pp.Opts())
    for shot in range(shots):
        sequence.add_block(events["inversion"])
        for _ in range(trains):
            sequence.add_block(pp.make_delay(0.5))
            for view in range(views):
                sequence.add_block(events["excitation"])
                sequence.add_block(pp.scale_grad(events["encode"], (view + 1) / views))
                adc = [events["adc"]] if acquire(shot) else []
                sequence.add_block(events["readout"], *adc)
        sequence.add_block(pp.make_delay(1.0))
    return sequence


def definitions(sequence, *blocks):
    """The definition ids of the given 1-based blocks."""
    table = sequence.block_definitions()
    return [int(table[block - 1]) for block in blocks]


def test_a_prepared_train_is_a_loop_inside_the_repetition(events):
    sequence = prepared_trains(events, shots=3)
    inversion, wait, excitation, encode, readout = definitions(sequence, 1, 2, 3, 4, 5)

    assert sequence.loops() == [
        ([inversion, wait, ([excitation, encode, readout], 4), wait], 3)
    ]


def test_dummy_shots_without_an_adc_are_iterations_of_the_same_loop(events):
    with_dummies = prepared_trains(events, shots=4, acquire=lambda shot: shot >= 2)

    assert with_dummies.loops() == prepared_trains(events, shots=4).loops()


def test_loops_nest_as_deep_as_the_scan(events):
    sequence = prepared_trains(events, shots=3, trains=2)
    inversion, wait, excitation, encode, readout = definitions(sequence, 1, 2, 3, 4, 5)
    train = ([excitation, encode, readout], 4)

    assert sequence.loops() == [([inversion, wait, ([train, wait], 2)], 3)]


def test_a_preparation_before_the_loop_stays_outside_it(events):
    sequence = pp.Sequence(pp.Opts())
    sequence.add_block(events["inversion"])
    sequence.add_block(pp.make_delay(10e-3))
    for _ in range(3):
        sequence.add_block(events["excitation"])
        for _ in range(4):
            sequence.add_block(events["refocusing"])
    inversion, wait, excitation, refocusing = definitions(sequence, 1, 2, 3, 4)

    assert sequence.loops() == [inversion, wait, ([excitation, ([refocusing], 4)], 3)]
    assert sequence.repetition() == (len(sequence), 1)


def test_the_outer_loop_is_the_repetition(events):
    """Trains of three and four echoes alternate, with a navigator after every fourth.

    Folded from its first block without the repetition, the scan would repeat
    the first pair of trains and then start an outer loop at the navigator.
    """
    sequence = pp.Sequence(pp.Opts())
    for train in range(12):
        sequence.add_block(events["excitation"])
        for _ in range(4 if train % 2 else 3):
            sequence.add_block(events["refocusing"])
        sequence.add_block(pp.make_delay(5e-3))
        if train % 4 == 3:
            sequence.add_block(events["readout"], events["adc"])
            sequence.add_block(pp.make_delay(5e-3))
    excitation, refocusing, wait, navigator = definitions(sequence, 1, 2, 5, 23)
    pair = (
        [excitation, ([refocusing], 3), wait, excitation, ([refocusing], 4), wait],
        2,
    )

    assert sequence.repetition() == (24, 1)
    assert sequence.loops() == [([pair, navigator, wait], 3)]


def test_a_wait_closes_the_iteration_it_follows(events):
    """Navigators in groups of three, each followed by a wait, with one more wait per group.

    The second wait of a group is neither a loop of two waits nor the start of
    the next group.
    """
    sequence = pp.Sequence(pp.Opts())
    for _ in range(5):
        for _ in range(3):
            sequence.add_block(events["readout"], events["adc"])
            sequence.add_block(pp.make_delay(25e-3))
        sequence.add_block(pp.make_delay(20e-6))
    navigator, wait = definitions(sequence, 1, 2)

    assert sequence.loops() == [([([navigator, wait], 3), wait], 5)]


@pytest.mark.parametrize("shots", [1, 2, 5])
@pytest.mark.parametrize("views", [1, 3, 8])
def test_the_loops_play_the_block_definitions_back(events, shots, views):
    sequence = prepared_trains(events, shots=shots, trains=2, views=views)
    sequence.add_block(events["readout"])

    assert played(sequence.loops()) == sequence.block_definitions().tolist()


def test_a_sequence_that_does_not_repeat_is_its_block_definitions(events):
    sequence = pp.Sequence(pp.Opts())
    for name in ("inversion", "excitation", "refocusing", "readout"):
        sequence.add_block(events[name])

    assert sequence.loops() == sequence.block_definitions().tolist()


def test_a_sequence_with_nothing_in_it_has_no_loops():
    assert pp.Sequence(pp.Opts()).loops() == []


def test_asking_for_loops_writes_nothing_into_the_sequence(events, tmp_path):
    sequence = prepared_trains(events, shots=2)
    before = tmp_path / "before.seq"
    after = tmp_path / "after.seq"
    sequence.write(before, create_signature=False)

    sequence.loops()
    sequence.write(after, create_signature=False)

    assert after.read_text() == before.read_text()
