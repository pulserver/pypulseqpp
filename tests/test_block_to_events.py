"""A block read back as the events it plays.

Ported from the reference toolbox's `test_block2events`. Upstream hands a
list of events back inside a tuple rather than as one, which its own
docstring says it does not; the toolbox is the authority and is followed.
"""

import math
from types import SimpleNamespace

import pytest

import pypulseqpp as pp
from pypulseqpp import _ext


def test_a_block_is_the_events_it_plays():
    rf = pp.make_block_pulse(math.pi / 2, duration=1e-3)
    gx = pp.make_trapezoid("x", area=1000, duration=1e-3)

    events = pp.block_to_events(SimpleNamespace(rf=rf, gx=gx, gy=None, gz=None))

    assert isinstance(events, tuple)
    assert len(events) == 2


def test_a_block_playing_one_thing_is_one_event():
    gx = pp.make_trapezoid("x", area=1000, duration=1e-3)

    events = pp.block_to_events(SimpleNamespace(rf=None, gx=gx, gy=None, gz=None))

    assert events == (gx,)


def test_events_already_in_a_list_come_back_as_they_went_in():
    gx = pp.make_trapezoid("x", area=1000, duration=1e-3)
    adc = pp.make_adc(num_samples=128, duration=1e-3)

    events = pp.block_to_events([gx, adc])

    assert isinstance(events, tuple)
    assert len(events) == 2


def test_a_list_wrapped_in_a_list_is_the_same_list():
    gx = pp.make_trapezoid("x", area=1000, duration=1e-3)

    assert len(pp.block_to_events([[gx]])) == 1


def test_nothing_at_all_is_no_events():
    assert pp.block_to_events() == ()


def test_only_one_block_at_a_time():
    block = SimpleNamespace(rf=None, gx=None, gy=None, gz=None)

    with pytest.raises(ValueError, match="one block"):
        pp.block_to_events(block, block)


def test_a_block_read_off_a_sequence_reads_back_compiled():
    """What get_block hands over is what add_block takes, either way round."""
    sequence = pp.Sequence(pp.Opts())
    gx = pp.make_trapezoid("x", area=1000, duration=1e-3)
    adc = pp.make_adc(num_samples=64, duration=1e-3)
    sequence.add_block(gx, adc)

    events = pp.block_to_events(sequence.get_block(1))
    played = [event for event in events if hasattr(event, "type")]

    assert {event.type for event in played} == {"trap", "adc"}
    assert all(isinstance(event, _ext.Event) for event in played)
