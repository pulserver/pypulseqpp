"""The namespace a design script imports.

`import pypulseqpp as pp` has to be `import pypulseq as pp` with the same
functions and the same signatures, so what is tested here is that upstream's
own vocabulary still works -- against events it was never written for.
"""

import math

import pypulseq as upstream
import pytest

import pypulseqpp as pp
from pypulseqpp import _ext


def test_every_factory_hands_back_a_compiled_event():
    """The whole point: a field is a slot, not a dictionary entry."""
    made = [
        pp.make_trapezoid("x", area=1000, duration=1e-3),
        pp.make_adc(num_samples=64, duration=1e-3),
        pp.make_delay(1e-3),
        pp.make_block_pulse(math.pi / 2, duration=1e-3, use="excitation"),
        pp.make_trigger("physio1", duration=1e-3),
    ]

    assert all(isinstance(event, _ext.Event) for event in made)


def test_a_factory_that_returns_several_converts_each_of_them():
    pulse, select, rephase = pp.make_sinc_pulse(
        flip_angle=math.pi / 8, duration=1e-3, slice_thickness=3e-3, return_gz=True
    )

    assert isinstance(pulse, _ext.RfEvent)
    assert isinstance(select, _ext.Event)
    assert isinstance(rephase, _ext.Event)


@pytest.mark.parametrize(
    "call",
    [
        lambda pp, gx: pp.calc_duration(gx),
        lambda pp, gx: pp.split_gradient_at(gx, 5e-4),
        lambda pp, gx: pp.align(right=[gx]),
        lambda pp, gx: pp.scale_grad(gx, 0.5),
    ],
    ids=["calc_duration", "split_gradient_at", "align", "scale_grad"],
)
def test_upstreams_own_helpers_take_a_compiled_event(call):
    """They check isinstance and deepcopy, and an event satisfies neither.

    The decorator hands them a namespace on the way in, which is what makes
    the whole upstream namespace usable without forking any of it.
    """
    call(pp, pp.make_trapezoid("x", area=1000, duration=1e-3))


def test_a_helper_gives_back_what_the_rest_of_the_package_wants():
    gx = pp.make_trapezoid("x", area=1000, duration=1e-3)

    halves = pp.split_gradient_at(gx, 5e-4)

    assert all(isinstance(part, _ext.Event) for part in halves)


def test_scaling_a_gradient_stays_in_the_compiled_form():
    """The hot call: a phase encode is scaled once per shot."""
    gx = pp.make_trapezoid("x", area=1000, duration=1e-3)

    scaled = pp.scale_grad(gx, 0.5)

    assert isinstance(scaled, _ext.TrapEvent)
    assert scaled.amplitude == pytest.approx(gx.amplitude * 0.5)


def test_calc_duration_agrees_with_upstream():
    """Same answer, whichever representation the event is in."""
    theirs = upstream.make_trapezoid("x", area=1000, duration=1e-3)
    ours = pp.make_trapezoid("x", area=1000, duration=1e-3)

    assert pp.calc_duration(ours) == pytest.approx(upstream.calc_duration(theirs))


def test_a_namespace_and_a_compiled_event_convert_both_ways():
    gx = pp.make_trapezoid("x", area=1000, duration=1e-3)

    namespace = pp.as_namespace(gx)
    back = pp.convert(namespace)

    assert namespace.type == "trap"
    assert namespace.area == pytest.approx(gx.area)
    assert isinstance(back, _ext.TrapEvent)
    assert back.amplitude == gx.amplitude


# --------------------------------------------------------------------------
# What is ours rather than upstream's, and why.
# --------------------------------------------------------------------------


def test_upstream_has_no_rotation_or_shim_and_this_does():
    """Both arrived with Pulseq 1.5.1; upstream 1.5.0 predates them."""
    assert not hasattr(upstream, "make_rotation")
    assert not hasattr(upstream, "make_rf_shim")

    assert pp.make_rotation(object()).type == "rot3D"
    assert pp.make_rf_shim([complex(1, 0), complex(0.5, 0.25)]).type == "rf_shim"


def test_a_label_upstream_refuses_is_accepted_here():
    """A label is named, not numbered, so the list need not be closed."""
    with pytest.raises(ValueError):
        upstream.make_label(label="SPARKLE", type="SET", value=1)

    event = pp.make_label("SPARKLE", "SET", 1)

    assert isinstance(event, _ext.LabelEvent)
    assert (event.label, event.value, event.type) == ("SPARKLE", 1, "labelset")


def test_a_label_still_has_to_say_what_it_does():
    with pytest.raises(ValueError, match="SET"):
        pp.make_label("LIN", "TOGGLE", 1)


# --------------------------------------------------------------------------
# The sequence.
# --------------------------------------------------------------------------


def test_the_sequence_is_ours_not_upstreams():
    assert pp.Sequence is not upstream.Sequence


def gradient_echo(lines=4):
    sequence = pp.Sequence(pp.Opts())
    pulse, select, rephase = pp.make_sinc_pulse(
        flip_angle=math.pi / 8, duration=1e-3, slice_thickness=3e-3, return_gz=True
    )
    read = pp.make_trapezoid("x", area=1000, duration=1e-3)
    encode = pp.make_trapezoid("y", area=1000, duration=1e-3)
    window = pp.make_adc(num_samples=64, duration=1e-3)
    for line in range(lines):
        sequence.add_block(pulse, select)
        sequence.add_block(read, pp.scale_grad(encode, line / lines), rephase)
        sequence.add_block(read, window, pp.make_label("LIN", "SET", line))
    return sequence


def test_a_sequence_built_through_the_facade_writes_and_reads_back(tmp_path):
    sequence = gradient_echo()
    assert len(sequence) == 12

    sequence.write(str(tmp_path / "gre.seq"))
    loaded = pp.Sequence()
    loaded.read(str(tmp_path / "gre.seq"))

    assert len(loaded) == len(sequence)
    assert loaded.duration() == pytest.approx(sequence.duration())


def test_the_binary_form_round_trips_through_the_facade(tmp_path):
    sequence = gradient_echo()

    sequence.write_binary(str(tmp_path / "gre.bin"))
    loaded = pp.Sequence()
    loaded.read(str(tmp_path / "gre.bin"))

    assert len(loaded) == len(sequence)


def test_a_definition_survives_a_write_and_a_read(tmp_path):
    sequence = gradient_echo()
    sequence.set_definition("Name", "gre")

    sequence.write(str(tmp_path / "gre.seq"))
    loaded = pp.Sequence()
    loaded.read(str(tmp_path / "gre.seq"))

    assert loaded.get_definition("Name") == "gre"


def test_a_block_lasts_as_long_as_the_longest_thing_in_it():
    sequence = pp.Sequence(pp.Opts())
    read = pp.make_trapezoid("x", area=1000, duration=1e-3)

    sequence.add_block(read, pp.make_delay(5e-3))

    assert sequence.duration() == pytest.approx(5e-3)


def test_the_namespace_carries_what_upstream_carries():
    """A script written against upstream finds every name it reaches for."""
    missing = [
        name
        for name in dir(upstream)
        if not name.startswith("_")
        and name not in {"compress_shape", "convert", "decompress_shape"}
        and not hasattr(pp, name)
    ]

    assert missing == []
