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
    assert loaded.duration()[0] == pytest.approx(sequence.duration()[0])


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

    assert sequence.duration()[0] == pytest.approx(5e-3)


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


def test_a_sequence_built_through_the_facade_passes_its_timing_check():
    system = upstream.Opts(
        rf_dead_time=100e-6, rf_ringdown_time=30e-6, adc_dead_time=10e-6
    )
    seq = pp.Sequence(system=system)
    seq.add_block(
        pp.make_block_pulse(math.pi / 2, duration=1e-3, system=system, use="excitation")
    )
    seq.add_block(
        pp.make_trapezoid("x", area=1000, duration=2e-3, system=system),
        pp.make_adc(num_samples=100, duration=2e-3, delay=100e-6, system=system),
    )

    is_ok, error_report = seq.check_timing()

    assert is_ok
    assert error_report == []


def test_a_pulse_that_starts_inside_the_dead_time_is_reported_by_name(capsys):
    """The report names the block, the event and what is wrong with it."""
    system = upstream.Opts(rf_dead_time=100e-6, rf_ringdown_time=30e-6)
    seq = pp.Sequence(system=system)
    pulse = pp.make_block_pulse(
        math.pi / 2, duration=1e-3, system=system, use="excitation"
    )
    pulse.delay = 0.0
    seq.add_block(pulse)

    is_ok, error_report = seq.check_timing(print_errors=True)

    assert not is_ok
    assert [e.error_type for e in error_report] == ["RF_DEAD_TIME"]
    assert error_report[0].block == 1
    assert error_report[0].event == "rf"

    printed = capsys.readouterr().out
    assert "Block 1:" in printed
    assert "rf.delay" in printed
    assert "RF dead time" in printed


def test_checking_timing_without_a_system_says_so():
    with pytest.raises(ValueError, match="needs the system"):
        pp.Sequence().check_timing()


# --------------------------------------------------------------------------
# The methods a design script reaches for besides building blocks.
# --------------------------------------------------------------------------


def test_the_duration_says_what_the_sequence_is_made_of():
    """Three values, in upstream's shape: seconds, blocks, events per column."""
    sequence = gradient_echo()

    seconds, blocks, counts = sequence.duration()

    assert seconds == pytest.approx(sum(sequence.block_durations))
    assert blocks == len(sequence)
    # delay, rf, gx, gy, gz, adc, extension -- upstream's columns, so a
    # caller indexing one of them reaches the same event kind here.
    assert list(counts) == [0.0, 4.0, 8.0, 4.0, 8.0, 4.0, 4.0]


def test_the_duration_matches_upstreams_on_the_same_sequence():
    system = upstream.Opts()
    theirs = upstream.Sequence(system=system)
    ours = pp.Sequence(system)
    for build, module in ((theirs, upstream), (ours, pp)):
        build.add_block(
            module.make_trapezoid("x", area=1000, duration=1e-3, system=system)
        )
        build.add_block(module.make_delay(2e-3))

    their_seconds, their_blocks, their_counts = theirs.duration()
    our_seconds, our_blocks, our_counts = ours.duration()

    assert our_seconds == pytest.approx(their_seconds)
    assert our_blocks == their_blocks
    assert list(our_counts) == list(their_counts)


def test_a_time_lands_in_the_block_playing_then():
    sequence = pp.Sequence(pp.Opts())
    for _ in range(4):
        sequence.add_block(pp.make_delay(1e-3))

    assert sequence.find_block_by_time(0.0) == 1
    assert sequence.find_block_by_time(1.5e-3) == 2
    assert sequence.find_block_by_time(3.5e-3) == 4
    assert sequence.find_block_by_time(10e-3) is None


def test_a_definition_that_was_never_set_reads_as_empty():
    """Upstream's contract: not a KeyError, and not None."""
    assert pp.Sequence().get_definition("Nothing") == ""


def test_definitions_can_be_taken_from_another_sequence():
    source = pp.Sequence(pp.Opts())
    source.set_definition("Name", "gre")
    source.set_definition("FOV", [0.256, 0.256, 0.005])
    target = pp.Sequence(pp.Opts())

    target.copy_definitions(source)

    assert target.get_definition("Name") == "gre"
    assert target.get_definition("FOV") == pytest.approx([0.256, 0.256, 0.005])


def test_an_extension_name_and_its_number_find_each_other():
    sequence = pp.Sequence()

    number = sequence.get_extension_type_ID("TRIGGERS")

    assert sequence.get_extension_type_string(number) == "TRIGGERS"
    assert sequence.get_extension_type_ID("TRIGGERS") == number


def test_an_extension_number_nothing_claims_is_an_error():
    with pytest.raises(ValueError, match="unknown"):
        pp.Sequence().get_extension_type_string(42)


def test_an_extension_can_be_pinned_to_a_chosen_number():
    sequence = pp.Sequence()

    sequence.set_extension_string_ID("ROTATIONS", 7)

    assert sequence.get_extension_type_ID("ROTATIONS") == 7


def test_a_repetition_named_twice_keeps_one_number():
    sequence = pp.Sequence(pp.Opts())

    assert sequence.get_or_create_trid_id("readout") == 1
    assert sequence.get_or_create_trid_id("prep") == 2
    assert sequence.get_or_create_trid_id("readout") == 1


def test_a_repetition_needs_a_name():
    with pytest.raises(ValueError, match="non-empty"):
        pp.Sequence(pp.Opts()).get_or_create_trid_id("")


def test_naming_a_repetition_adds_the_block_that_labels_it():
    sequence = pp.Sequence(pp.Opts())

    sequence.add_trid("readout")

    assert len(sequence) == 1


# -- collapsing ------------------------------------------------------------


def repeated_gradient(times=5):
    sequence = pp.Sequence(pp.Opts())
    gradient = pp.make_trapezoid("x", area=1000, duration=1e-3)
    for _ in range(times):
        sequence.add_block(gradient)
    return sequence


def test_collapsing_duplicates_leaves_the_sequence_it_was_asked_about_alone():
    sequence = repeated_gradient()

    collapsed = sequence.remove_duplicates()

    assert collapsed is not sequence
    assert collapsed._native.num_gradients() == 1
    assert sequence._native.num_gradients() == 5
    assert collapsed.duration()[0] == pytest.approx(sequence.duration()[0])


def test_collapsing_duplicates_in_place_changes_the_sequence():
    sequence = repeated_gradient()

    collapsed = sequence.remove_duplicates(in_place=True)

    assert collapsed is sequence
    assert sequence._native.num_gradients() == 1


def test_a_collapsed_copy_carries_the_definitions_and_the_blocks(tmp_path):
    sequence = repeated_gradient()
    sequence.set_definition("Name", "repeated")

    collapsed = sequence.remove_duplicates()

    assert collapsed.get_definition("Name") == "repeated"
    assert len(collapsed) == len(sequence)


# -- writing ---------------------------------------------------------------


def test_writing_hands_back_the_signature_it_wrote(tmp_path):
    sequence = gradient_echo()
    path = tmp_path / "gre.seq"

    signature = sequence.write(str(path))

    assert signature is not None
    assert f"Hash {signature}" in path.read_text()
    assert sequence.signature_value == signature
    assert sequence.signature_type == "md5"


def test_an_unsigned_file_has_no_signature_to_hand_back(tmp_path):
    sequence = gradient_echo()
    path = tmp_path / "gre.seq"

    assert sequence.write(str(path), create_signature=False) is None
    assert "[SIGNATURE]" not in path.read_text()


def test_writing_collapses_duplicates_without_touching_the_sequence(tmp_path):
    sequence = repeated_gradient()

    sequence.write(str(tmp_path / "repeated.seq"))

    assert sequence._native.num_gradients() == 5
    loaded = pp.Sequence()
    loaded.read(str(tmp_path / "repeated.seq"))
    assert loaded._native.num_gradients() == 1


def test_writing_can_be_asked_to_judge_the_timing_first(tmp_path):
    system = upstream.Opts(rf_dead_time=100e-6, rf_ringdown_time=30e-6)
    sequence = pp.Sequence(system=system)
    pulse = pp.make_block_pulse(
        math.pi / 2, duration=1e-3, system=system, use="excitation"
    )
    pulse.delay = 0.0
    sequence.add_block(pulse)

    with pytest.warns(UserWarning, match="1 timing errors"):
        sequence.write(str(tmp_path / "bad.seq"), check_timing=True)


def test_a_1_4_1_file_can_be_asked_for_through_write(tmp_path):
    sequence = gradient_echo()
    path = tmp_path / "gre141.seq"

    sequence.write(str(path), v141_compat=True)

    assert "minor 4" in path.read_text()


# -- aliases ---------------------------------------------------------------


def test_write_file_writes_the_file_unsigned(tmp_path):
    sequence = gradient_echo()
    path = tmp_path / "gre.seq"

    sequence.write_file(str(path))

    assert "[SIGNATURE]" not in path.read_text()
    assert path.read_text().startswith("# Pulseq sequence file")


def test_read_binary_reads_the_binary_form(tmp_path):
    sequence = gradient_echo()
    path = tmp_path / "gre.bin"
    sequence.write_binary(str(path))

    loaded = pp.Sequence()
    loaded.read_binary(str(path))

    assert len(loaded) == len(sequence)


def test_the_binary_form_is_never_signed(tmp_path):
    """It has no signature section, so there is nothing to hand back."""
    assert gradient_echo().write_binary(str(tmp_path / "gre.bin")) is None


# -- no-ops ----------------------------------------------------------------


def test_the_caches_a_script_asks_about_are_empty_because_there_are_none():
    sequence = gradient_echo()

    assert sequence.block_cache_size == 0
    assert sequence.event_cache_size == 0
    sequence.clear_block_cache()
    sequence.clear_event_cache()
    sequence.clear_caches()

    assert len(sequence) == 12


def test_a_cache_flag_reads_back_as_it_was_set():
    sequence = pp.Sequence(pp.Opts(), use_block_cache=False)

    assert sequence.use_block_cache is False
    sequence.use_block_cache = True
    assert sequence.use_block_cache is True

    sequence.use_event_cache = False
    assert sequence.use_event_cache is False


def test_a_sequence_can_be_used_as_a_context():
    with pp.Sequence(pp.Opts()) as sequence:
        sequence.add_block(pp.make_delay(1e-3))

    assert len(sequence) == 1


def test_printing_a_sequence_says_what_is_in_it():
    printed = str(gradient_echo())

    assert printed.startswith("Sequence:")
    assert "blocks: 12" in printed
