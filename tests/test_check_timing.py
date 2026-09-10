"""Timing findings against the reference toolbox; soft-delay ID zero is valid."""

import numpy as np
import pytest

pytest.importorskip(
    "pypulseq_matlab_like",
    reason="the toolbox that defines the format; see reference.py",
)

import convert
import pypulseq_matlab_like as pp
from pypulseq_matlab_like.check_timing import check_timing as toolbox_check_timing
from pypulseq_matlab_like.compress_shape import compress_shape

import pypulseqpp
from pypulseqpp import _ext


def system():
    """A system with dead times long enough that clipping one is visible."""
    return pp.Opts(
        rf_dead_time=100e-6,
        rf_ringdown_time=30e-6,
        adc_dead_time=10e-6,
        rf_raster_time=1e-6,
        grad_raster_time=10e-6,
        adc_raster_time=100e-9,
        block_duration_raster=10e-6,
    )


@pytest.fixture
def build():
    """A four-block sequence holding one event of every kind, all playable."""

    def make():
        opts = system()
        seq = pp.Sequence(system=opts)
        seq.use_block_cache = False
        rf = pp.make_sinc_pulse(
            flip_angle=np.pi / 6,
            duration=1e-3,
            system=opts,
            use="excitation",
            return_gz=False,
        )
        gx = pp.make_trapezoid(channel="x", flat_area=1000, flat_time=2e-3, system=opts)
        adc = pp.make_adc(
            num_samples=100, duration=2e-3, delay=gx.rise_time, system=opts
        )
        gy = pp.make_arbitrary_grad(
            channel="y", waveform=np.linspace(0, 1000, 20), system=opts
        )
        trigger = pp.make_digital_output_pulse(
            channel="osc0", duration=100e-6, system=opts
        )
        seq.add_block(rf)
        seq.add_block(gx, adc)
        seq.add_block(gy)
        seq.add_block(trigger, pp.make_delay(1e-3))
        return seq

    return make


@pytest.fixture
def build_soft_delays():
    """Two soft delays under one numeric id, standing for the same wait."""

    def make(hints=("TE", "TE"), numbers=(1, 1)):
        opts = system()
        seq = pp.Sequence(system=opts)
        seq.use_block_cache = False
        rf = pp.make_sinc_pulse(
            flip_angle=np.pi / 6,
            duration=1e-3,
            system=opts,
            use="excitation",
            return_gz=False,
        )
        for hint, number in zip(hints, numbers, strict=True):
            seq.add_block(rf)
            seq.add_block(
                pp.make_delay(5e-3),
                pp.make_soft_delay(numID=number, hint=hint, offset=0.0, factor=1.0),
            )
        return seq

    return make


def findings(seq):
    """What pypulseqpp finds wrong with ``seq``, judged over the same rows."""
    opts = seq.system
    return _ext.check_timing(
        convert.to_core(seq),
        rf_raster_time=opts.rf_raster_time,
        grad_raster_time=opts.grad_raster_time,
        adc_raster_time=opts.adc_raster_time,
        block_duration_raster=opts.block_duration_raster,
        rf_dead_time=opts.rf_dead_time,
        rf_ringdown_time=opts.rf_ringdown_time,
        adc_dead_time=opts.adc_dead_time,
        adc_samples_divisor=getattr(opts, "adc_samples_divisor", 1.0) or 1.0,
    )


def _plain(value):
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def assert_agrees_with_toolbox(seq, kinds=()):
    """Hold both reports equal, and that ``kinds`` are among what they say.

    Parameters
    ----------
    seq
        The sequence to judge.
    kinds : iterable of str
        Error types the break is meant to produce. Naming them keeps a test
        that stops breaking anything from passing quietly.
    """
    theirs = [
        {key: _plain(value) for key, value in e.__dict__.items()}
        for e in toolbox_check_timing(seq)[1]
    ]
    ours = findings(seq)

    assert [e["error_type"] for e in ours] == [e["error_type"] for e in theirs]
    for mine, theirs_ in zip(ours, theirs, strict=True):
        assert set(mine) == set(theirs_)
        for key in theirs_:
            if isinstance(theirs_[key], float):
                assert mine[key] == pytest.approx(theirs_[key], rel=1e-12, abs=1e-15)
            else:
                assert mine[key] == theirs_[key]

    assert set(kinds) <= {e["error_type"] for e in ours}
    return ours


def rf_sample_times(seq, times):
    """Point the sequence's pulse at a time shape holding ``times``."""
    row = list(seq.rf_library.data[1])
    samples = len(times)
    for column, values in (
        (1, np.ones(samples)),
        (2, np.zeros(samples)),
        (3, np.asarray(times, dtype=float)),
    ):
        shape = compress_shape(np.asarray(values, dtype=float))
        new_id = max(seq.shape_library.data) + 1
        seq.shape_library.insert(
            new_id, np.concatenate([[shape.num_samples], shape.data]), ""
        )
        row[column] = new_id
    seq.rf_library.data[1] = row
    return seq


# -- a sequence with nothing wrong with it -----------------------------


def test_a_playable_sequence_has_nothing_reported_against_it(build):
    assert assert_agrees_with_toolbox(build()) == []


def test_soft_delays_standing_for_the_same_wait_are_not_reported(build_soft_delays):
    assert assert_agrees_with_toolbox(build_soft_delays()) == []


# -- times the sequencer cannot address --------------------------------


def test_an_rf_delay_between_two_rf_ticks_is_reported(build):
    seq = build()
    seq.rf_library.data[1] = list(seq.rf_library.data[1])
    seq.rf_library.data[1][5] = 123.4e-6
    assert_agrees_with_toolbox(seq, kinds=["RASTER"])


def test_a_negative_rf_delay_is_reported(build):
    seq = build()
    seq.rf_library.data[1] = list(seq.rf_library.data[1])
    seq.rf_library.data[1][5] = -1e-5
    assert_agrees_with_toolbox(seq, kinds=["NEGATIVE_DELAY"])


def test_a_negative_gradient_delay_is_reported(build):
    seq = build()
    seq.grad_library.data[1] = list(seq.grad_library.data[1])
    seq.grad_library.data[1][4] = -1e-5
    assert_agrees_with_toolbox(seq, kinds=["NEGATIVE_DELAY"])


def test_a_ramp_between_two_gradient_ticks_is_reported(build):
    seq = build()
    seq.grad_library.data[1] = list(seq.grad_library.data[1])
    seq.grad_library.data[1][1] = 123e-6
    assert_agrees_with_toolbox(seq, kinds=["RASTER"])


def test_a_dwell_between_two_adc_ticks_is_reported(build):
    seq = build()
    seq.adc_library.data[1] = list(seq.adc_library.data[1])
    seq.adc_library.data[1][1] = 3.13e-7
    assert_agrees_with_toolbox(seq, kinds=["RASTER"])


def test_a_dwell_shorter_than_the_adc_raster_is_reported(build):
    seq = build()
    seq.adc_library.data[1] = list(seq.adc_library.data[1])
    seq.adc_library.data[1][1] = 3.13e-8
    assert_agrees_with_toolbox(seq, kinds=["RASTER"])


def test_a_trigger_lasting_between_two_ticks_is_reported(build):
    seq = build()
    seq.trigger_library.data[1] = list(seq.trigger_library.data[1])
    seq.trigger_library.data[1][3] = 3.4e-7
    assert_agrees_with_toolbox(seq, kinds=["RASTER"])


def test_a_block_lasting_between_two_ticks_is_reported(build):
    seq = build()
    seq.block_durations[4] = 1.0034e-3
    assert_agrees_with_toolbox(seq, kinds=["RASTER"])


def test_rf_samples_spaced_off_the_sample_raster_are_reported(build):
    samples = 40
    seq = rf_sample_times(build(), (np.arange(1, samples + 1) - 0.5) * 0.35)
    assert_agrees_with_toolbox(seq, kinds=["RASTER"])


def test_rf_sample_times_off_the_rf_raster_are_reported(build):
    samples = 40
    seq = rf_sample_times(build(), np.cumsum(np.linspace(1.0, 1.7, samples)))
    assert_agrees_with_toolbox(seq, kinds=["RASTER"])


# -- settling windows --------------------------------------------------


def test_a_pulse_starting_inside_the_rf_dead_time_is_reported(build):
    seq = build()
    seq.rf_library.data[1] = list(seq.rf_library.data[1])
    seq.rf_library.data[1][5] = 0.0
    assert_agrees_with_toolbox(seq, kinds=["RF_DEAD_TIME"])


def test_a_block_ending_before_the_coil_has_rung_down_is_reported(build):
    seq = build()
    seq.block_durations[1] = seq.block_durations[1] - 200e-6
    assert_agrees_with_toolbox(seq, kinds=["RF_RINGDOWN_TIME"])


def test_an_adc_opening_inside_its_dead_time_is_reported(build):
    seq = build()
    seq.adc_library.data[1] = list(seq.adc_library.data[1])
    seq.adc_library.data[1][2] = 0.0
    assert_agrees_with_toolbox(seq, kinds=["ADC_DEAD_TIME"])


def test_an_adc_running_past_the_end_of_its_block_is_reported(build):
    seq = build()
    seq.block_durations[2] = 2.0e-3
    assert_agrees_with_toolbox(
        seq, kinds=["POST_ADC_DEAD_TIME", "BLOCK_DURATION_MISMATCH"]
    )


def test_a_stored_duration_shorter_than_the_block_content_is_reported(build):
    seq = build()
    seq.block_durations[3] = 100e-6
    assert_agrees_with_toolbox(seq, kinds=["BLOCK_DURATION_MISMATCH"])


def test_a_sample_count_the_reconstruction_cannot_divide_is_reported(build):
    seq = build()
    seq.system.adc_samples_divisor = 8
    assert_agrees_with_toolbox(seq, kinds=["ADC_SAMPLES_DIVISOR"])


# -- soft delays -------------------------------------------------------


def test_a_soft_delay_with_a_zero_factor_is_reported(build_soft_delays):
    seq = build_soft_delays()
    seq.soft_delay_library.data[1] = list(seq.soft_delay_library.data[1])
    seq.soft_delay_library.data[1][2] = 0.0
    assert_agrees_with_toolbox(seq, kinds=["SOFT_DELAY_FACTOR"])


def test_one_soft_delay_standing_for_two_waits_is_reported(build_soft_delays):
    seq = build_soft_delays()
    seq.block_durations[4] = 7e-3
    assert_agrees_with_toolbox(seq, kinds=["SOFT_DELAY_DUR_INCONSISTENCY"])


def test_one_numeric_id_under_two_names_is_reported(build_soft_delays):
    seq = build_soft_delays(hints=("TE", "TR"), numbers=(1, 2))
    row = list(seq.soft_delay_library.data[2])
    row[0] = 1.0
    seq.soft_delay_library.data[2] = row
    assert_agrees_with_toolbox(seq, kinds=["SOFT_DELAY_HINT_INCONSISTENCY"])


def test_a_soft_delay_numbered_below_zero_is_reported(build_soft_delays):
    seq = build_soft_delays()
    for key in list(seq.soft_delay_library.data):
        row = list(seq.soft_delay_library.data[key])
        row[0] = -1.0
        seq.soft_delay_library.data[key] = row
    assert_agrees_with_toolbox(seq, kinds=["SOFT_DELAY_INVALID_NUMID"])


def test_a_soft_delay_numbered_zero_is_addressable(build_soft_delays):
    """The toolbox writes the first soft delay as zero, so zero is a number.

    Its `check_timing` module rejects zero, which condemns its own reference
    files; MATLAB Pulseq accepts it. Nothing is reported here.
    """
    seq = build_soft_delays()
    for key in list(seq.soft_delay_library.data):
        row = list(seq.soft_delay_library.data[key])
        row[0] = 0.0
        seq.soft_delay_library.data[key] = row

    assert findings(seq) == []
    assert [e.error_type for e in toolbox_check_timing(seq)[1]] == [
        "SOFT_DELAY_INVALID_NUMID",
        "SOFT_DELAY_INVALID_NUMID",
    ]


# -- gradients that do not carry on from one another -------------------
#
# What is asked of a sequence as a whole rather than of one event time: a
# waveform has to be picked up where the axis already is and left where the
# next one will find it, or the amplifier is asked for a step in no time at
# all. The toolbox refuses two of these four when the block is added and
# reports the other two, so where it can hold the sequence at all the two
# verdicts are compared, and where it cannot, its refusal is what is held.

#: How high the ramps here reach, in Hz/m.
HELD = 1e6


def ramp(factory, opts, start, stop, **kwargs):
    """One axis ramping between two amplitudes, as ``factory`` builds it."""
    return factory.make_arbitrary_grad(
        channel="x",
        waveform=np.linspace(start, stop, 40),
        first=float(start),
        last=float(stop),
        system=opts,
        **kwargs,
    )


def as_ours(theirs):
    """The toolbox's sequence, judged by this package."""
    ours = pypulseqpp.Sequence(theirs.system)
    ours._native = convert.to_core(theirs)
    return ours


def kinds(error_report):
    """The kinds of problem a report names."""
    return {finding.error_type for finding in error_report}


def only(error_report, kind):
    """The one finding of ``kind`` in ``error_report``."""
    return next(finding for finding in error_report if finding.error_type == kind)


def test_a_gradient_starting_where_the_last_one_did_not_end_is_reported():
    opts = system()
    theirs = pp.Sequence(system=opts)
    theirs.add_block(ramp(pp, opts, 0, HELD))
    theirs.add_block(ramp(pp, opts, 0, HELD))

    is_ok, found = as_ours(theirs).check_timing()

    assert not is_ok
    assert not theirs.check_timing()[0]
    jump = only(found, "GRADIENT_DISCONTINUITY")
    assert jump.block == 2
    assert jump.event == "gx"
    assert jump.before == pytest.approx(HELD)
    assert jump.value == pytest.approx(0.0)


def test_a_sequence_ending_with_a_gradient_still_on_is_reported():
    opts = system()
    theirs = pp.Sequence(system=opts)
    theirs.add_block(ramp(pp, opts, 0, HELD))

    is_ok, found = as_ours(theirs).check_timing()

    assert not is_ok
    assert not theirs.check_timing()[0]
    assert only(found, "GRADIENT_NOT_RAMPED_DOWN").block == 1


def test_a_gradient_picked_up_after_a_delay_is_reported():
    opts = system()
    ours = pypulseqpp.Sequence(opts)
    ours.add_block(ramp(pypulseqpp, opts, 0, HELD))
    ours.add_block(ramp(pypulseqpp, opts, HELD, 0, delay=20e-6))

    is_ok, found = ours.check_timing()

    assert not is_ok
    late = only(found, "GRADIENT_START_DELAY")
    assert late.block == 2
    assert late.value == pytest.approx(20e-6)
    assert late.amplitude == pytest.approx(HELD)

    theirs = pp.Sequence(system=opts)
    theirs.add_block(ramp(pp, opts, 0, HELD))
    with pytest.raises(RuntimeError):
        theirs.add_block(ramp(pp, opts, HELD, 0, delay=20e-6))


def test_a_gradient_left_on_before_the_block_ends_is_reported():
    opts = system()
    ours = pypulseqpp.Sequence(opts)
    ours.add_block(ramp(pypulseqpp, opts, 0, HELD), pypulseqpp.make_delay(1e-3))
    ours.add_block(ramp(pypulseqpp, opts, HELD, 0))

    is_ok, found = ours.check_timing()

    assert not is_ok
    short = only(found, "GRADIENT_END_NONZERO")
    assert short.block == 1
    assert short.amplitude == pytest.approx(HELD)
    assert short.duration == pytest.approx(1e-3)

    theirs = pp.Sequence(system=opts)
    with pytest.raises(RuntimeError):
        theirs.add_block(ramp(pp, opts, 0, HELD), pp.make_delay(1e-3))


def test_gradients_that_carry_on_from_one_another_are_not_reported():
    opts = system()
    theirs = pp.Sequence(system=opts)
    theirs.add_block(ramp(pp, opts, 0, HELD))
    theirs.add_block(ramp(pp, opts, HELD, 0))

    assert as_ours(theirs).check_timing() == (True, [])
    assert theirs.check_timing()[0]


# -- how long the sequence lasts ---------------------------------------


@pytest.fixture
def playable():
    """A sequence with nothing wrong with it, built on the core."""
    opts = system()
    seq = pypulseqpp.Sequence(opts)
    seq.add_block(ramp(pypulseqpp, opts, 0, HELD))
    seq.add_block(ramp(pypulseqpp, opts, HELD, 0))
    return seq


def test_the_first_check_records_how_long_the_sequence_lasts(playable):
    assert playable.get_definition("TotalDuration") == ""
    assert playable.check_timing()[0]

    recorded = playable.get_definition("TotalDuration")

    assert float(recorded[0]) == pytest.approx(playable.duration()[0])
    assert playable.check_timing()[0]


def test_a_recorded_duration_that_is_not_the_blocks_is_reported(playable):
    playable.check_timing()
    playable.set_definition("TotalDuration", 1.0)

    is_ok, found = playable.check_timing()

    assert not is_ok
    assert kinds(found) == {"TOTAL_DURATION_MISMATCH"}
    assert found[-1].value == pytest.approx(1.0)
    assert found[-1].duration == pytest.approx(playable.duration()[0])


def test_a_sequence_that_changed_has_its_duration_recorded_again(playable):
    playable.check_timing()

    playable.add_block(pypulseqpp.make_delay(1e-3))

    assert playable.check_timing()[0]
    assert float(playable.get_definition("TotalDuration")[0]) == pytest.approx(
        playable.duration()[0]
    )


def test_a_file_that_declares_a_duration_it_does_not_have_is_reported(
    playable, tmp_path
):
    playable.check_timing()
    playable.set_definition("TotalDuration", 1.0)
    playable.write(tmp_path / "wrong.seq", create_signature=False)

    read_back = pypulseqpp.Sequence(playable.system)
    read_back.read(tmp_path / "wrong.seq")

    assert kinds(read_back.check_timing()[1]) == {"TOTAL_DURATION_MISMATCH"}


# -- how far off resonance a pulse may sit -----------------------------


def _off_resonance(**offsets):
    opts = system()
    opts.max_freq_offset = 5000.0
    seq = pypulseqpp.Sequence(opts)
    seq.add_block(
        pypulseqpp.make_block_pulse(
            0.5,
            duration=1e-3,
            system=opts,
            use="excitation",
            freq_offset=offsets.get("rf_hz", 0.0),
            freq_ppm=offsets.get("rf_ppm", 0.0),
        )
    )
    seq.add_block(
        pypulseqpp.make_adc(
            64,
            dwell=1e-5,
            delay=10e-6,
            system=opts,
            freq_offset=offsets.get("adc_hz", 0.0),
            freq_ppm=offsets.get("adc_ppm", 0.0),
        ),
        pypulseqpp.make_delay(1e-3),
    )
    return seq


def test_a_pulse_within_the_frequency_limit_is_not_reported():
    assert _off_resonance(rf_hz=4000.0).check_timing() == (True, [])


def test_a_pulse_too_far_off_resonance_is_reported():
    is_ok, found = _off_resonance(rf_hz=12000.0).check_timing()

    assert not is_ok
    assert [(f.error_type, f.event, f.block) for f in found] == [
        ("FREQ_OFFSET", "rf", 1)
    ]
    assert found[0].limit == 5000.0
    assert found[0].offset == pytest.approx(12000.0)


def test_a_window_too_far_off_resonance_is_reported():
    is_ok, found = _off_resonance(adc_hz=-9000.0).check_timing()

    assert not is_ok
    assert [(f.error_type, f.event, f.block) for f in found] == [
        ("FREQ_OFFSET", "adc", 2)
    ]


def test_a_ppm_shift_counts_against_the_same_limit():
    """A shift in parts per million is a shift of the Larmor frequency."""
    opts = system()
    larmor = opts.gamma * opts.B0
    # A hundred ppm of 63.9 MHz is about 6.4 kHz.
    is_ok, found = _off_resonance(rf_ppm=100.0).check_timing()

    assert not is_ok
    assert found[0].offset == pytest.approx(100.0e-6 * larmor)


def test_two_offsets_within_the_limit_can_be_over_it_together():
    """Either alone passes; what the pulse is actually asked for does not."""
    assert _off_resonance(rf_hz=3000.0).check_timing()[0]
    assert _off_resonance(rf_ppm=40.0).check_timing()[0]

    is_ok, found = _off_resonance(rf_hz=3000.0, rf_ppm=40.0).check_timing()

    assert not is_ok
    assert found[0].offset > 5000.0


def test_a_scanner_that_names_no_limit_asks_nothing():
    """Upstream's Opts carries no such field, so a sequence built against one
    is not judged on it."""
    opts = system()
    opts.max_freq_offset = 0.0
    seq = pypulseqpp.Sequence(opts)
    seq.add_block(
        pypulseqpp.make_block_pulse(
            0.5, duration=1e-3, system=opts, use="excitation", freq_offset=1e6
        )
    )

    assert seq.check_timing() == (True, [])
