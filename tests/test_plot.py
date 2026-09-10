"""What `Sequence.plot` hands SeqEyes, and which blocks a range names.

SeqEyes is a separate program and an optional one, so most of what is held
here is the file written for it: the blocks asked for, played at the times
the scan plays them, with the labels the scan has reached. The test that
opens it skips when it is not installed.
"""

import math
import subprocess
import sys
import types

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import _plot


@pytest.fixture
def gradient_echo():
    """One shot is three blocks -- excite, encode, read -- after a prologue."""

    def make(lines=8, prologue=2):
        sequence = pp.Sequence(pp.Opts())
        pulse, select, rephase = pp.make_sinc_pulse(
            flip_angle=math.pi / 12,
            duration=1e-3,
            slice_thickness=3e-3,
            use="excitation",
            return_gz=True,
        )
        read = pp.make_trapezoid("x", area=1000, duration=2e-3)
        window = pp.make_adc(num_samples=64, duration=2e-3)
        encode = pp.make_trapezoid("y", area=500, duration=1e-3)
        step = pp.make_label(label="LIN", type="INC", value=1)

        for _ in range(prologue):
            sequence.add_block(pp.make_delay(5e-3))
        for line in range(lines):
            sequence.add_block(pulse, select)
            sequence.add_block(pp.scale_grad(encode, (line / lines) or 1e-9), rephase)
            sequence.add_block(read, window, step)
        return sequence

    return make


def _read(text, tmp_path):
    path = tmp_path / "excerpt.seq"
    path.write_bytes(text)
    sequence = pp.Sequence(pp.Opts())
    sequence.read(str(path))
    return sequence


# -- which blocks a range names ---------------------------------------------


def test_a_repetition_is_counted_from_the_first_full_one(gradient_echo):
    sequence = gradient_echo(prologue=2)

    assert sequence._detect_tr() == (3, 3)
    assert _plot.blocks_for(sequence, tr_range=(2, 3)) == (6, 11)


def test_an_open_repetition_range_runs_to_the_last_repetition(gradient_echo):
    sequence = gradient_echo(lines=8, prologue=2)

    assert _plot.blocks_for(sequence, tr_range=(7, math.inf)) == (21, 26)


def test_a_block_range_is_taken_as_given(gradient_echo):
    sequence = gradient_echo()

    assert _plot.blocks_for(sequence, block_range=(4, 9)) == (4, 9)
    assert _plot.blocks_for(sequence, block_range=(4, math.inf)) == (4, 26)


def test_a_time_range_names_every_block_it_touches(gradient_echo):
    sequence = gradient_echo(prologue=2)
    ends = np.cumsum(sequence._native.block_durations())

    # From inside the second delay to just past the start of the first shot.
    touched = _plot.blocks_for(sequence, time_range=(ends[0] + 1e-4, ends[1] + 1e-5))

    assert touched == (2, 3)


def test_no_range_is_the_whole_sequence(gradient_echo):
    sequence = gradient_echo()

    assert _plot.blocks_for(sequence) == (1, sequence.num_blocks)


def test_ranges_are_given_one_at_a_time(gradient_echo):
    with pytest.raises(ValueError, match="not several"):
        _plot.blocks_for(gradient_echo(), block_range=(1, 2), tr_range=(1, 1))


@pytest.mark.parametrize(
    "asked",
    [
        {"block_range": (1, 1000)},
        {"block_range": (0, 3)},
        {"tr_range": (9, 9)},
        {"time_range": (10.0, 11.0)},
        {"block_range": (5, 4)},
    ],
)
def test_a_range_outside_the_sequence_is_refused(gradient_echo, asked):
    with pytest.raises(ValueError):
        _plot.blocks_for(gradient_echo(), **asked)


def test_a_sequence_that_does_not_repeat_has_no_repetition_to_count():
    sequence = pp.Sequence(pp.Opts())
    for k in range(4):
        sequence.add_block(pp.make_trapezoid("x", area=100, duration=1e-3 * (k + 1)))

    with pytest.raises(ValueError, match="does not repeat"):
        _plot.blocks_for(sequence, tr_range=(1, 1))


# -- what is written ----------------------------------------------------------


def test_an_excerpt_plays_its_blocks_when_the_scan_plays_them(gradient_echo, tmp_path):
    sequence = gradient_echo(prologue=2)
    before = sequence.num_blocks

    excerpt = _read(_plot._excerpt(sequence, 6, 11), tmp_path)

    assert sequence.num_blocks == before
    # A label block of no duration leads, then the six blocks asked for.
    assert excerpt.num_blocks == 1 + 6
    durations = np.asarray(sequence._native.block_durations())
    written = np.asarray(excerpt._native.block_durations())
    assert written[0] == 0
    np.testing.assert_allclose(written[1:], durations[5:11])

    # Held against the whole scan written the same way, since the text form
    # carries an amplitude to six significant digits. A block range is timed
    # from its first block in both.
    whole = _read(pp._ext.write_text(sequence._native, False), tmp_path)
    theirs = whole.waveforms(block_range=(6, 11))
    ours = excerpt.waveforms(block_range=(2, 7))
    for axis in range(3):
        np.testing.assert_array_equal(ours[axis], theirs[axis])


def test_an_excerpt_carries_the_labels_the_scan_has_reached(gradient_echo, tmp_path):
    sequence = gradient_echo(prologue=2)

    excerpt = _read(_plot._excerpt(sequence, 9, 14), tmp_path)

    assert excerpt.evaluate_labels() == sequence.evaluate_labels(block_range=(1, 14))


def test_an_excerpt_from_the_first_block_needs_no_lead(gradient_echo, tmp_path):
    sequence = gradient_echo(prologue=2)

    excerpt = _read(_plot._excerpt(sequence, 1, 5), tmp_path)

    np.testing.assert_allclose(
        excerpt._native.block_durations(), sequence._native.block_durations()[:5]
    )


def test_an_excerpt_carries_only_the_events_its_blocks_play(gradient_echo, tmp_path):
    """A phase encode is a row per line, so the scan's library grows with it."""
    sequence = gradient_echo(lines=64, prologue=2)
    size, start = sequence._detect_tr()

    excerpt = _read(
        _plot._excerpt(sequence, start + 10 * size, start + 11 * size - 1), tmp_path
    )

    assert sequence._native.num_gradients() > 64
    # Slice select, its rephaser, one phase encode, and the readout.
    assert excerpt._native.num_gradients() == 4


def test_a_block_the_sequence_does_not_have_is_not_written(gradient_echo):
    sequence = gradient_echo()

    with pytest.raises(IndexError, match="not one of the sequence's"):
        pp._ext.write_text(sequence._native, False, [1, sequence.num_blocks + 1])


# -- the call -----------------------------------------------------------------


class _Window:
    """Stands in for SeqEyes: a process that has already closed."""

    def __init__(self, arguments, *_, **__):
        self.arguments = arguments

    def poll(self):
        return 0

    def wait(self, *_):
        return 0


@pytest.fixture
def launched(monkeypatch):
    """What SeqEyes would have been started with, without starting it."""
    started = []

    def run(program, arguments, folder, **environment):
        window = _Window(arguments)
        started.append((arguments, environment))
        return window

    monkeypatch.setattr(_plot, "executable", lambda: _plot.Path("seqeyes"))
    monkeypatch.setattr(_plot, "_run", run)
    return started


def test_an_option_seqeyes_does_not_take_is_reported_not_refused(
    gradient_echo, launched
):
    with pytest.warns(UserWarning, match="does not take stacked, time_disp"):
        gradient_echo().plot(stacked=True, time_disp="ms")

    assert len(launched) == 1


def test_upstreams_defaults_ask_for_nothing_to_be_reported(gradient_echo, launched):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gradient_echo().plot()


def test_an_option_upstream_does_not_have_is_refused(gradient_echo):
    with pytest.raises(TypeError, match="colour"):
        _plot.plot(gradient_echo(), colour="red")


def test_repetitions_are_drawn_whole_and_titled_with_where_they_play(
    gradient_echo, launched
):
    sequence = gradient_echo(prologue=2)

    sequence.plot(tr_range=(2, 2))

    arguments, _ = launched[0]
    assert "--time-range" not in arguments
    title = arguments[arguments.index("--name") + 1]
    start = sequence._native.block_durations()[:5].sum()
    assert title == f"seq_plot: blocks 6 to 8 of 26, from {start:g} s"


def test_a_time_range_is_drawn_on_the_excerpts_clock_in_whole_milliseconds(
    gradient_echo, launched
):
    sequence = gradient_echo(prologue=2)
    ends = np.cumsum(sequence._native.block_durations())
    asked = (ends[5] + 2e-4, ends[7] - 3e-4)

    sequence.plot(time_range=asked)

    arguments, _ = launched[0]
    window = arguments[arguments.index("--time-range") + 1]
    low, high = (int(edge) for edge in window.split("~"))
    # The range starts inside block 7, so the excerpt starts there.
    assert low <= 1e3 * (asked[0] - ends[5]) < low + 1
    assert high - 1 < 1e3 * (asked[1] - ends[5]) <= high


def test_the_whole_sequence_is_drawn_without_a_range(gradient_echo, launched):
    gradient_echo().plot()

    arguments, _ = launched[0]
    assert "--time-range" not in arguments


def test_a_missing_viewer_says_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "pypulseqpp_seqeyes", None)
    monkeypatch.setattr(_plot.shutil, "which", lambda _name: None)

    with pytest.raises(ModuleNotFoundError, match=r"pypulseqpp\[plot\]"):
        _plot.executable()


def test_the_packaged_viewer_is_pointed_at_its_qt_and_no_other_is(
    monkeypatch, tmp_path
):
    ours = tmp_path / "bin" / "seqeyes"
    packaged = types.ModuleType("pypulseqpp_seqeyes")
    packaged.executable = lambda: ours
    packaged.environment = lambda base: {**base, "QT_PLUGIN_PATH": "pyside6/plugins"}
    monkeypatch.setitem(sys.modules, "pypulseqpp_seqeyes", packaged)

    env = _plot._environment(ours, QT_QPA_PLATFORM="offscreen")
    other = _plot._environment(tmp_path / "elsewhere" / "seqeyes")

    assert env["QT_PLUGIN_PATH"] == "pyside6/plugins"
    assert env["QT_QPA_PLATFORM"] == "offscreen"
    assert other.get("QT_PLUGIN_PATH") == _plot.os.environ.get("QT_PLUGIN_PATH")


def test_a_viewer_that_fails_says_what_seqeyes_printed(tmp_path):
    folder = tmp_path / "drawing"
    folder.mkdir()
    process = subprocess.Popen(
        [sys.executable, "-c", "import sys; print('no display'); sys.exit(3)"],
        stdout=(folder / "seqeyes.log").open("wb"),
        stderr=subprocess.STDOUT,
    )
    viewer = _plot.Viewer(process, folder)

    with pytest.raises(RuntimeError, match="status 3:\nno display"):
        viewer.wait()
    assert not folder.exists()


# -- SeqEyes itself -----------------------------------------------------------


def _seqeyes():
    try:
        return _plot.executable()
    except ModuleNotFoundError:
        return None


@pytest.mark.skipif(_seqeyes() is None, reason="SeqEyes is not installed")
def test_seqeyes_draws_the_repetitions_asked_for(gradient_echo, tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.chdir(tmp_path)

    viewer = gradient_echo(lines=64).plot(tr_range=(10, 12), save=True, plot_now=False)
    viewer.close()

    assert (tmp_path / "seq_plot_seq.png").stat().st_size > 0
    assert (tmp_path / "seq_plot_traj.png").stat().st_size > 0
    assert not viewer.path.exists()
