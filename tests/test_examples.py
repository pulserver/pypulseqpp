"""The sequence zoo, and the command line every entry is also reachable through.

What is held here is what a zoo entry promises: that it builds a legal
sequence, that the prescription it was asked for is the one in the file, and
that the module answers for its own ``main``.
"""

import subprocess
import sys
from itertools import pairwise

import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import cli, examples

SMALL = {"n_x": 32, "n_y": 16, "n_slices": 1, "n_acs": 0, "n_dummy": 0}


@pytest.mark.parametrize("name", examples.__all__)
def test_a_zoo_entry_is_callable_as_the_sequence_it_builds(name):
    """A script is its main, docstring and signature included."""
    script = getattr(examples, name)

    assert callable(script)
    assert script.__doc__ == script.main.__doc__
    assert "system" in script.__signature__.parameters


@pytest.mark.parametrize("name", examples.__all__)
def test_a_zoo_entry_builds_a_sequence_that_passes_its_timing_check(name):
    seq = getattr(examples, name)(**SMALL)

    assert isinstance(seq, pp.Sequence)
    is_ok, errors = seq.check_timing()
    assert is_ok, errors


# -- what the 2D gradient echo is ------------------------------------------


def gre(**kwargs):
    return examples.gre2D_sequence(**{**SMALL, **kwargs})


def test_every_line_is_one_repetition_of_the_same_blocks():
    seq = gre(n_y=16)

    assert len(seq.block_events) % 16 == 0


def test_the_prescription_asked_for_is_the_one_written_down():
    seq = gre(n_x=64, n_y=32, n_slices=3, slice_thickness=4e-3, fov=0.2)

    assert seq.definitions["Matrix"] == [64.0, 32.0, 3.0]
    assert seq.definitions["FOV"][:2] == [0.2, 0.2]
    assert seq.definitions["Name"] == "gre_2d"
    assert len(seq.definitions["SlicePositions"]) == 3


def test_the_slices_of_a_pass_are_not_neighbours():
    """A TR too short for every slice deals them into passes, spread out."""
    kernel = examples.gre2D_sequence.GREKernel(
        pp.Opts(), n_x=32, n_y=16, n_slices=8, tr=40e-3, n_acs=0, n_dummy=0
    )

    assert len(kernel.passes) > 1
    for group in kernel.passes:
        ordered = sorted(group)
        # Every slice of one pass is a whole pass-count away from the next, so
        # no two neighbours in the slab are excited in the same pass.
        assert all(b - a >= len(kernel.passes) for a, b in pairwise(ordered))


@pytest.mark.parametrize(
    ("n_x", "n_slices"),
    [(64, 120), (256, 120), (256, 30), (256, 7)],
    ids=["even passes", "odd pass", "even", "one pass"],
)
def test_the_scan_repeats_from_its_first_block_whatever_the_slices_divide_into(
    n_x, n_slices
):
    """A pass that holds one slice more is a longer wait, not a different shot.

    Slices are dealt round-robin, so their count need not divide evenly and
    one pass can hold one more than the rest. What squares them up is the
    pure delay that closes every shot -- and a pure delay is one definition
    however long it waits, so the block stream reads as one shot repeating
    rather than as two structurally different halves.
    """
    seq = gre(n_x=n_x, n_y=32, n_slices=n_slices, n_acs=8)

    _size, start = seq._detect_tr()

    assert start == 1


def test_every_slice_is_excited_at_the_repetition_time_asked_for():
    """Including the odd pass, which holds a slice more and waits less."""
    lines, tr = 8, 0.25
    seq = gre(n_x=256, n_y=lines, n_slices=120, tr=tr)
    kernel = examples.gre2D_sequence.GREKernel(
        pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s"),
        n_x=256,
        n_y=lines,
        n_slices=120,
        tr=tr,
        n_acs=0,
        n_dummy=0,
    )
    excited = np.asarray(seq.rf_times()[0])

    assert len({len(group) for group in kernel.passes}) == 2  # the case worth asking

    at = 0
    for group in kernel.passes:
        # A slice's repetition time is the gap between its own excitations.
        spacing = np.diff(excited[at : at + len(group) * lines][:: len(group)])
        assert spacing == pytest.approx(tr, abs=1e-9)
        at += len(group) * lines


def test_a_shorter_echo_than_the_readout_admits_is_refused():
    with pytest.raises(ValueError):
        gre(te=1e-6)


def test_undersampling_acquires_fewer_lines_than_it_encodes():
    full = gre(n_y=32, acceleration=1)
    half = gre(n_y=32, acceleration=2)

    assert len(half.block_events) < len(full.block_events)


def test_the_calibration_block_leads_the_scan():
    """A reconstruction calibrates while the rest of the scan is arriving."""
    kernel = examples.gre2D_sequence.GREKernel(
        pp.Opts(), n_x=32, n_y=32, acceleration=2, n_acs=8, n_dummy=0
    )

    assert kernel.n_calibration == 8
    assert list(kernel.sampled_lines[:8]) == sorted(kernel.sampled_lines[:8])


# -- the command line ------------------------------------------------------


def test_the_command_line_writes_what_the_call_builds(tmp_path):
    path = tmp_path / "gre.seq"

    status = cli.run(
        examples.gre2D_sequence.main,
        [
            "-o",
            str(path),
            "--n-x",
            "32",
            "--n-y",
            "16",
            "--n-acs",
            "0",
            "--n-dummy",
            "0",
        ],
    )

    assert status == 0
    assert path.read_text().startswith("# Pulseq sequence file")


def test_a_flag_is_named_and_described_by_the_function_it_runs(capsys):
    with pytest.raises(SystemExit):
        cli.run(examples.gre2D_sequence.main, ["--help"])

    printed = capsys.readouterr().out

    assert "--flip-angle-deg" in printed
    assert "Excitation flip angle, in degrees." in printed


def test_the_binary_form_is_smaller_and_carries_no_signature(tmp_path):
    seq = gre()
    text, binary = tmp_path / "s.seq", tmp_path / "s.bin"

    signature = cli.write_sequence(seq, str(text))
    assert cli.write_sequence(seq, str(binary), offline=False) is None

    assert len(signature) == 32
    assert binary.stat().st_size < text.stat().st_size
    assert "[SIGNATURE]" in text.read_text()


def test_a_written_sequence_reads_back_as_itself(tmp_path):
    path = tmp_path / "gre.seq"
    cli.write_sequence(gre(), str(path))

    read_back = pp.Sequence()
    read_back.read(str(path))
    again = tmp_path / "again.seq"
    read_back.write(str(again))

    assert again.read_text() == path.read_text()


def test_running_the_module_as_a_script_writes_a_sequence(tmp_path):
    path = tmp_path / "gre.seq"

    # The interpreter running these tests, and a path pytest made.
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "pypulseqpp.examples.sequence.gre2D_sequence",
            "-o",
            str(path),
            "--n-x",
            "32",
            "--n-y",
            "16",
            "--n-acs",
            "0",
            "--n-dummy",
            "0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert path.exists()
