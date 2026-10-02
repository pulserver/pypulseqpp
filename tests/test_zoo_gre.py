"""Gradient-echo example sequences: 3D, and multi-echo 2D and 3D."""

import importlib
from itertools import pairwise

import numpy as np
import pytest
from zoo import packets

import pypulseqpp as pp
from pypulseqpp import cli

#: A prescription small enough to build in a moment, per example sequence.
SMALL = {
    "gre3D_sequence": {"n_x": 32, "n_y": 16, "n_z": 8, "n_acs_y": 0, "n_acs_z": 0},
    "gre_multiecho2D_sequence": {"n_x": 32, "n_y": 16, "n_echoes": 3, "n_acs_y": 0},
    "gre_multiecho3D_sequence": {
        "n_x": 32,
        "n_y": 16,
        "n_z": 8,
        "n_echoes": 3,
        "n_acs_y": 0,
        "n_acs_z": 0,
    },
}

#: The definition that records each prescribed time.
RECORDED = {"te": "TE", "tr": "TR"}

MULTIECHO = ["gre_multiecho2D_sequence", "gre_multiecho3D_sequence"]


def module(name):
    return importlib.import_module(f"pypulseqpp.sequences.sequence.{name}")


def built(name, **kwargs):
    """The sequence ``name`` designs, without dummies unless asked for."""
    return module(name).main(**{**SMALL[name], "n_dummy": 0, **kwargs})


def gre3d(**kwargs):
    """The 3D gradient echo, without dummies unless asked for."""
    return built("gre3D_sequence", **kwargs)


def adc_labels(seq, *names):
    """Each label's value at every acquisition, as an int array per name."""
    found = seq.evaluate_labels(evolution="adc")
    return [np.atleast_1d(found.get(name, 0)) for name in names]


def acquisitions(seq):
    """The blocks that acquire, in play order."""
    blocks = (seq.get_block(i) for i in range(1, len(seq.block_events) + 1))
    return [block for block in blocks if block.adc is not None]


@pytest.mark.parametrize("name", SMALL)
def test_a_small_prescription_passes_its_timing_check(name):
    seq = module(name).main(**SMALL[name])

    is_ok, errors = seq.check_timing()
    assert is_ok, errors


@pytest.mark.parametrize("name", SMALL)
@pytest.mark.parametrize("prescription", [{"te": 1e-6}, {"tr": 1e-3}], ids=str)
def test_an_echo_or_repetition_shorter_than_the_readout_is_refused(name, prescription):
    with pytest.raises(ValueError, match="shorter than"):
        module(name).main(**{**SMALL[name], **prescription})


@pytest.mark.parametrize("name", SMALL)
@pytest.mark.parametrize("parameter", ["te", "tr"])
def test_an_echo_or_repetition_left_to_the_design_is_written_as_the_shortest_one(
    name, parameter
):
    def written(**kwargs):
        return np.atleast_1d(built(name, **kwargs).definitions[RECORDED[parameter]])[0]

    shortest = written(**{parameter: None})
    raster = pp.Opts().block_duration_raster

    assert written(**{parameter: shortest}) == pytest.approx(shortest)
    with pytest.raises(ValueError, match="shorter than"):
        built(name, **{parameter: shortest - raster})


@pytest.mark.parametrize(
    ("name", "flag", "help_text"),
    [
        ("gre3D_sequence", "--n-z", "Matrix size along the readout"),
        ("gre_multiecho2D_sequence", "--n-echoes", "Echoes per excitation."),
        ("gre_multiecho3D_sequence", "--echo-spacing", "Echo spacing (s)."),
    ],
)
def test_a_flag_is_named_and_described_by_the_function_it_runs(
    capsys, name, flag, help_text
):
    with pytest.raises(SystemExit):
        cli.run(module(name).main, ["--help"])

    printed = capsys.readouterr().out

    assert flag in printed
    assert help_text in printed


# -- 3D: one (line, partition) pair per excitation ----------------------------


def test_every_sampled_view_is_acquired_in_order_with_its_calibration_flags():
    seq = gre3d(n_dummy=3, ry=2, rz=2, n_acs_y=4, n_acs_z=2)
    lin, par, ima, seg = adc_labels(seq, "LIN", "PAR", "IMA", "SEG")
    shape = (SMALL["gre3D_sequence"]["n_y"], SMALL["gre3D_sequence"]["n_z"])
    calibration, imaging = pp.make_cartesian_plane_sampling(
        shape, (2, 2), (4, 2), elliptical=True
    )
    views = [*calibration, *imaging]
    marked = [int(view in calibration) for view in views]

    assert calibration
    assert views[: len(calibration)] == sorted(calibration)
    assert list(zip(lin, par, strict=True)) == views
    assert list(ima) == marked
    assert list(seg) == [1 - mark for mark in marked]


@pytest.mark.parametrize("n_z", [8, 9])
@pytest.mark.parametrize(
    ("ry", "rz", "shift"), [(2, 1, 0), (1, 3, 0), (2, 3, 1), (3, 2, 1)]
)
def test_the_caipirinha_lattice_always_holds_the_centre_and_climbs_per_line(
    ry, rz, shift, n_z
):
    n_y = 16
    seq = gre3d(n_y=n_y, n_z=n_z, ry=ry, rz=rz, caipi_shift=shift, n_acs_y=0, n_acs_z=0)
    lin, par = adc_labels(seq, "LIN", "PAR")
    views = list(zip(lin.tolist(), par.tolist(), strict=True))

    assert (n_y // 2, n_z // 2) in views
    assert all((y - n_y // 2) % ry == 0 for y, _ in views)
    assert all(
        (z - n_z // 2 - shift * ((y - n_y // 2) // ry)) % rz == 0 for y, z in views
    )


def test_partial_fourier_drops_the_lines_and_partitions_before_the_centre():
    seq = gre3d(
        n_y=16,
        n_z=8,
        partial_fourier_y=0.75,
        partial_fourier_z=0.75,
        elliptical_sampling=False,
    )
    lin, par = adc_labels(seq, "LIN", "PAR")

    assert list(zip(lin, par, strict=True)) == [
        (y, z) for y in range(4, 16) for z in range(2, 8)
    ]


@pytest.mark.parametrize(
    "prescription",
    [
        {"caipi_shift": 2, "rz": 2},
        {"excitation": "adiabatic"},
        {"partial_fourier_z": 0.7},
        {"ry": 0},
    ],
    ids=str,
)
def test_an_out_of_range_3d_prescription_is_refused(prescription):
    with pytest.raises(ValueError):
        gre3d(**prescription)


@pytest.mark.parametrize("excitation", ["slab", "nonselective", "spsp"])
def test_every_excitation_builds_a_3d_gradient_echo_that_passes_its_timing_check(
    excitation,
):
    seq = gre3d(excitation=excitation)

    assert seq.check_timing()[0]
    assert seq.definitions["Excitation"] == excitation


def test_every_partition_is_encoded_at_the_step_its_label_names():
    """The pre-phasing area before a read is the label's offset from the centre, over the FOV."""
    n_z, fov_z = SMALL["gre3D_sequence"]["n_z"], 0.1
    seq = gre3d(fov_z=fov_z)
    (par,) = adc_labels(seq, "PAR")

    blocks = [seq.get_block(i) for i in range(1, len(seq.block_events) + 1)]
    areas = [
        blocks[i - 1].gz.area for i, block in enumerate(blocks) if block.adc is not None
    ]

    assert sorted(set(par)) == list(range(n_z))
    assert areas == pytest.approx((par - n_z // 2) / fov_z)


# -- multi-echo: every view read at each echo time ----------------------------


def test_each_2d_acquisition_carries_the_line_slice_and_echo_it_reads():
    seq = built("gre_multiecho2D_sequence", n_dummy=2, n_slices=3, ry=2, n_acs_y=4)
    lin, slc, eco, ima = adc_labels(seq, "LIN", "SLC", "ECO", "IMA")
    calibrating, imaging = pp.make_cartesian_axis_sampling(
        16, 2, 4, partial_fourier=1.0
    )
    # One packet, whose even slices are excited before its odd ones.
    expected = [
        (line, s, echo)
        for line in [*calibrating, *imaging]
        for s in (0, 2, 1)
        for echo in range(3)
    ]

    assert list(zip(lin, slc, eco, strict=True)) == expected
    assert list(ima) == [int(line in calibrating) for line, _, _ in expected]


def test_each_3d_acquisition_carries_the_view_and_echo_it_reads():
    name = "gre_multiecho3D_sequence"
    seq = built(name, n_dummy=3, ry=2, rz=2, n_acs_y=4, n_acs_z=2)
    lin, par, eco, ima = adc_labels(seq, "LIN", "PAR", "ECO", "IMA")
    shape = (SMALL[name]["n_y"], SMALL[name]["n_z"])
    calibration, imaging = pp.make_cartesian_plane_sampling(
        shape, (2, 2), (4, 2), elliptical=True
    )

    expected = [
        (*view, echo)
        for view in [*calibration, *imaging]
        for echo in range(SMALL[name]["n_echoes"])
    ]
    assert calibration
    assert list(zip(lin, par, eco, strict=True)) == expected
    assert list(ima) == [int((y, z) in calibration) for y, z, _ in expected]


@pytest.mark.parametrize("name", MULTIECHO)
@pytest.mark.parametrize("flyback", [True, False], ids=["monopolar", "bipolar"])
@pytest.mark.parametrize("echo_spacing", [None, 4e-3], ids=["shortest", "spaced"])
def test_the_echo_times_written_are_the_first_plus_whole_spacings(
    name, flyback, echo_spacing
):
    seq = built(name, n_echoes=4, flyback=flyback, echo_spacing=echo_spacing)
    written = np.atleast_1d(seq.definitions["TE"])
    spacing = written[1] - written[0]

    assert written == pytest.approx(written[0] + spacing * np.arange(4))
    if echo_spacing is not None:
        assert spacing == pytest.approx(echo_spacing)


@pytest.mark.parametrize("name", MULTIECHO)
@pytest.mark.parametrize("flyback", [True, False], ids=["monopolar", "bipolar"])
def test_every_echo_crosses_k_zero_at_the_echo_time_written(name, flyback):
    seq = built(name, n_dummy=0, n_echoes=3, flyback=flyback, echo_spacing=3e-3)
    adc = acquisitions(seq)[0].adc
    k, _, t_excitation, _, t_adc = seq.calculate_kspace()
    n = int(adc.num_samples)
    kx, t_adc = k[0, : 3 * n].reshape(3, n), np.asarray(t_adc)[: 3 * n].reshape(3, n)
    written = np.atleast_1d(seq.definitions["TE"])

    for echo in range(3):
        # The crossing lies between two sample centres, half a dwell from the
        # nearer one.
        nearest = int(np.argmin(np.abs(kx[echo])))
        assert t_adc[echo, nearest] - t_excitation[0] == pytest.approx(
            written[echo], abs=0.5 * float(adc.dwell) + 1e-9
        )


@pytest.mark.parametrize("name", MULTIECHO)
def test_a_bipolar_train_reads_alternate_echoes_backwards(name):
    seq = built(name, n_echoes=4, flyback=False)
    signs = [np.sign(b.gx.amplitude) for b in acquisitions(seq)]

    assert signs[:4] == [1, -1, 1, -1]
    assert len(set(map(tuple, np.reshape(signs, (-1, 4))))) == 1


@pytest.mark.parametrize("name", MULTIECHO)
def test_a_bipolar_train_spaces_its_echoes_closer_than_a_monopolar_one(name):
    mono = np.diff(built(name, n_echoes=3).definitions["TE"])
    bi = np.diff(built(name, n_echoes=3, flyback=False).definitions["TE"])

    assert bi[0] < mono[0]


@pytest.mark.parametrize("name", MULTIECHO)
@pytest.mark.parametrize("flyback", [True, False], ids=["monopolar", "bipolar"])
def test_a_longer_echo_spacing_waits_after_every_echo_but_the_last(name, flyback):
    seq = built(name, n_echoes=3, flyback=flyback, echo_spacing=5e-3)
    blocks = [seq.get_block(i) for i in range(1, len(seq.block_events) + 1)]
    reads = [i for i, block in enumerate(blocks) if block.adc is not None][:3]

    for first, second in pairwise(reads):
        between = blocks[first + 1 : second]
        # A monopolar train rewinds before it waits.
        assert len(between) == (2 if flyback else 1)
        assert between[-1].gx is None and between[-1].adc is None
        if flyback:
            # The flyback rewinds the area of the lobe just read.
            assert between[0].gx.area == pytest.approx(-blocks[first].gx.area)


@pytest.mark.parametrize("name", MULTIECHO)
def test_an_echo_spacing_shorter_than_the_readout_is_refused(name):
    with pytest.raises(ValueError, match=r"echo spacing .* shorter than"):
        module(name).main(**SMALL[name], echo_spacing=1e-4)


@pytest.mark.parametrize("name", MULTIECHO)
def test_a_bipolar_train_with_a_partial_echo_is_refused(name):
    with pytest.raises(ValueError, match="bipolar"):
        module(name).main(**SMALL[name], flyback=False, partial_fourier_x=0.8)


@pytest.mark.parametrize("name", MULTIECHO)
@pytest.mark.parametrize("flyback", [True, False], ids=["monopolar", "bipolar"])
def test_a_multiecho_scan_repeats_from_its_first_block(name, flyback):
    seq = built(name, n_dummy=2, flyback=flyback, echo_spacing=4e-3, ry=2)

    _size, start = seq.repetition()

    assert start == 1


def test_every_slice_is_excited_at_the_repetition_time_written():
    """Including the smaller packet, which waits longer."""
    tr, lines = 30e-3, 16
    seq = built("gre_multiecho2D_sequence", n_slices=7, tr=tr)
    dealt = packets(seq)
    excited = np.asarray(seq.rf_times()[0])

    assert len({len(packet) for packet in dealt}) == 2
    assert np.atleast_1d(seq.definitions["TR"])[0] == pytest.approx(tr)
    at = 0
    for packet in dealt:
        spacing = np.diff(excited[at : at + len(packet) * lines][:: len(packet)])
        assert spacing == pytest.approx(tr, abs=1e-9)
        at += len(packet) * lines
