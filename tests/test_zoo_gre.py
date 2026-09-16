"""Gradient-echo example sequences: 3D, and multi-echo 2D and 3D."""

import importlib
from itertools import pairwise

import numpy as np
import pytest

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

APPS = {
    "gre3D_sequence": "Gre3DApp",
    "gre_multiecho2D_sequence": "GreMultiecho2DApp",
    "gre_multiecho3D_sequence": "GreMultiecho3DApp",
}

MULTIECHO = ["gre_multiecho2D_sequence", "gre_multiecho3D_sequence"]


def module(name):
    return importlib.import_module(f"pypulseqpp.sequences.sequence.{name}")


def app(name, **kwargs):
    """The application ``name``, without dummies unless asked for."""
    cls = getattr(module(name), APPS[name])
    return cls(pp.Opts(), **{**SMALL[name], "n_dummy": 0, **kwargs})


def gre3d(**kwargs):
    return app("gre3D_sequence", **kwargs)


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
    a = gre3d(n_dummy=3, ry=2, rz=2, n_acs_y=4, n_acs_z=2)
    lin, par, ima, seg = adc_labels(a.design(), "LIN", "PAR", "IMA", "SEG")
    calibrating = [int(view in a.calibration) for view in a.views]

    assert a.calibration
    assert a.views[: len(a.calibration)] == sorted(a.calibration)
    assert list(zip(lin, par, strict=True)) == a.views
    assert list(ima) == calibrating
    assert list(seg) == [1 - c for c in calibrating]


@pytest.mark.parametrize("n_z", [8, 9])
@pytest.mark.parametrize(
    ("ry", "rz", "shift"), [(2, 1, 0), (1, 3, 0), (2, 3, 1), (3, 2, 1)]
)
def test_the_caipirinha_lattice_always_holds_the_centre_and_climbs_per_line(
    ry, rz, shift, n_z
):
    a = gre3d(n_y=16, n_z=n_z, ry=ry, rz=rz, caipi_shift=shift, n_acs_y=0, n_acs_z=0)
    n_y = a.matrix[1]

    assert (n_y // 2, n_z // 2) in a.views
    assert all((y - n_y // 2) % ry == 0 for y, _ in a.views)
    assert all(
        (z - n_z // 2 - shift * ((y - n_y // 2) // ry)) % rz == 0 for y, z in a.views
    )


def test_partial_fourier_drops_the_lines_and_partitions_before_the_centre():
    a = gre3d(n_y=16, n_z=8, partial_fourier_y=0.75, partial_fourier_z=0.75)

    assert a.views == [(y, z) for y in range(4, 16) for z in range(2, 8)]


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
    seq = gre3d(excitation=excitation).design()

    assert seq.check_timing()[0]
    assert seq.definitions["Excitation"] == excitation


def test_every_partition_is_encoded_at_the_step_its_label_names():
    a = gre3d()
    seq = a.design()
    (par,) = adc_labels(seq, "PAR")
    n_z = a.matrix[2]

    blocks = [seq.get_block(i) for i in range(1, len(seq.block_events) + 1)]
    steps = [
        blocks[i - 1].gz.amplitude / a.ro.gz_pre.amplitude
        for i, block in enumerate(blocks)
        if block.adc is not None
    ]

    assert sorted(set(par)) == list(range(n_z))
    assert steps == pytest.approx((par - n_z // 2) / (n_z / 2))


# -- multi-echo: every view read at each echo time ----------------------------


def test_each_2d_acquisition_carries_the_line_slice_and_echo_it_reads():
    a = app("gre_multiecho2D_sequence", n_dummy=2, n_slices=3, ry=2, n_acs_y=4)
    lin, slc, eco, ima = adc_labels(a.design(), "LIN", "SLC", "ECO", "IMA")

    expected = [
        (line, s, echo)
        for packet in a.packets
        for line in a.lines
        for s in packet
        for echo in range(a.n_echoes)
    ]
    assert list(zip(lin, slc, eco, strict=True)) == expected
    assert list(ima) == [int(line in a.calibration) for line, _, _ in expected]


def test_each_3d_acquisition_carries_the_view_and_echo_it_reads():
    a = app("gre_multiecho3D_sequence", n_dummy=3, ry=2, rz=2, n_acs_y=4, n_acs_z=2)
    lin, par, eco, ima = adc_labels(a.design(), "LIN", "PAR", "ECO", "IMA")

    expected = [(*view, echo) for view in a.views for echo in range(a.n_echoes)]
    assert a.calibration
    assert list(zip(lin, par, eco, strict=True)) == expected
    assert list(ima) == [int((y, z) in a.calibration) for y, z, _ in expected]


@pytest.mark.parametrize("name", MULTIECHO)
@pytest.mark.parametrize("flyback", [True, False], ids=["monopolar", "bipolar"])
@pytest.mark.parametrize("echo_spacing", [None, 4e-3], ids=["shortest", "spaced"])
def test_the_echo_times_written_are_the_first_plus_whole_spacings(
    name, flyback, echo_spacing
):
    a = app(name, n_echoes=4, flyback=flyback, echo_spacing=echo_spacing)
    written = np.atleast_1d(a.design().definitions["TE"])
    spacing = a.ro.echo_spacing

    assert written == pytest.approx(a.ro.echo_time + spacing * np.arange(4))
    if echo_spacing is not None:
        assert spacing == pytest.approx(echo_spacing)


@pytest.mark.parametrize("name", MULTIECHO)
@pytest.mark.parametrize("flyback", [True, False], ids=["monopolar", "bipolar"])
def test_every_echo_crosses_k_zero_at_the_echo_time_written(name, flyback):
    a = app(name, n_dummy=0, n_echoes=3, flyback=flyback, echo_spacing=3e-3)
    seq = a.design()
    k, _, t_excitation, _, t_adc = seq.calculate_kspace()
    n = int(a.ro.adc.num_samples)
    kx, t_adc = k[0, : 3 * n].reshape(3, n), np.asarray(t_adc)[: 3 * n].reshape(3, n)
    written = np.atleast_1d(seq.definitions["TE"])

    for echo in range(3):
        # The crossing lies between two sample centres, half a dwell from the
        # nearer one.
        nearest = int(np.argmin(np.abs(kx[echo])))
        assert t_adc[echo, nearest] - t_excitation[0] == pytest.approx(
            written[echo], abs=0.5 * float(a.ro.adc.dwell) + 1e-9
        )


@pytest.mark.parametrize("name", MULTIECHO)
def test_a_bipolar_train_reads_alternate_echoes_backwards(name):
    a = app(name, n_echoes=4, flyback=False)
    signs = [np.sign(b.gx.amplitude) for b in acquisitions(a.design())]

    assert signs[:4] == [1, -1, 1, -1]
    assert len(set(map(tuple, np.reshape(signs, (-1, 4))))) == 1


@pytest.mark.parametrize("name", MULTIECHO)
def test_a_bipolar_train_spaces_its_echoes_closer_than_a_monopolar_one(name):
    mono = app(name, n_echoes=3)
    bi = app(name, n_echoes=3, flyback=False)

    assert bi.ro.echo_spacing < mono.ro.echo_spacing


@pytest.mark.parametrize("name", MULTIECHO)
@pytest.mark.parametrize("flyback", [True, False], ids=["monopolar", "bipolar"])
def test_a_longer_echo_spacing_waits_after_every_echo_but_the_last(name, flyback):
    a = app(name, n_echoes=3, flyback=flyback, echo_spacing=5e-3)
    seq = a.design()
    blocks = [seq.get_block(i) for i in range(1, len(seq.block_events) + 1)]
    reads = [i for i, block in enumerate(blocks) if block.adc is not None][:3]

    for first, second in pairwise(reads):
        between = blocks[first + 1 : second]
        # A monopolar train rewinds before it waits.
        assert len(between) == (2 if flyback else 1)
        assert between[-1].gx is None and between[-1].adc is None
        if flyback:
            assert between[0].gx.amplitude == pytest.approx(a.ro.gx_flyback.amplitude)


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
    seq = app(name, n_dummy=2, flyback=flyback, echo_spacing=4e-3, ry=2).design()

    _size, start = seq._detect_tr()

    assert start == 1


def test_every_slice_is_excited_at_the_repetition_time_written():
    """Including the smaller packet, which waits longer."""
    tr, lines = 30e-3, 16
    a = app("gre_multiecho2D_sequence", n_slices=7, tr=tr)
    seq = a.design()
    excited = np.asarray(seq.rf_times()[0])

    assert len({len(packet) for packet in a.packets}) == 2
    assert np.atleast_1d(seq.definitions["TR"])[0] == pytest.approx(tr)
    at = 0
    for packet in a.packets:
        spacing = np.diff(excited[at : at + len(packet) * lines][:: len(packet)])
        assert spacing == pytest.approx(tr, abs=1e-9)
        at += len(packet) * lines
