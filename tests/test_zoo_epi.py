"""The 2D and 3D EPI example sequences: linked prescans, sampling, timing and labels."""

from itertools import groupby
from pathlib import Path

import numpy as np
import pytest
from zoo import chain

import pypulseqpp as pp
from pypulseqpp import cli, sequences

epi2d = sequences.epi2D_sequence
epi3d = sequences.epi3D_sequence

SMALL_2D = {"n_x": 32, "n_y": 16, "n_dummy": 0}
SMALL_3D = {"n_x": 32, "n_y": 16, "n_z": 8, "n_dummy": 0}


def written(seq, key):
    """The first value of the numeric definition ``key``."""
    return float(np.atleast_1d(seq.definitions[key])[0])


class Scan:
    """The chain a prescription designs, and what the tests read back from it.

    Everything but the prescription comes from the designed sequences: the
    definitions of the main sequence, its labels and its first acquisition.
    """

    def __init__(self, module, **prescription):
        self.module = module
        small = SMALL_2D if module is epi2d else SMALL_3D
        self.prescription = {**small, **prescription}
        self.chain = chain(module(**self.prescription))
        self.main = self.chain[-1]

    def prescan(self, name):
        """The ``calibration`` or the ``reference`` the chain plays before the scan."""
        stem = self.main.definitions["Name"]
        (found,) = (
            seq
            for seq in self.chain[:-1]
            if seq.definitions["Name"] == f"{stem}_{name}"
        )
        return found

    @property
    def system(self):
        return self.main.system

    @property
    def matrix(self):
        return [int(size) for size in self.main.definitions["Matrix"]]

    @property
    def fov(self):
        return list(self.main.definitions["FOV"])

    @property
    def etl(self):
        """The lines a shot reads after its navigator."""
        return int(written(self.main, "EPIFactor"))

    @property
    def esp(self):
        return written(self.main, "EchoSpacing")

    @property
    def echo_time(self):
        return written(self.main, "TE")

    @property
    def repetition_time(self):
        return written(self.main, "TR")

    @property
    def n_shots(self):
        """The shots the phase encode of a volume is segmented into."""
        return self.prescription.get("n_shots", 1)

    @property
    def adc(self):
        """The ADC event of the first acquisition."""
        seq = self.main
        blocks = (seq.get_block(i) for i in range(1, len(seq.block_events) + 1))
        return next(block.adc for block in blocks if block.adc is not None)

    @property
    def samples(self):
        return int(self.adc.num_samples)

    @property
    def dwell(self):
        return float(self.adc.dwell)

    @property
    def shots_per_volume(self):
        """The shots that acquire one volume, counted from the lines it acquires."""
        (rep,) = labels(self.main, "REP")
        lines = int(np.count_nonzero(rep == 0))
        return lines // (self.module.NAVIGATOR_LINES + self.etl)


def scan2d(**kwargs):
    return Scan(epi2d, **kwargs)


def scan3d(**kwargs):
    return Scan(epi3d, **kwargs)


#: One prescription per case the shared tests run on.
CASES = {
    "2D": lambda **kw: scan2d(**kw),
    "2D segmented": lambda **kw: scan2d(n_y=32, ry=2, n_shots=3, **kw),
    "2D multiband": lambda **kw: scan2d(n_slices=4, multiband=2, n_shots=2, **kw),
    "3D": lambda **kw: scan3d(**kw),
    "3D skipped-CAIPI": lambda **kw: scan3d(ry=2, rz=2, n_shots=2, **kw),
}


def labels(seq, *names):
    """Each label's value at every acquisition, as an int array per name."""
    found = seq.evaluate_labels(evolution="adc")
    return [np.atleast_1d(found.get(name, 0)) for name in names]


def echoes(seq, n_samples):
    """``(ky, kz, echo time)`` of every acquired line, at its read-axis centre."""
    k, _, t_excitation, _, t_adc = seq.calculate_kspace()
    t_excitation = np.asarray(t_excitation)
    kx, t = k[0].reshape(-1, n_samples), np.asarray(t_adc).reshape(-1, n_samples)
    ky, kz = k[1].reshape(-1, n_samples), k[2].reshape(-1, n_samples)
    rows = []
    for r in range(len(kx)):
        i = int(np.argmin(np.abs(kx[r])))
        excited = t_excitation[t_excitation <= t[r, i]].max()
        rows.append((ky[r, i], kz[r, i], t[r, i] - excited))
    return np.array(rows)


def partition_fov(scan):
    """The field of view the partition (band) encode is counted over, and its size."""
    if scan.module is epi3d:
        return scan.fov[2], scan.matrix[2]
    step = written(scan.main, "SliceThickness") + written(scan.main, "SliceGap")
    return scan.matrix[2] * step, int(written(scan.main, "MultibandFactor"))


def played_slices(scan):
    """The ``SLC`` of each shot of a scan with one frame, in play order."""
    (slc,) = labels(scan.main, "SLC")
    return [s for s, _ in groupby(slc)]


def excitations(seq):
    blocks = (seq.get_block(i) for i in range(1, len(seq.block_events) + 1))
    return sum(block.rf is not None for block in blocks)


# -- the linked chain ------------------------------------------------------


@pytest.mark.parametrize(
    ("scan", "files"),
    [
        (
            lambda: scan2d(n_slices=2, ry=2, n_acs_y=4),
            [
                ("scan.seq", "epi_2d_calibration"),
                ("scan_epi_2d_reference.seq", "epi_2d_reference"),
                ("scan_epi_2d.seq", "epi_2d"),
            ],
        ),
        (
            lambda: scan2d(n_dummy=1),
            [("scan.seq", "epi_2d_reference"), ("scan_epi_2d.seq", "epi_2d")],
        ),
        (
            lambda: scan2d(n_slices=4, multiband=2, n_acs_y=4),
            [
                ("scan.seq", "epi_2d_calibration"),
                ("scan_epi_2d_reference.seq", "epi_2d_reference"),
                ("scan_epi_2d.seq", "epi_2d"),
            ],
        ),
        (
            lambda: scan3d(ry=2, n_acs_y=4, n_acs_z=2),
            [
                ("scan.seq", "epi_3d_calibration"),
                ("scan_epi_3d_reference.seq", "epi_3d_reference"),
                ("scan_epi_3d.seq", "epi_3d"),
            ],
        ),
        (
            lambda: scan3d(n_dummy=1),
            [("scan.seq", "epi_3d_reference"), ("scan_epi_3d.seq", "epi_3d")],
        ),
    ],
    ids=["2D undersampled", "2D", "2D multiband", "3D undersampled", "3D"],
)
def test_the_chain_is_written_in_play_order_each_file_naming_the_next(
    tmp_path, scan, files
):
    scan = scan()
    paths = sequences.write(tmp_path / "scan.seq", scan.chain)

    assert [Path(p).name for p in paths] == [name for name, _ in files]
    for i, path in enumerate(paths):
        seq = pp.Sequence(system=scan.system)
        seq.read(path)
        assert seq.definitions["Name"] == files[i][1]
        following = files[i + 1][0] if i + 1 < len(files) else None
        assert seq.definitions.get("NextSequence") == following
        is_ok, errors = seq.check_timing()
        assert is_ok, errors


@pytest.mark.parametrize("case", CASES)
def test_every_file_of_the_chain_repeats_from_its_first_block(tmp_path, case):
    scan = CASES[case](n_dummy=2, n_frames=2, volume_output=True, n_acs_y=4)
    for path in sequences.write(tmp_path / "scan.seq", scan.chain):
        seq = pp.Sequence(system=scan.system)
        seq.read(path)
        assert seq.repetition()[1] == 1, path


# -- what every shot samples ------------------------------------------------


@pytest.mark.parametrize("case", CASES)
def test_every_line_is_labelled_with_the_view_it_samples(case):
    scan = CASES[case]()
    rows = echoes(scan.main, scan.samples)
    lin, par, nav = labels(scan.main, "LIN", "PAR", "NAV")
    imaging = nav == 0
    n_y = scan.matrix[1]

    assert rows[imaging, 0] == pytest.approx((lin[imaging] - n_y // 2) / scan.fov[1])
    if case != "2D" and case != "2D segmented":
        fov, n = partition_fov(scan)
        assert rows[imaging, 1] == pytest.approx((par[imaging] - n // 2) / fov)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("prescan", [None, "reference"], ids=["main", "reference"])
def test_every_shot_opens_with_its_navigator_at_the_centre_of_k_space(case, prescan):
    scan = CASES[case]()
    seq = scan.main if prescan is None else scan.prescan(prescan)
    rows = echoes(seq, scan.samples)
    nav, rev = labels(seq, "NAV", "REV")
    navigator = scan.module.NAVIGATOR_LINES
    per_shot = navigator + scan.etl
    shots = len(nav) // per_shot

    assert list(nav) == ([1] * navigator + [0] * scan.etl) * shots
    assert list(rev) == [i % 2 for i in range(per_shot)] * shots
    assert rows[nav == 1, :2] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("te", [None, 25e-3], ids=["shortest", "25 ms"])
def test_the_centre_line_is_echoed_at_the_echo_time_written(case, te):
    scan = CASES[case](te=te)
    rows = echoes(scan.main, scan.samples)
    lin, nav = labels(scan.main, "LIN", "NAV")
    centre = (nav == 0) & (lin == scan.matrix[1] // 2)
    half_dwell = 0.5 * scan.dwell

    assert centre.any()
    assert rows[centre, 2] == pytest.approx(scan.echo_time, abs=half_dwell + 1e-9)
    if te is not None:
        assert scan.echo_time == pytest.approx(
            te, abs=scan.system.block_duration_raster
        )


@pytest.mark.parametrize("case", ["2D segmented", "2D multiband", "3D skipped-CAIPI"])
def test_shifted_echoes_grow_evenly_across_the_lattice_lines(case):
    """Each shot is delayed by its share of the echo spacing."""
    scan = CASES[case]()
    rows = echoes(scan.main, scan.samples)
    lin, nav = labels(scan.main, "LIN", "NAV")
    first = {}
    for line, (_, _, t) in zip(lin[nav == 0], rows[nav == 0], strict=True):
        first.setdefault(int(line), t)
    lines = sorted(first)
    steps = np.diff([first[line] for line in lines])
    raster = scan.system.block_duration_raster

    assert scan.n_shots > 1
    assert steps == pytest.approx(scan.esp / scan.n_shots, abs=raster + 1e-9)


@pytest.mark.parametrize("case", CASES)
def test_the_reference_reverses_the_phase_encode_and_keeps_the_labels(case):
    scan = CASES[case]()
    forward, reversed_ = scan.main, scan.prescan("reference")
    n = scan.samples
    ahead, behind = echoes(forward, n), echoes(reversed_, n)
    lin, nav = labels(forward, "LIN", "NAV")
    lin_r, nav_r, set_r = labels(reversed_, "LIN", "NAV", "SET")

    assert list(lin_r) == list(lin)
    assert set(set_r) == {1}
    assert behind[nav_r == 0, 0] == pytest.approx(-ahead[nav == 0, 0])
    assert behind[:, 1] == pytest.approx(ahead[:, 1])


# -- volumes, dummies and the output ----------------------------------------


@pytest.mark.parametrize("case", CASES)
def test_every_volume_is_marked_by_a_digital_output_dummies_included(case):
    scan = CASES[case](n_frames=3, n_dummy=2, volume_output=True)
    seq = scan.main
    blocks = [seq.get_block(i) for i in range(1, len(seq.block_events) + 1)]
    starts, t = [], 0.0
    for block in blocks:
        if getattr(block, "trig", None):
            assert block.rf is not None
            assert [trigger.channel for trigger in block.trig] == ["ext1"]
            starts.append(t)
        t += block.block_duration

    assert len(starts) == 3 + 2
    assert np.diff(starts) == pytest.approx(scan.repetition_time)


@pytest.mark.parametrize("case", CASES)
def test_a_time_series_labels_its_volumes_after_its_dummy_volumes(case):
    scan = CASES[case](n_frames=3, n_dummy=2)
    (rep,) = labels(scan.main, "REP")

    assert np.bincount(rep).tolist() == [len(rep) // 3] * 3
    assert excitations(scan.main) == (3 + 2) * scan.shots_per_volume


def test_a_single_2d_volume_plays_its_dummies_per_slice():
    scan = scan2d(n_slices=3, n_shots=2, n_dummy=2)

    assert excitations(scan.main) == 3 * (2 + 2)


def test_a_single_3d_volume_plays_its_dummies_as_shots():
    scan = scan3d(rz=2, n_dummy=3)

    assert excitations(scan.main) == 3 + scan3d(rz=2).shots_per_volume


@pytest.mark.parametrize("case", CASES)
def test_the_repetition_time_is_the_volume_asked_for(case):
    shortest = CASES[case]().repetition_time
    scan = CASES[case](tr=2 * shortest)
    # Each shot's pad is rounded onto the block raster.
    tolerance = scan.shots_per_volume * scan.system.block_duration_raster

    assert scan.repetition_time == pytest.approx(2 * shortest, abs=tolerance)
    assert scan.main.check_timing()[0]


# -- 2D slices and bands ----------------------------------------------------


def test_the_even_slices_of_a_packet_are_excited_before_the_odd_ones():
    assert played_slices(scan2d(n_slices=6)) == [0, 2, 4, 1, 3, 5]
    assert played_slices(scan2d(n_slices=8, multiband=2)) == [0, 2, 1, 3]


def test_a_single_volume_deals_slices_one_tr_cannot_hold_into_packets():
    shortest = scan2d().repetition_time
    scan = scan2d(n_slices=6, tr=2 * shortest)
    excited = np.asarray(scan.main.rf_times()[0])

    # Packets of two slices, so that the slices of a packet are not neighbours.
    assert played_slices(scan) == [0, 3, 1, 4, 2, 5]
    assert excited[1] - excited[0] == pytest.approx(shortest, abs=1e-9)
    assert np.diff(excited[::2]) == pytest.approx(scan.repetition_time, abs=1e-9)
    assert scan.main.check_timing()[0]


def test_a_time_series_whose_tr_cannot_hold_every_slice_is_refused():
    with pytest.raises(ValueError, match="cannot hold"):
        scan2d(n_slices=6, n_frames=2, tr=2 * scan2d().repetition_time)


def test_the_slice_timing_is_each_slices_excitation_within_the_volume():
    scan = scan2d(n_slices=4, multiband=2)
    excited = np.asarray(scan.main.rf_times()[0])
    shot = excited[1] - excited[0]

    # Groups 0 and 1 are excited in that order; slice k belongs to group k % 2.
    assert scan.main.definitions["SliceTiming"] == pytest.approx([0.0, shot, 0.0, shot])


@pytest.mark.parametrize(("ry", "multiband"), [(1, 2), (2, 3), (1, 4)])
def test_a_multiband_train_walks_the_caipi_lattice_from_the_centre_band(ry, multiband):
    scan = scan2d(n_slices=2 * multiband, multiband=multiband, ry=ry, n_acs_y=4)
    lin, par, nav = labels(scan.main, "LIN", "PAR", "NAV")
    shift = epi2d.caipi_shift(ry, multiband)
    lin, par = lin[nav == 0], par[nav == 0]

    assert list(par) == list((multiband // 2 + shift * ((lin - 8) // ry)) % multiband)


def test_a_multiband_shot_is_centred_on_its_slice_group():
    scan = scan2d(n_slices=6, slice_spacing=1e-3, multiband=3)
    seq = scan.main
    offsets = sorted(
        {
            round(seq.get_block(i).rf.freq_offset, 6)
            for i in range(1, len(seq.block_events) + 1)
            if seq.get_block(i).rf is not None
        }
    )
    # Slice k belongs to group k % 2, and the offset is the selection gradient's
    # amplitude, in Hz/m, times the mean position of the slices in the group.
    step = written(seq, "SliceThickness") + written(seq, "SliceGap")
    positions = (np.arange(6) - 5 / 2) * step
    centres = [np.mean(positions[g::2]) for g in range(2)]

    assert len(offsets) == 2
    assert offsets / np.asarray(centres) == pytest.approx(offsets[0] / centres[0])


def test_an_undersampled_2d_scan_calibrates_every_slice_with_a_gradient_echo():
    scan = scan2d(n_slices=3, ry=2, n_acs_y=4)
    lin, slc, ref = labels(scan.prescan("calibration"), "LIN", "SLC", "REF")

    assert list(zip(slc, lin, strict=True)) == [
        (s, line) for s in (0, 2, 1) for line in range(6, 10)
    ]
    assert set(ref) == {1}


@pytest.mark.parametrize(
    "prescription",
    [
        {"partial_fourier_y": 0.4},
        {"n_slices": 3, "multiband": 2},
        {"n_shots": 0},
        {"te": 1e-3},
    ],
    ids=str,
)
def test_an_impossible_2d_prescription_is_refused(prescription):
    with pytest.raises(ValueError):
        scan2d(**prescription)


# -- 3D shells --------------------------------------------------------------


@pytest.mark.parametrize(
    ("ry", "rz", "shift"),
    [
        (1, 1, 0),
        (1, 2, 1),
        (1, 3, 1),
        # Patterns Stirnberg and Stöcker (MRM 2021) found best, Tables 1 and 2.
        (2, 2, 1),
        (3, 2, 1),
        (4, 2, 1),
        (2, 3, 1),
        (2, 4, 2),
        (1, 4, 2),
        (1, 5, 2),
        (1, 6, 2),
        (1, 7, 2),
    ],
)
def test_the_caipi_shift_keeps_the_aliases_furthest_apart(ry, rz, shift):
    assert epi3d.caipi_shift(ry, rz) == shift
    assert epi2d.caipi_shift(ry, rz) == shift


def blip_cycle(steps):
    """The shortest period the blip sequence repeats with."""
    return next(
        n
        for n in range(1, len(steps) + 1)
        if all(steps[i] == steps[i % n] for i in range(len(steps)))
    )


@pytest.mark.parametrize(
    ("ry", "rz", "n_shots"),
    [(1, 4, 1), (1, 4, 2), (1, 4, 3), (2, 2, 1), (1, 6, 1), (1, 6, 4), (1, 5, 3)],
)
def test_the_partition_blips_are_the_two_the_paper_derives(ry, rz, n_shots):
    """Stirnberg and Stöcker (MRM 2021), Appendix, equations A1 and A2."""
    scan = scan3d(n_y=48, n_z=2 * rz, ry=ry, rz=rz, n_shots=n_shots)
    b1 = (n_shots * epi3d.caipi_shift(ry, rz)) % rz
    b2 = (rz - b1) % rz
    smallest = min(b1, b2)
    if smallest == 0:
        cycle = 1
    elif rz % smallest == 0:
        cycle = rz // smallest
    else:
        cycle = rz

    par, nav = labels(scan.main, "PAR", "NAV")
    for shot in par[nav == 0].reshape(-1, scan.etl):
        steps = list(np.diff(shot))
        assert set(steps) <= {b1, -b2}
        assert cycle % blip_cycle(steps) == 0


@pytest.mark.parametrize(
    ("ry", "rz", "n_shots"), [(1, 1, 1), (2, 2, 1), (2, 2, 2), (1, 3, 2), (2, 4, 3)]
)
def test_every_lattice_view_is_read_once_per_volume(ry, rz, n_shots):
    scan = scan3d(n_y=24, n_z=12, ry=ry, rz=rz, n_shots=n_shots)
    lin, par, nav = labels(scan.main, "LIN", "PAR", "NAV")
    views = list(zip(lin[nav == 0], par[nav == 0], strict=True))
    shift = epi3d.caipi_shift(ry, rz)
    lattice = {
        (y, z)
        for y in range(24)
        if (y - 12) % ry == 0
        for z in range(12)
        if (z - 6 - shift * ((y - 12) // ry)) % rz == 0
    }

    assert len(views) == len(set(views))
    assert set(views) <= lattice
    assert (12, 6) in views
    covered = {(y, z) for y, z in lattice if min(epi3d.shell_bases(12, rz, 1.0)) <= z}
    assert len(covered - set(views)) < n_shots * 24 // ry


def test_a_shell_is_read_whole_before_the_next_and_the_centre_shell_first():
    scan = scan3d(n_z=12, rz=3)
    par, nav = labels(scan.main, "PAR", "NAV")
    shots = par[nav == 0].reshape(-1, scan.etl)
    shells = [int(shot.min()) for shot in shots]

    assert all(
        set(shot) == set(range(s, s + 3)) for s, shot in zip(shells, shots, strict=True)
    )
    assert shells[0] <= 6 < shells[0] + 3
    assert sorted(shells) == [0, 3, 6, 9]


def test_shots_that_climb_a_whole_cycle_stay_in_their_partition():
    """Shot-selective CAIPI: the shift times the shot count is a multiple of rz."""
    scan = scan3d(ry=1, rz=2, n_shots=2)
    par, nav = labels(scan.main, "PAR", "NAV")
    shots = par[nav == 0].reshape(-1, scan.etl)

    assert written(scan.main, "CaipiShift") * scan.n_shots % 2 == 0
    assert all(len(set(shot)) == 1 for shot in shots)


def test_one_shot_per_shell_is_blipped_caipi():
    scan = scan3d(ry=1, rz=2)
    par, nav = labels(scan.main, "PAR", "NAV")
    shots = par[nav == 0].reshape(-1, scan.etl)

    assert all(len(set(shot)) == 2 for shot in shots)


def test_partial_fourier_drops_the_leading_lines_and_shells():
    full = scan3d(n_z=8, rz=2)
    partial = scan3d(n_z=8, rz=2, partial_fourier_y=0.75, partial_fourier_z=0.5)
    par, nav = labels(partial.main, "PAR", "NAV")
    shells = [int(shot.min()) for shot in par[nav == 0].reshape(-1, partial.etl)]

    assert len(shells) == 2 and sorted(shells) == [4, 6]
    assert partial.etl == 12 and full.etl == 16
    assert partial.echo_time < full.echo_time


def test_an_undersampled_3d_scan_calibrates_over_the_central_rectangle():
    scan = scan3d(ry=2, n_acs_y=4, n_acs_z=2)
    lin, par, ref = labels(scan.prescan("calibration"), "LIN", "PAR", "REF")

    assert list(zip(lin, par, strict=True)) == [
        (y, z) for y in range(6, 10) for z in range(3, 5)
    ]
    assert set(ref) == {1}


@pytest.mark.parametrize("excitation", ["slab", "nonselective", "spsp"])
def test_every_excitation_builds_a_3d_epi_that_passes_its_timing_check(excitation):
    seq = scan3d(excitation=excitation, rz=2).main

    assert seq.check_timing()[0]
    assert seq.definitions["Excitation"] == excitation


@pytest.mark.parametrize(
    "prescription",
    [
        {"partial_fourier_z": 0.4},
        {"excitation": "adiabatic"},
        {"rz": 0},
        {"rz": 16},
        {"tr": 1e-3},
    ],
    ids=str,
)
def test_an_impossible_3d_prescription_is_refused(prescription):
    with pytest.raises(ValueError):
        scan3d(**prescription)


@pytest.mark.parametrize(
    ("module", "flag", "help_text"),
    [
        (epi2d, "--multiband", "Slices excited at once."),
        (epi3d, "--volume-output", "Play a digital output on"),
    ],
)
def test_an_epi_flag_is_named_and_described_by_its_docstring(
    capsys, module, flag, help_text
):
    with pytest.raises(SystemExit):
        cli.run(module.main, ["--help"])

    printed = " ".join(capsys.readouterr().out.split())

    assert flag in printed
    assert help_text in printed
