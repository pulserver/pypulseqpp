"""The 2D and 3D EPI example sequences: linked prescans, sampling, timing and labels."""

import re
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
    "2D spin echo": lambda **kw: scan2d(spin_echo=True, ry=2, n_y=32, **kw),
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
    """The excitation pulses of ``seq``; a refocusing pulse is not one."""
    blocks = (seq.get_block(i) for i in range(1, len(seq.block_events) + 1))
    return sum(
        block.rf is not None and getattr(block.rf, "use", "") != "refocusing"
        for block in blocks
    )


# -- the linked chain ------------------------------------------------------


@pytest.mark.parametrize(
    ("scan", "files"),
    [
        (
            lambda: scan2d(n_slices=2, ry=2, n_acs_y=4),
            [
                ("scan.seq", "epi_2d_calibration"),
                ("scan_epi_2d_field_forward.seq", "epi_2d_field_forward"),
                ("scan_epi_2d_field_reversed.seq", "epi_2d_field_reversed"),
                ("scan_epi_2d.seq", "epi_2d"),
            ],
        ),
        (
            lambda: scan2d(n_dummy=1),
            [
                ("scan.seq", "epi_2d_calibration"),
                ("scan_epi_2d_field_forward.seq", "epi_2d_field_forward"),
                ("scan_epi_2d_field_reversed.seq", "epi_2d_field_reversed"),
                ("scan_epi_2d.seq", "epi_2d"),
            ],
        ),
        (
            lambda: scan2d(n_slices=4, multiband=2, n_acs_y=4),
            [
                ("scan.seq", "epi_2d_calibration"),
                ("scan_epi_2d_field_forward.seq", "epi_2d_field_forward"),
                ("scan_epi_2d_field_reversed.seq", "epi_2d_field_reversed"),
                ("scan_epi_2d.seq", "epi_2d"),
            ],
        ),
        (
            lambda: scan3d(ry=2, n_acs_y=4, n_acs_z=2),
            [
                ("scan.seq", "epi_3d_calibration"),
                ("scan_epi_3d_field_forward.seq", "epi_3d_field_forward"),
                ("scan_epi_3d_field_reversed.seq", "epi_3d_field_reversed"),
                ("scan_epi_3d.seq", "epi_3d"),
            ],
        ),
        (
            lambda: scan3d(n_dummy=1),
            [
                ("scan.seq", "epi_3d_calibration"),
                ("scan_epi_3d_field_forward.seq", "epi_3d_field_forward"),
                ("scan_epi_3d_field_reversed.seq", "epi_3d_field_reversed"),
                ("scan_epi_3d.seq", "epi_3d"),
            ],
        ),
        (
            lambda: scan2d(n_slices=2, ry=2, n_acs_y=4, calibration="gre"),
            [
                ("scan.seq", "epi_2d_calibration"),
                ("scan_epi_2d_reference.seq", "epi_2d_reference"),
                ("scan_epi_2d_field_forward.seq", "epi_2d_field_forward"),
                ("scan_epi_2d_field_reversed.seq", "epi_2d_field_reversed"),
                ("scan_epi_2d.seq", "epi_2d"),
            ],
        ),
        (
            lambda: scan3d(ry=2, n_acs_y=4, n_acs_z=2, calibration="gre"),
            [
                ("scan.seq", "epi_3d_calibration"),
                ("scan_epi_3d_reference.seq", "epi_3d_reference"),
                ("scan_epi_3d_field_forward.seq", "epi_3d_field_forward"),
                ("scan_epi_3d_field_reversed.seq", "epi_3d_field_reversed"),
                ("scan_epi_3d.seq", "epi_3d"),
            ],
        ),
    ],
    ids=[
        "2D undersampled",
        "2D fully sampled",
        "2D multiband",
        "3D undersampled",
        "3D fully sampled",
        "2D gradient-echo calibration",
        "3D gradient-echo calibration",
    ],
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


@pytest.mark.parametrize(
    "make",
    [
        lambda: scan2d(ry=2, partial_fourier_y=0.75),
        lambda: scan2d(ry=2, partial_fourier_y=0.75, calibration="gre"),
        lambda: scan3d(ry=2, rz=2, partial_fourier_y=0.75, partial_fourier_z=0.75),
        lambda: scan3d(ry=2, rz=2, partial_fourier_z=0.75, calibration="gre"),
    ],
    ids=["2D FLEET", "2D GRE", "3D FLEET", "3D GRE"],
)
def test_every_file_of_a_partial_fourier_chain_states_the_k_space_centre(make):
    """A reconstruction centres a file that states none on the lines it read."""
    scan = make()
    for seq in scan.chain:
        assert written(seq, "kSpaceCenterLine") == scan.matrix[1] // 2
        # The field pair of a 3D scan is a stack of slices.
        stack = "_field_" in seq.definitions["Name"]
        if scan.module is epi3d and scan.matrix[2] > 1 and not stack:
            assert written(seq, "kSpaceCenterPartition") == scan.matrix[2] // 2


@pytest.mark.parametrize("case", CASES)
def test_every_file_of_the_chain_repeats_from_its_first_block(tmp_path, case):
    scan = CASES[case](n_dummy=2, n_frames=2, volume_output=True, n_acs_y=4)
    for path in sequences.write(tmp_path / "scan.seq", scan.chain):
        seq = pp.Sequence(system=scan.system)
        seq.read(path)
        assert seq.repetition()[1] == 1, path


# -- what every shot samples ------------------------------------------------


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("prescan", [None, "reference"], ids=["main", "reference"])
def test_every_line_is_labelled_with_the_view_it_samples(case, prescan):
    scan = CASES[case](**({} if prescan is None else {"calibration": "gre"}))
    seq = scan.main if prescan is None else scan.prescan(prescan)
    rows = echoes(seq, scan.samples)
    lin, par, nav = labels(seq, "LIN", "PAR", "NAV")
    imaging = nav == 0
    n_y = scan.matrix[1]

    # The centre line's k is the sum of the encodes played, inverted by a
    # refocusing pulse, zero to the rounding of those additions.
    assert rows[imaging, 0] == pytest.approx(
        (lin[imaging] - n_y // 2) / scan.fov[1], abs=1e-9
    )
    if case not in ("2D", "2D segmented", "2D spin echo"):
        fov, n = partition_fov(scan)
        # A centre partition's k is the sum of the encodes it played, zero to
        # the rounding of a few hundred additions.
        assert rows[imaging, 1] == pytest.approx(
            (par[imaging] - n // 2) / fov, abs=1e-9
        )


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("prescan", [None, "reference"], ids=["main", "reference"])
def test_every_shot_opens_with_its_navigator_at_the_centre_of_k_space(case, prescan):
    scan = CASES[case](**({} if prescan is None else {"calibration": "gre"}))
    rows = echoes(scan.main, scan.samples)
    nav, rev, lin = labels(scan.main, "NAV", "REV", "LIN")
    navigator = scan.module.NAVIGATOR_LINES
    per_shot = navigator + scan.etl
    shots = len(nav) // per_shot
    if prescan is not None:
        # The reference samples every line of the scan but the mirror of line 0.
        kept = (nav == 1) | (lin != 0)
        nav, rev = nav[kept], rev[kept]
        rows = echoes(scan.prescan(prescan), scan.samples)
        found_nav, found_rev = labels(scan.prescan(prescan), "NAV", "REV")
        assert list(found_nav) == list(nav)
        assert list(found_rev) == list(rev)
    else:
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
def test_the_reference_reads_the_mirror_of_every_line_and_is_labelled_there(case):
    scan = CASES[case](calibration="gre")
    forward, reversed_ = scan.main, scan.prescan("reference")
    n, n_y = scan.samples, scan.matrix[1]
    ahead, behind = echoes(forward, n), echoes(reversed_, n)
    lin, nav = labels(forward, "LIN", "NAV")
    lin_r, nav_r, set_r = labels(reversed_, "LIN", "NAV", "SET")
    # The mirror of line 0, n_y, lies outside the matrix and is not sampled.
    kept = (nav == 1) | (lin != 0)
    imaging = (nav == 0) & kept

    assert list(lin_r[nav_r == 0]) == list(n_y - lin[imaging])
    assert set(set_r) == {1}
    assert behind[nav_r == 0, 0] == pytest.approx(-ahead[imaging, 0])
    assert behind[:, 1:] == pytest.approx(ahead[kept, 1:])


@pytest.mark.parametrize("case", CASES)
def test_the_reference_defines_the_echo_and_repetition_time_of_the_main_scan(case):
    scan = CASES[case](calibration="gre")
    reference = scan.prescan("reference")

    for key in ("TE", "TR"):
        assert reference.definitions[key] == pytest.approx(scan.main.definitions[key])


@pytest.mark.parametrize("prescan", ["calibration", "reference"])
def test_each_prescan_of_a_2d_scan_places_its_slices_as_the_scan_does(prescan):
    scan = scan2d(n_slices=3, slice_spacing=2e-3, ry=2, n_acs_y=4, calibration="gre")
    seq = scan.prescan(prescan)

    for key in ("SlicePositions", "SliceThickness", "SliceGap"):
        assert seq.definitions[key] == pytest.approx(scan.main.definitions[key]), key


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
    scan = scan2d(n_slices=3, ry=2, n_acs_y=4, calibration="gre")
    lin, slc, ref = labels(scan.prescan("calibration"), "LIN", "SLC", "REF")

    assert list(zip(slc, lin, strict=True)) == [
        (s, line) for s in (0, 2, 1) for line in range(6, 10)
    ]
    assert set(ref) == {1}


@pytest.mark.parametrize(
    "make",
    [
        lambda flip: scan2d(n_slices=2, ry=2, n_acs_y=4, flip_angle_deg=flip),
        lambda flip: scan2d(n_slices=2, multiband=2, flip_angle_deg=flip),
        *(
            lambda flip, excitation=excitation: scan3d(
                ry=2, n_acs_y=4, n_acs_z=2, excitation=excitation, flip_angle_deg=flip
            )
            for excitation in sequences.EXCITATIONS
        ),
    ],
    ids=["2D", "2D multiband", *(f"3D {e}" for e in sequences.EXCITATIONS)],
)
@pytest.mark.parametrize("flip", [20.0, 90.0])
def test_the_calibration_plays_the_imaging_pulse_at_its_own_flip_angle(make, flip):
    scan = make(flip)
    calibration = scan.prescan("calibration").rf_instances()
    # A multiband scan calibrates single-band, with the pulse of one band.
    imaging = scan.main if scan.prescription.get("multiband", 1) == 1 else None
    (pulse,) = calibration.definitions

    assert pulse.flip_deg * calibration.amplitude == pytest.approx(
        scan.module.CALIBRATION_FLIP_DEG, rel=1e-4
    )
    if imaging is not None:
        instances = imaging.rf_instances()
        excitation = next(d for d in instances.definitions if d.use == "excitation")
        assert pulse.time == pytest.approx(excitation.time)
        assert pulse.waveform == pytest.approx(excitation.waveform)


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
    partial = scan3d(
        n_z=8,
        rz=2,
        partial_fourier_y=0.75,
        partial_fourier_z=0.5,
        full_k_reference=False,
    )
    par, nav = labels(partial.main, "PAR", "NAV")
    shells = [int(shot.min()) for shot in par[nav == 0].reshape(-1, partial.etl)]

    assert len(shells) == 2 and sorted(shells) == [4, 6]
    assert partial.etl == 12 and full.etl == 16
    assert partial.echo_time < full.echo_time


def _echo_slope(seq, sets):
    """Echo time against line, per set, of the readouts of ``seq`` that are not navigators (s per line)."""
    _, _, t_excitation, _, t_adc = seq.calculate_kspace()
    found = seq.evaluate_labels(evolution="adc")
    count = len(found["LIN"])
    middle = t_adc.reshape(count, -1)[:, t_adc.size // count // 2]
    echo = middle - t_excitation[np.searchsorted(t_excitation, middle) - 1]
    lines = found["LIN"][found["NAV"] == 0]
    echo = echo[found["NAV"] == 0]
    polarity = found.get("SET", np.zeros(count, int))[found["NAV"] == 0]
    return [np.polyfit(lines[polarity == s], echo[polarity == s], 1)[0] for s in sets]


@pytest.mark.parametrize(
    "make",
    [
        lambda: scan2d(n_slices=2, ry=2, n_acs_y=8),
        lambda: scan2d(ry=2, n_shots=2, n_acs_y=8),
        lambda: scan2d(n_slices=2, multiband=2, n_acs_y=8),
        lambda: scan3d(ry=2, rz=2, n_acs_y=8, n_acs_z=2),
    ],
    ids=["2D", "2D segmented", "2D multiband", "3D"],
)
def test_the_fleet_calibration_steps_through_its_lines_as_the_imaging_does_at_a_shorter_echo_time(
    make,
):
    scan = make()
    calibration = scan.prescan("calibration")
    forward, reversed_ = _echo_slope(calibration, (0, 1))
    (imaging,) = _echo_slope(scan.main, (0,))

    assert written(calibration, "EchoSpacing") == pytest.approx(scan.esp)
    assert written(calibration, "TE") < scan.echo_time
    assert forward == pytest.approx(imaging, rel=0.02)
    assert reversed_ == pytest.approx(-imaging, rel=0.02)


def test_an_undersampled_2d_scan_calibrates_each_slice_with_fleet_shots_in_both_polarities():
    scan = scan2d(n_slices=3, ry=2, n_acs_y=4)
    lin, slc, ref, seg, polarity, nav = labels(
        scan.prescan("calibration"), "LIN", "SLC", "REF", "SEG", "SET", "NAV"
    )
    read = nav == 0

    # Two shots of two lines a slice, forward then reversed, each line of the
    # reversed shots at its mirror n_y - L.
    assert [
        (s, p, g, line)
        for s, p, g, line in zip(
            slc[read], polarity[read], seg[read], lin[read], strict=True
        )
    ] == [
        (s, p, g, 16 - line if p else line)
        for s in (0, 2, 1)
        for p in (0, 1)
        for g in (0, 1)
        for line in (6 + g, 8 + g)
    ]
    assert set(ref) == {1}


def test_an_undersampled_3d_scan_calibrates_over_the_central_rectangle():
    scan = scan3d(ry=2, n_acs_y=4, n_acs_z=2, calibration="gre")
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


def _amplitude(event):
    """Largest magnitude a gradient event reaches (Hz/m)."""
    if event is None:
        return 0.0
    if event.type == "trap":
        return abs(float(event.amplitude))
    return float(np.max(np.abs(np.asarray(event.waveform)), initial=0.0))


def _shot_of(seq, first_readout, last_readout):
    """Blocks from the excitation before ``first_readout`` to the rewinder after ``last_readout``, 1-based."""
    start = first_readout
    while seq.get_block(start).rf is None:
        start -= 1
    return range(start, last_readout + 2)


def _readout_blocks(seq):
    blocks = np.asarray(seq._native.block_events())
    return np.flatnonzero(blocks[:, 4] > 0) + 1


@pytest.mark.parametrize(
    "case", ["2D", "2D segmented", "2D multiband", "3D", "3D skipped-CAIPI"]
)
def test_the_phase_reference_is_the_imaging_train_with_every_encode_at_zero(case):
    scan = CASES[case](phase_reference=True)
    seq = scan.main
    pc, echo, rev, nav = labels(seq, "PHASECORR", "USER0", "REV", "NAV")
    readouts = _readout_blocks(seq)
    reference = readouts[pc != 0]
    imaging = readouts[pc == 0]

    assert not nav.any()
    assert reference.size and reference.size % scan.etl == 0
    first = reference[: scan.etl]
    assert list(echo[pc != 0][: scan.etl]) == list(range(scan.etl))
    encoded = ("gy", "gz") if scan.module is epi3d else ("gy",)
    for block in _shot_of(seq, int(first[0]), int(first[-1])):
        found = seq.get_block(block)
        if found.rf is None:
            assert max(_amplitude(getattr(found, ch)) for ch in encoded) == 0.0
    # Echo k of the reference is echo k of an imaging shot: read lobe, ADC,
    # block duration and polarity.
    lines = dict(zip(echo[pc == 0][: scan.etl], imaging[: scan.etl], strict=True))
    polarity = dict(zip(readouts, rev, strict=True))
    for k, block in enumerate(first):
        ref, img = seq.get_block(int(block)), seq.get_block(int(lines[k]))
        assert ref.block_duration == pytest.approx(img.block_duration)
        assert _amplitude(ref.gx) == pytest.approx(_amplitude(img.gx))
        assert polarity[block] == polarity[lines[k]]
        for field in ("num_samples", "dwell", "delay"):
            assert getattr(ref.adc, field) == pytest.approx(getattr(img.adc, field))


def test_the_phase_reference_replaces_the_navigator_and_shortens_te_and_tr():
    with_navigator = scan2d(n_slices=2, n_dummy=2)
    referenced = scan2d(n_slices=2, n_dummy=2, phase_reference=True)
    (pc,) = labels(referenced.main, "PHASECORR")

    assert referenced.echo_time < with_navigator.echo_time
    assert referenced.repetition_time < with_navigator.repetition_time
    assert np.count_nonzero(pc) == 2 * referenced.etl


def test_a_phase_reference_without_dummies_plays_one_dummy_cycle_for_it():
    scan = scan2d(n_slices=2, n_dummy=0, phase_reference=True)
    (pc, slc, once) = labels(scan.main, "PHASECORR", "SLC", "ONCE")

    assert np.count_nonzero(pc) == 2 * scan.etl
    assert sorted(set(slc[pc != 0])) == [0, 1]
    assert (once[pc != 0] == 1).all() and (once[pc == 0] == 0).all()


@pytest.mark.parametrize("case", ["2D segmented", "3D skipped-CAIPI"])
def test_the_fleet_calibration_takes_its_reference_from_its_dummy_shot(case):
    scan = CASES[case](phase_reference=True)
    calibration = scan.prescan("calibration")
    pc, echo, ref = labels(calibration, "PHASECORR", "USER0", "REF")

    assert pc.any()
    first = int(np.flatnonzero(pc == 0)[0])
    assert (pc[:first] != 0).all() and list(echo[:first]) == list(range(first))
    assert (ref[pc == 0] == 1).all()


def test_the_prescan_gain_calibration_does_not_count_the_reference_as_imaging():
    scan = scan2d(n_dummy=1, phase_reference=True)
    (pc,) = labels(scan.main, "PHASECORR")

    # The dummy shot carrying the reference plays before the first imaging
    # shot, which the prescan plays through.
    assert sequences.gain_calibration_readouts(scan.main) >= 2 * scan.etl
    assert np.count_nonzero(pc) == scan.etl


@pytest.mark.parametrize("multiband", [2, 3, 4])
def test_each_multiband_excitation_places_its_bands_on_the_slices_of_its_group(
    multiband,
):
    n_slices = 2 * multiband
    *_, seq = sequences.epi2D_sequence(
        **SMALL_2D, n_slices=n_slices, multiband=multiband, slice_spacing=1e-3
    )
    positions = np.asarray(seq.get_definition("SlicePositions"), dtype=float)
    groups = n_slices // multiband
    pulse = sequences.SmsExcitation(
        seq.system,
        70.0,
        thickness_m=5e-3,
        slice_gap_m=groups * (positions[1] - positions[0]),
        n_bands=multiband,
        duration_s=3e-3,
        time_bw_product=4.0,
    )
    offsets = sorted(
        {
            round(float(seq.get_block(n).rf.freq_offset), 6)
            for n in seq.block_events
            if seq.get_block(n).rf is not None
            and getattr(seq.get_block(n).rf, "use", "") == "excitation"
        }
    )

    excited = [
        np.sort(offset / float(pulse.gz.amplitude) + np.asarray(pulse.band_positions_m))
        for offset in offsets
    ]
    expected = [np.sort(positions[g::groups]) for g in range(groups)]
    np.testing.assert_allclose(sorted(excited, key=lambda b: b[0]), expected, atol=1e-6)


@pytest.mark.parametrize("ry", [1, 2])
def test_each_multiband_line_is_received_at_the_phase_its_partition_encode_gives_its_group_centre(
    ry,
):
    *_, seq = sequences.epi2D_sequence(
        **SMALL_2D, n_slices=4, multiband=2, ry=ry, slice_spacing=1e-3
    )
    positions = np.asarray(seq.get_definition("SlicePositions"), dtype=float)
    pulse = sequences.SmsExcitation(
        seq.system,
        70.0,
        thickness_m=5e-3,
        slice_gap_m=2 * (positions[1] - positions[0]),
        n_bands=2,
        duration_s=3e-3,
        time_bw_product=4.0,
    )
    k_adc = seq.calculate_kspace()[0]

    sample, centre, checked = 0, 0.0, 0
    for n in seq.block_events:
        block = seq.get_block(n)
        if block.rf is not None and getattr(block.rf, "use", "") == "excitation":
            offset = float(block.rf.freq_offset) / float(pulse.gz.amplitude)
            centre = offset + float(np.mean(pulse.band_positions_m))
        if block.adc is None:
            continue
        samples = int(block.adc.num_samples)
        kz = k_adc[2, sample + samples // 2]
        sample += samples
        expected = 2 * np.pi * kz * centre
        assert np.angle(
            np.exp(1j * (block.adc.phase_offset - expected))
        ) == pytest.approx(0.0, abs=1e-6)
        checked += abs(kz) > 0
    assert checked > 0


@pytest.mark.parametrize("make", [scan2d, scan3d], ids=["2D", "3D"])
def test_a_fleet_calibration_takes_the_place_of_the_reversed_reference_volume(make):
    names = [seq.definitions["Name"] for seq in make(ry=2, field_pair=False).chain[:-1]]

    assert names == [make().main.definitions["Name"] + "_calibration"]


# -- spin echo and diffusion weighting ---------------------------------------


def shot_b_values(seq, echo_time):
    """The b-value (s/mm^2) of every excitation's echo at ``echo_time``, from the played gradients."""
    wave, t_excitation, t_refocusing, _, _ = seq.waveforms_and_times()
    raster = 1e-6
    values = []
    for start in np.asarray(t_excitation[0]):
        t = np.arange(start, start + echo_time, raster)
        sign = np.where(
            (np.asarray(t_refocusing[0])[None, :] > start)
            & (np.asarray(t_refocusing[0])[None, :] < t[:, None]),
            -1.0,
            1.0,
        ).prod(axis=1)
        k = np.zeros((3, t.size))
        for axis, channel in enumerate(wave[:3]):
            if np.size(channel):
                g = np.interp(t, channel[0], channel[1], left=0.0, right=0.0)
                k[axis] = np.cumsum(sign * g) * raster
        values.append((2 * np.pi) ** 2 * np.sum(k**2) * raster / 1e6)
    return np.array(values)


def test_a_spin_echo_refocuses_the_centre_line_at_the_echo_time_written():
    scan = scan2d(spin_echo=True, te=60e-3)
    _, _, _, t_refocusing, _ = scan.main.calculate_kspace()
    _, _, t_excitation, _, _ = scan.main.calculate_kspace()

    assert scan.echo_time == pytest.approx(60e-3, abs=scan.system.block_duration_raster)
    assert np.asarray(t_refocusing)[0] - np.asarray(t_excitation)[0] == pytest.approx(
        scan.echo_time / 2, abs=scan.system.block_duration_raster
    )


def test_a_field_pair_is_a_forward_and_a_reversed_spin_echo_volume_labelled_two_and_three():
    scan = scan2d(n_slices=2, field_pair=True)
    forward, reversed_ = scan.prescan("field_forward"), scan.prescan("field_reversed")
    n = scan.samples

    for seq, polarity in ((forward, 2), (reversed_, 3)):
        (sets,) = labels(seq, "SET")
        _, _, t_excitation, t_refocusing, _ = seq.calculate_kspace()
        assert set(sets) == {polarity}
        assert len(np.asarray(t_refocusing)) == len(np.asarray(t_excitation)) > 0
    lin, nav = labels(forward, "LIN", "NAV")
    lin_r, nav_r = labels(reversed_, "LIN", "NAV")
    # The mirror of line 0, n_y, lies outside the matrix and is not sampled.
    imaging = (nav == 0) & (lin != 0)
    assert list(lin_r[nav_r == 0]) == list(scan.matrix[1] - lin[imaging])
    assert echoes(reversed_, n)[nav_r == 0, 0] == pytest.approx(
        -echoes(forward, n)[imaging, 0]
    )


def test_a_field_pair_is_played_after_the_calibration_and_before_the_scan():
    names = [seq.definitions["Name"] for seq in scan2d(field_pair=True).chain]

    assert names == [
        "epi_2d_calibration",
        "epi_2d_field_forward",
        "epi_2d_field_reversed",
        "epi_2d",
    ]


def test_a_field_pair_leaves_the_scan_unchanged(tmp_path):
    plain, paired = scan2d(te=40e-3), scan2d(te=40e-3, field_pair=True)
    plain.main.write(tmp_path / "plain.seq")
    paired.main.write(tmp_path / "paired.seq")

    assert (tmp_path / "paired.seq").read_text() == (tmp_path / "plain.seq").read_text()


def test_a_partial_fourier_gradient_echo_scan_plays_a_full_k_reference_at_its_echo_time():
    scan = scan2d(n_y=32, ry=2, partial_fourier_y=0.75)
    reference = scan.prescan("full_k_reference")

    assert [seq.definitions["Name"] for seq in scan.chain][-2:] == [
        "epi_2d_full_k_reference",
        "epi_2d",
    ]
    assert reference.definitions["TE"] == scan.main.definitions["TE"]
    (sets,) = labels(reference, "SET")
    assert set(sets) == {4}
    lin, nav = labels(reference, "LIN", "NAV")
    lin_scan, nav_scan = labels(scan.main, "LIN", "NAV")
    assert min(lin[nav == 0]) == 0 < min(lin_scan[nav_scan == 0])
    assert max(lin[nav == 0]) == max(lin_scan[nav_scan == 0])


def test_a_prescribed_echo_time_below_that_of_the_full_k_reference_is_refused():
    with pytest.raises(ValueError, match="full-k phase reference"):
        epi2d(**{**SMALL_2D, "n_y": 64}, partial_fourier_y=0.6, te=8e-3)


@pytest.mark.parametrize(
    "prescription",
    [{"partial_fourier_y": 1.0}, {"partial_fourier_y": 0.75, "spin_echo": True}],
    ids=["fully sampled", "spin echo"],
)
def test_a_fully_sampled_or_spin_echo_scan_plays_no_full_k_reference(prescription):
    names = [seq.definitions["Name"] for seq in scan2d(**prescription).chain]

    assert "epi_2d_full_k_reference" not in names


def test_a_3d_field_pair_is_a_spin_echo_stack_over_the_slab_at_the_scan_echo_spacing():
    scan = scan3d(ry=2, rz=2, partial_fourier_y=0.75)
    forward, reversed_ = scan.prescan("field_forward"), scan.prescan("field_reversed")
    fov_z, n_z = scan.main.definitions["FOV"][2], scan.matrix[2]

    for seq, polarity in ((forward, 2), (reversed_, 3)):
        (sets,) = labels(seq, "SET")
        _, _, t_excitation, t_refocusing, _ = seq.calculate_kspace()
        assert set(sets) == {polarity}
        assert len(np.asarray(t_refocusing)) == len(np.asarray(t_excitation)) > 0
        assert seq.definitions["EchoSpacing"] == scan.main.definitions["EchoSpacing"]
        assert seq.definitions["Matrix"][:2] == scan.main.definitions["Matrix"][:2]
        positions = np.asarray(seq.definitions["SlicePositions"])
        assert len(positions) == n_z
        assert np.ptp(positions) == pytest.approx(fov_z * (n_z - 1) / n_z)


def test_a_partial_fourier_3d_scan_plays_a_full_k_reference_at_its_echo_time():
    scan = scan3d(ry=2, rz=2, partial_fourier_y=0.75, partial_fourier_z=0.75)
    reference = scan.prescan("full_k_reference")

    assert [seq.definitions["Name"] for seq in scan.chain][-2:] == [
        "epi_3d_full_k_reference",
        "epi_3d",
    ]
    assert reference.definitions["TE"] == scan.main.definitions["TE"]
    (sets,) = labels(reference, "SET")
    assert set(sets) == {4}
    lin, par, nav = labels(reference, "LIN", "PAR", "NAV")
    lin_scan, par_scan, nav_scan = labels(scan.main, "LIN", "PAR", "NAV")
    assert min(lin[nav == 0]) < min(lin_scan[nav_scan == 0])
    assert min(par[nav == 0]) < min(par_scan[nav_scan == 0])


def test_a_multiband_scan_plays_its_field_pair_single_band_over_the_same_slices():
    *_, forward, reversed_volume, scan = epi2d(
        **SMALL_2D, n_slices=4, multiband=2, field_pair=True
    )

    for volume in (forward, reversed_volume):
        assert "MultibandFactor" not in volume.definitions
        np.testing.assert_allclose(
            volume.definitions["SlicePositions"], scan.definitions["SlicePositions"]
        )
    assert scan.definitions["MultibandFactor"] == [2.0]


def test_each_diffusion_volume_is_weighted_with_the_b_value_of_its_table_row():
    # The crushers and the read and phase-encode prewinders add cross terms
    # of a few percent to the lobes' b-value.
    scan = scan2d(spin_echo=True, diffusion_directions=6, diffusion_b0=2)
    b_table, _ = pp.io.get_diffusion_table(scan.main)
    played = shot_b_values(scan.main, scan.echo_time)

    assert played.size == b_table.size
    assert played[b_table == 0] == pytest.approx(0.0, abs=5.0)
    assert played[b_table > 0] == pytest.approx(b_table[b_table > 0], rel=0.08)


def test_a_diffusion_scan_file_carries_its_table_for_the_reconstruction(tmp_path):
    scan = scan2d(spin_echo=True, diffusion_directions=6, diffusion_b0=2)
    scan.main.write(tmp_path / "dwi.seq")
    b, g = pp.io.get_diffusion_table(pp.io.read(tmp_path / "dwi.seq"))

    assert b.tolist() == [0.0, 0.0] + [1000.0] * 6
    assert g[2:] == pytest.approx(pp.io.uniform_directions(6), abs=1e-6)


def test_a_higher_b_value_lengthens_the_shortest_spin_echo():
    low = scan2d(spin_echo=True, diffusion_directions=6, b_value=500e6)
    high = scan2d(spin_echo=True, diffusion_directions=6, b_value=2000e6)

    assert high.echo_time > low.echo_time


@pytest.mark.parametrize(
    "prescription",
    [
        {"diffusion_directions": 6},
        {"spin_echo": True, "n_slices": 2, "multiband": 2},
        {"spin_echo": True, "diffusion_directions": 6, "te": 30e-3},
        {"spin_echo": True, "diffusion_directions": 6, "diffusion_b0": -1},
    ],
    ids=[
        "diffusion without spin echo",
        "multiband spin echo",
        "TE too short for b",
        "negative b = 0 count",
    ],
)
def test_an_impossible_spin_echo_prescription_is_refused(prescription):
    with pytest.raises(ValueError):
        scan2d(**prescription)


@pytest.mark.parametrize("b_value", [500e6, 2000e6])
def test_a_spin_echo_at_the_minimum_te_it_reports_is_designed(b_value):
    shortest = scan2d(spin_echo=True, diffusion_directions=6, b_value=b_value)
    longer = scan2d(
        spin_echo=True,
        diffusion_directions=6,
        b_value=b_value,
        te=round(shortest.echo_time, 4) + 1e-4,
    )
    with pytest.raises(ValueError, match=f"{shortest.echo_time * 1e3:.2f} ms"):
        scan2d(
            spin_echo=True,
            diffusion_directions=6,
            b_value=b_value,
            te=shortest.echo_time - 1e-3,
        )

    assert longer.echo_time >= shortest.echo_time


# -- inversion recovery ------------------------------------------------------

#: An interleaved inversion-recovery prescription: five slices in one TR.
INVERTED = {"n_x": 32, "n_y": 32, "n_slices": 5, "tr": 0.6, "n_dummy": 0}


def pulse_centres(seq, use):
    """The centre time and frequency offset of every pulse of ``use`` in ``seq``."""
    t, found = 0.0, []
    for i in range(1, len(seq.block_events) + 1):
        block = seq.get_block(i)
        rf = block.rf
        if rf is not None and rf.use == use:
            found.append((t + rf.delay + rf.center, rf.freq_offset))
        t += block.block_duration
    return found


def refusal(ti):
    """The message refusing ``ti`` for :data:`INVERTED`; ``None`` where it is played."""
    try:
        epi2d(**INVERTED, ti=ti)
    except ValueError as error:
        return str(error)
    return None


def test_every_acquired_slice_is_inverted_the_written_inversion_time_before_its_excitation():
    ti = next(t for t in np.arange(0.2, 0.5, 0.01) if refusal(t) is None)
    *_, seq = epi2d(**INVERTED, ti=ti)
    inversions = pulse_centres(seq, "inversion")
    excitations = pulse_centres(seq, "excitation")
    # Offsets rank the slices alike on both pulses, whose selection
    # gradients differ.
    rank_of = {
        use: {f: r for r, f in enumerate(sorted({f for _, f in found}))}
        for use, found in (("inversion", inversions), ("excitation", excitations))
    }
    written = float(np.atleast_1d(seq.definitions["TI"])[0])
    assert written == pytest.approx(ti, abs=1e-5)
    for time, offset in excitations[-INVERTED["n_slices"] :]:
        rank = rank_of["excitation"][offset]
        inverted = max(
            t for t, f in inversions if t < time and rank_of["inversion"][f] == rank
        )
        assert time - inverted == pytest.approx(written, abs=1e-6)


def test_an_inversion_time_that_overlaps_a_shot_is_refused_naming_ones_that_fit():
    refused = next(
        (t, message)
        for t in np.arange(0.2, 0.5, 0.01)
        if (message := refusal(t)) is not None
    )
    ti, message = refused
    suggested = [
        float(v) * 1e-3 for v in re.findall(r"(\d+\.\d) ms", message.split("fit")[1])
    ]
    assert suggested
    for t in suggested:
        assert refusal(t) is None, (ti, t)


@pytest.mark.parametrize(
    "prescription",
    [
        {"ti": 0.3, "n_shots": 2, "n_y": 32},
        {"ti": 0.3, "n_slices": 4, "multiband": 2},
        {"ti": 0.3, "inversion_slab": 0.0},
    ],
    ids=str,
)
def test_an_inversion_is_refused_where_it_is_not_designed(prescription):
    with pytest.raises(ValueError):
        scan2d(**prescription)
