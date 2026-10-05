"""VOP SAR: time-averaged local and global SAR over each real repetition."""

import math

import numpy as np
import pytest
from scipy.io import savemat

import pypulseqpp as pp
from pypulseqpp import _ext, safety
from pypulseqpp.safety import VopModel

CHANNELS = 4


@pytest.fixture
def system():
    return pp.Opts(rf_ringdown_time=20e-6, rf_dead_time=100e-6, adc_dead_time=10e-6)


@pytest.fixture
def model():
    """Random Hermitian positive semidefinite VOPs, uncompressed, and a global matrix."""
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(5, CHANNELS, CHANNELS)) + 1j * rng.normal(
        size=(5, CHANNELS, CHANNELS)
    )
    vops = raw @ np.conj(np.swapaxes(raw, -1, -2)) * 1e-3
    return VopModel(vops, vops.mean(axis=0) * 0.3, cores=vops)


def resampled(rf, dt=1e-6):
    """Per-channel waveform on the calc_rf_power grid, (channels, samples)."""
    t = np.asarray(rf.t, dtype=float)
    signal = np.asarray(rf.signal)
    grid = (np.arange(int(np.round(rf.shape_dur / dt))) + 0.5) * dt
    channels = int(np.count_nonzero(t == t[0]))
    channels = channels if channels > 1 and t.size % channels == 0 else 1
    per = t.size // channels
    return np.vstack(
        [
            np.interp(
                grid,
                t[c * per : (c + 1) * per],
                signal[c * per : (c + 1) * per],
                left=0,
                right=0,
            )
            for c in range(channels)
        ]
    )


def plain_sar(seq, model, blocks, drive, shims, default_shim, dt=1e-6):
    """Local (per VOP) and global SAR of one window, by direct summation."""
    local = np.zeros(len(model.vops))
    whole = 0.0
    duration = 0.0
    for number in blocks:
        block = seq.get_block(number)
        duration += block.block_duration
        if block.rf is None:
            continue
        waves = resampled(block.rf, dt)
        shim = shims.get(number)
        if shim is None:
            shim = default_shim if waves.shape[0] == 1 else np.ones(CHANNELS)
        if waves.shape[0] == 1:
            waves = np.repeat(waves, CHANNELS, axis=0)
        v = (drive * shim)[:, None] * waves
        local += np.einsum("in,kij,jn->k", v.conj(), model.vops, v).real * dt
        whole += np.einsum("in,ij,jn->", v.conj(), model.global_matrix, v).real * dt
    return local / duration, whole / duration


def shimmed(system, repeats=5):
    """Repetitions of a shimmed pulse that grows, an unshimmed one and a pTx pulse.

    Every repetition plays the same shapes, so the definitions repeat from the
    first block; only the shimmed pulse's amplitude changes.
    """
    rng = np.random.default_rng(3)
    waveform = rng.normal(size=(CHANNELS, 200)) + 1j * rng.normal(size=(CHANNELS, 200))
    ptx = pp.make_ptx_pulse(100.0 * waveform, system=system)
    weak = pp.make_sinc_pulse(0.3 * math.pi / 4, duration=1e-3, system=system)
    shim = np.exp(1j * 0.7 * np.arange(CHANNELS)) * np.linspace(0.5, 1.2, CHANNELS)

    seq = pp.Sequence(system)
    shims = {}
    for repeat in range(repeats):
        grown = pp.make_sinc_pulse(
            (0.5 + 0.1 * repeat) * math.pi / 4, duration=1e-3, system=system
        )
        seq.add_block(grown, pp.make_rf_shim(shim))
        shims[seq.num_blocks] = shim
        seq.add_block(pp.make_delay(2e-3))
        seq.add_block(weak)
        seq.add_block(ptx)
        seq.add_block(pp.make_delay(5e-3))
    # Each factory call registers its own shapes; collapsing them is what lets
    # the definitions, and so the repetition, show.
    seq.remove_duplicates(in_place=True)
    return seq, shims


# -- the native path is the plain one ----------------------------------------------


def test_the_native_sar_is_the_plain_sum(system, model):
    seq, shims = shimmed(system)
    drive = np.linspace(0.8, 1.3, CHANNELS)
    default = np.exp(-1j * np.arange(CHANNELS))

    _, report = safety.check_sar(seq, model, drive_per_hz=drive, default_shim=default)

    windows = report.windows
    assert report.tr_size == 5
    for w in range(windows.first.size):
        blocks = range(int(windows.first[w]), int(windows.last[w]) + 1)
        local, whole = plain_sar(seq, model, blocks, drive, shims, default)
        assert windows.local_sar[w] == pytest.approx(local.max(), rel=1e-9)
        assert windows.vop[w] == int(np.argmax(local))
        assert windows.global_sar[w] == pytest.approx(whole, rel=1e-9)


def test_the_global_sar_is_the_largest_over_the_body_models(system, model):
    seq, _ = shimmed(system)
    population = VopModel(
        model.vops,
        np.stack(
            [0.5 * model.global_matrix, 2 * model.global_matrix, model.global_matrix]
        ),
    )

    _, one = safety.check_sar(seq, model, drive_per_hz=1.0)
    _, many = safety.check_sar(seq, population, drive_per_hz=1.0)

    np.testing.assert_allclose(
        many.windows.global_sar, 2 * one.windows.global_sar, rtol=1e-9
    )
    assert many.worst_global.body == 1
    assert many.windows.global_body.tolist() == [1] * many.windows.first.size


def test_sar_goes_with_the_square_of_the_drive(system, model):
    seq, _ = shimmed(system)

    _, once = safety.check_sar(seq, model, drive_per_hz=1.0)
    _, twice = safety.check_sar(seq, model, drive_per_hz=2.0)

    np.testing.assert_allclose(twice.windows.local_sar, 4 * once.windows.local_sar)


def test_the_worst_repetition_decides(system, model):
    seq, _ = shimmed(system)

    _, report = safety.check_sar(seq, model, drive_per_hz=1.0)

    # The shimmed pulse grows repetition by repetition, so the last is worst.
    assert report.worst_local.window == report.windows.first.size - 1
    assert report.worst_local.sar == pytest.approx(report.windows.local_sar.max())


def test_checking_sar_writes_no_repeating_unit_into_the_sequence(system, model):
    seq, _ = shimmed(system)

    _, report = safety.check_sar(seq, model, drive_per_hz=1.0)

    assert report.tr_size == seq.repetition()[0]
    assert seq.get_definition("TRSize") == ""


def test_the_blocks_before_the_first_repetition_are_a_window_of_their_own(
    system, model
):
    """The windows follow the repetition they are given; here it starts at
    block 3, so blocks 1 and 2 are a window of their own."""
    body = pp.make_sinc_pulse(math.pi / 6, duration=1e-3, system=system)
    seq = pp.Sequence(system)
    seq.add_block(pp.make_block_pulse(math.pi, duration=1e-3, system=system))
    seq.add_block(pp.make_delay(20e-3))
    for _ in range(4):
        seq.add_block(body)
        seq.add_block(pp.make_delay(5e-3))

    found = _ext.vop_sar(
        seq._native,
        vops=np.ascontiguousarray(model.vops),
        drive=np.ones(CHANNELS),
        default_shim=np.ones(CHANNELS, dtype=complex),
        size=2,
        start=3,
    )

    assert found["first"].tolist() == [1, 3, 5, 7, 9]
    assert found["last"].tolist() == [2, 4, 6, 8, 10]


def test_a_sequence_that_does_not_repeat_is_one_window(system, model):
    seq = pp.Sequence(system)
    seq.add_block(pp.make_block_pulse(math.pi / 2, duration=1e-3, system=system))
    seq.add_block(pp.make_trapezoid("x", area=1000, duration=2e-3, system=system))

    _, report = safety.check_sar(seq, model, drive_per_hz=1.0)

    assert report.windows.first.tolist() == [1]
    assert report.windows.last.tolist() == [2]


def test_the_limits_decide_the_verdict(system, model):
    seq, _ = shimmed(system)
    _, report = safety.check_sar(seq, model, drive_per_hz=1.0)
    worst = report.worst_local.sar

    within, _ = safety.check_sar(
        seq, model, drive_per_hz=1.0, local_limit=2 * worst, global_limit=1e9
    )
    over, _ = safety.check_sar(
        seq, model, drive_per_hz=1.0, local_limit=0.5 * worst, global_limit=1e9
    )

    assert within
    assert not over


def test_a_pulse_on_the_wrong_number_of_channels_is_refused(system, model):
    seq = pp.Sequence(system)
    seq.add_block(pp.make_ptx_pulse(np.ones((3, 50)) * 100.0, system=system))

    with pytest.raises(ValueError, match="channels"):
        safety.check_sar(seq, model, drive_per_hz=1.0)


# -- VOP files -----------------------------------------------------------------------


def test_a_mat_file_stack_is_read_batch_first(tmp_path, model):
    path = tmp_path / "vops.mat"
    savemat(
        path,
        {"VOP": np.transpose(model.vops, (1, 2, 0)), "Sglobal": model.global_matrix},
    )

    read = safety.read_vops(path)

    np.testing.assert_allclose(read.vops, model.vops)
    np.testing.assert_allclose(read.global_matrix, model.global_matrix)


def test_an_npz_file_round_trips(tmp_path, model):
    path = tmp_path / "vops.npz"
    np.savez(path, vops=model.vops, global_matrix=model.global_matrix)

    read = safety.read_vops(path)

    np.testing.assert_allclose(read.vops, model.vops)


def test_a_one_body_npz_file_as_mariepy_writes_it_is_read(tmp_path, model):
    path = tmp_path / "vops.npz"
    np.savez_compressed(
        path,
        vops=model.vops,
        global_matrix=model.global_matrix[None],
        metadata=np.array('{"bodies": ["head"]}'),
    )

    read = safety.read_vops(path)

    np.testing.assert_allclose(read.vops, model.vops)
    np.testing.assert_allclose(read.global_matrix, model.global_matrix[None])
    assert read.metadata["bodies"] == ["head"]


def test_a_population_file_keeps_one_global_matrix_per_body(tmp_path, model):
    path = tmp_path / "vops.npz"
    bodies = np.stack([model.global_matrix, 2 * model.global_matrix])
    np.savez(
        path,
        vops=model.vops,
        global_matrix=bodies,
        metadata=np.array('{"bodies": ["subject04", "subject05"]}'),
    )

    read = safety.read_vops(path)

    np.testing.assert_allclose(read.global_matrix, bodies)
    assert read.metadata["bodies"] == ["subject04", "subject05"]


def test_a_mat_file_that_counts_the_points_first_is_read_as_it_stands(tmp_path, model):
    path = tmp_path / "vops.mat"
    savemat(path, {"VOPm": model.vops})

    read = safety.read_vops(path)

    np.testing.assert_allclose(read.vops, model.vops)


def test_a_mat_file_of_as_many_points_as_channels_is_read_in_matlab_order(
    tmp_path, model
):
    points = model.vops[:CHANNELS]
    path = tmp_path / "vops.mat"
    savemat(path, {"VOP": np.transpose(points, (1, 2, 0))})

    read = safety.read_vops(path)

    np.testing.assert_allclose(read.vops, points)


def test_an_npz_file_of_as_many_points_as_channels_is_read_point_first(tmp_path, model):
    points = model.vops[:CHANNELS]
    path = tmp_path / "vops.npz"
    np.savez(path, vops=points)

    read = safety.read_vops(path)

    np.testing.assert_allclose(read.vops, points)


def test_a_file_that_holds_no_stack_of_square_matrices_is_refused(tmp_path):
    path = tmp_path / "vops.npz"
    np.savez(path, vops=np.zeros((3, 4, 5)))

    with pytest.raises(ValueError, match="stack of square matrices"):
        safety.read_vops(path)


def test_a_global_matrix_on_the_wrong_number_of_channels_is_refused(system, model):
    with pytest.raises(ValueError, match="channels"):
        safety.check_sar(
            pp.Sequence(system),
            VopModel(model.vops, np.eye(CHANNELS + 1, dtype=complex)),
            drive_per_hz=1.0,
        )


def test_a_vop_that_is_not_hermitian_is_refused(system, model):
    broken = model.vops.copy()
    broken[0, 0, 1] += 1.0

    with pytest.raises(ValueError, match="Hermitian"):
        safety.check_sar(pp.Sequence(system), VopModel(broken), drive_per_hz=1.0)


# -- the synthetic example ------------------------------------------------------------


def test_the_example_vops_are_positive_semidefinite():
    example = safety.example_vops(8)

    vops = example.model.vops
    assert vops.shape[1:] == (8, 8)
    assert np.linalg.eigvalsh(vops).min() > -1e-12 * np.abs(vops).max()
    assert np.linalg.eigvalsh(example.model.global_matrix).min() > -1e-12


def test_the_example_global_sar_is_below_its_worst_local_sar(system):
    example = safety.example_vops(8)
    seq = pp.Sequence(system)
    seq.add_block(pp.make_block_pulse(math.pi / 2, duration=1e-3, system=system))
    seq.add_block(pp.make_delay(9e-3))

    _, report = safety.check_sar(
        seq,
        example.model,
        drive_per_hz=example.drive_per_hz,
        default_shim=example.cp_shim,
    )

    assert 0 < report.worst_global.sar < report.worst_local.sar


# -- comparing with a reference ------------------------------------------------------


def cp_fid(system, tr=10e-3):
    """The CP-mode reference: one 1 ms 180 degree hard pulse per repetition."""
    rf = pp.make_block_pulse(math.pi, duration=1e-3, system=system)
    seq = pp.Sequence(system)
    seq.add_block(rf)
    seq.add_block(pp.make_delay(tr - pp.calc_duration(rf)))
    return seq


def test_a_sequence_compared_with_itself_scales_by_one(system, model):
    fid = cp_fid(system)

    _, report = safety.check_sar(fid, model, drive_per_hz=1.0, reference=fid)

    assert report.reference.sar_ratio == pytest.approx(1.0)
    assert report.reference.energy_ratio == pytest.approx(1.0)
    assert report.reference.global_sar_ratio == pytest.approx(1.0)


def test_a_cp_only_sequence_scales_by_its_rf_energy_whatever_the_vops(system, model):
    cp = np.exp(0.5j * np.pi * np.arange(CHANNELS))
    body = pp.make_sinc_pulse(math.pi / 3, duration=2e-3, system=system)
    seq = pp.Sequence(system)
    for _ in range(3):
        seq.add_block(body)
        seq.add_block(pp.make_delay(6e-3))
    fid = cp_fid(system)

    _, report = safety.check_sar(
        seq, model, drive_per_hz=1.0, default_shim=cp, reference=fid
    )

    energy = seq.calc_rf_power(block_range=(1, 2))[3] / fid.calc_rf_power()[3]
    assert report.reference.energy_ratio == pytest.approx(energy, rel=1e-9)
    assert report.reference.global_energy_ratio == pytest.approx(energy, rel=1e-9)
    # Every VOP sees the same increase.
    assert report.windows.reference_ratio == pytest.approx(
        np.full(3, report.reference.sar_ratio), rel=1e-9
    )


def test_the_ratio_is_of_peak_local_sar_not_of_each_vop(system):
    # One VOP sees channel 0 alone; the other all channels in phase, which the
    # reference heats four times as much.
    alone = np.zeros((2, 2), dtype=complex)
    alone[0, 0] = 1.0
    stack = np.stack([alone, np.ones((2, 2), dtype=complex)])
    two = VopModel(stack, cores=stack)
    fid = cp_fid(system)
    targeted = pp.Sequence(system)
    rf = pp.make_block_pulse(math.pi, duration=1e-3, system=system)
    targeted.add_block(rf, pp.make_rf_shim([1.0, 0.0]))
    targeted.add_block(pp.make_delay(10e-3 - pp.calc_duration(rf)))

    _, reference = safety.check_sar(fid, two, drive_per_hz=1.0)
    _, report = safety.check_sar(targeted, two, drive_per_hz=1.0, reference=fid)

    assert report.worst_local.sar / reference.worst_local.sar == pytest.approx(0.25)
    assert report.reference.sar_ratio == pytest.approx(0.25)


def test_the_global_ratio_is_taken_body_by_body(system):
    # The first body sees channel 0 alone; the second all channels in phase,
    # which the reference heats four times as much. The targeted pulse heats
    # the first body as much as the reference does.
    alone = np.zeros((2, 2), dtype=complex)
    alone[0, 0] = 1.0
    stack = np.stack([alone, np.ones((2, 2), dtype=complex)])
    two = VopModel(stack, stack, cores=stack)
    fid = cp_fid(system)
    targeted = pp.Sequence(system)
    rf = pp.make_block_pulse(math.pi, duration=1e-3, system=system)
    targeted.add_block(rf, pp.make_rf_shim([1.0, 0.0]))
    targeted.add_block(pp.make_delay(10e-3 - pp.calc_duration(rf)))

    _, from_sequence = safety.check_sar(targeted, two, drive_per_hz=1.0, reference=fid)
    _, earlier = safety.check_sar(fid, two, drive_per_hz=1.0)
    _, from_report = safety.check_sar(
        targeted, two, drive_per_hz=1.0, reference=earlier
    )

    assert from_sequence.reference.global_sar_ratio == pytest.approx(1.0)
    assert from_report.reference.global_sar_ratio == pytest.approx(1.0)
    assert from_sequence.reference.sar_ratio == pytest.approx(0.25)


def compressed(model, margin):
    """Points of an overestimation margin times the largest eigenvalue, and their cores."""
    matrices = model.vops
    largest = np.linalg.eigvalsh(matrices)[:, -1]
    added = margin * largest.max() * np.eye(matrices.shape[-1])
    left = list(np.argsort(-largest))
    points, cores = [], []
    while left:
        core = matrices[left[0]]
        point = core + added
        left = [
            index
            for index in left
            if np.linalg.eigvalsh(point - matrices[index])[0] < -1e-12 * largest.max()
        ]
        points.append(point)
        cores.append(core)
    return np.stack(points), np.stack(cores)


def shim_trains(system, count, seed):
    """Trains of one hard pulse in a random RF shim, as many as asked."""
    rng = np.random.default_rng(seed)
    rf = pp.make_block_pulse(math.pi / 2, duration=1e-3, system=system)
    for _ in range(count):
        weights = rng.normal(size=CHANNELS) + 1j * rng.normal(size=CHANNELS)
        seq = pp.Sequence(system)
        seq.add_block(rf, pp.make_rf_shim(weights / np.linalg.norm(weights) * 2.0))
        seq.add_block(pp.make_delay(10e-3 - pp.calc_duration(rf)))
        yield seq


def test_a_compressed_model_never_gives_a_ratio_below_the_exact_one(system):
    exact = safety.example_vops(CHANNELS)
    points, cores = compressed(exact.model, 0.25)
    model = VopModel(points, exact.model.global_matrix, cores=cores)
    fid = cp_fid(system)
    drive = {"drive_per_hz": exact.drive_per_hz, "default_shim": exact.cp_shim}

    for seq in shim_trains(system, 40, seed=5):
        _, bound = safety.check_sar(seq, model, reference=fid, **drive)
        _, truth = safety.check_sar(seq, exact.model, reference=fid, **drive)
        assert bound.reference.sar_ratio >= truth.reference.sar_ratio * (1 - 1e-9)


def test_points_over_points_can_fall_below_the_exact_ratio(system):
    """The reason the reference's side is read through the cores."""
    exact = safety.example_vops(CHANNELS)
    points, _ = compressed(exact.model, 0.25)
    own = VopModel(points, cores=points)
    fid = cp_fid(system)
    drive = {"drive_per_hz": exact.drive_per_hz, "default_shim": exact.cp_shim}

    def ratio(seq, model):
        _, report = safety.check_sar(seq, model, reference=fid, **drive)
        return report.reference.sar_ratio

    trains = shim_trains(system, 40, seed=5)
    assert any(ratio(seq, own) < ratio(seq, exact.model) for seq in trains)


def test_a_ratio_needs_the_cores(system, model):
    bare = VopModel(model.vops, model.global_matrix)
    fid = cp_fid(system)

    with pytest.raises(ValueError, match="lower bound"):
        safety.check_sar(fid, bare, drive_per_hz=1.0, reference=fid)


def test_a_report_on_compressed_points_cannot_stand_for_its_sequence(system, model):
    points, cores = compressed(model, 0.25)
    squeezed = VopModel(points, cores=cores)
    fid = cp_fid(system)
    _, earlier = safety.check_sar(fid, squeezed, drive_per_hz=1.0)

    assert earlier.worst_local.floor is None
    with pytest.raises(ValueError, match="pass the reference sequence"):
        safety.check_sar(fid, squeezed, drive_per_hz=1.0, reference=earlier)


def test_a_vop_that_does_not_dominate_its_core_is_refused(system, model):
    with pytest.raises(ValueError, match="dominate its core"):
        safety.check_sar(
            cp_fid(system), VopModel(model.vops, cores=2 * model.vops), drive_per_hz=1.0
        )


def test_the_safety_factor_multiplies_local_sar_and_its_ratio_not_global(system, model):
    seq, _ = shimmed(system)
    fid = cp_fid(system)
    marked = model._replace(metadata={"safety_factor": 2.0})

    _, plain = safety.check_sar(seq, model, drive_per_hz=1.0, reference=fid)
    _, factored = safety.check_sar(seq, marked, drive_per_hz=1.0, reference=fid)

    assert factored.safety_factor == 2.0
    assert factored.worst_local.sar == pytest.approx(2.0 * plain.worst_local.sar)
    assert factored.reference.sar_ratio == pytest.approx(
        2.0 * plain.reference.sar_ratio
    )
    assert factored.worst_global.sar == pytest.approx(plain.worst_global.sar)
    assert factored.reference.global_sar_ratio == pytest.approx(
        plain.reference.global_sar_ratio
    )


def test_a_given_safety_factor_takes_the_place_of_the_model_s(system, model):
    marked = model._replace(metadata={"safety_factor": 2.0})
    fid = cp_fid(system)

    _, report = safety.check_sar(fid, marked, drive_per_hz=1.0, safety_factor=3.0)

    assert report.safety_factor == 3.0


@pytest.mark.parametrize("factor", [0.5, float("nan")])
def test_a_safety_factor_below_one_is_refused(system, model, factor):
    with pytest.raises(ValueError, match="at least 1"):
        safety.check_sar(cp_fid(system), model, drive_per_hz=1.0, safety_factor=factor)


def test_a_file_s_cores_and_safety_factor_are_read(tmp_path, model):
    points, cores = compressed(model, 0.25)
    path = tmp_path / "vops.npz"
    np.savez(
        path,
        vops=points,
        cores=cores,
        global_matrix=model.global_matrix[None],
        metadata=np.array('{"safety_factor": 1.5}'),
    )

    read = safety.read_vops(path)

    np.testing.assert_allclose(read.cores, cores)
    assert read.metadata["safety_factor"] == 1.5


def test_a_report_can_stand_for_its_sequence(system, model):
    seq, _ = shimmed(system)
    fid = cp_fid(system)
    _, earlier = safety.check_sar(fid, model, drive_per_hz=1.0)

    _, from_sequence = safety.check_sar(seq, model, drive_per_hz=1.0, reference=fid)
    _, from_report = safety.check_sar(seq, model, drive_per_hz=1.0, reference=earlier)

    assert from_report.reference == from_sequence.reference


def test_the_ratios_do_not_depend_on_the_drive_calibration(system, model):
    seq, _ = shimmed(system)
    fid = cp_fid(system)

    _, once = safety.check_sar(seq, model, drive_per_hz=1.0, reference=fid)
    _, thrice = safety.check_sar(seq, model, drive_per_hz=3.0, reference=fid)

    assert thrice.reference.sar_ratio == pytest.approx(once.reference.sar_ratio)
    assert thrice.reference.energy_ratio == pytest.approx(once.reference.energy_ratio)
