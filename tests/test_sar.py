"""VOP SAR: time-averaged local and global SAR over each real repetition."""

import math

import numpy as np
import pytest
from scipy.io import savemat

import pypulseqpp as pp
from pypulseqpp import safety
from pypulseqpp.safety import VopModel

CHANNELS = 4


@pytest.fixture
def system():
    return pp.Opts(rf_ringdown_time=20e-6, rf_dead_time=100e-6, adc_dead_time=10e-6)


@pytest.fixture
def model():
    """Random Hermitian positive semidefinite VOPs and a global matrix."""
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(5, CHANNELS, CHANNELS)) + 1j * rng.normal(
        size=(5, CHANNELS, CHANNELS)
    )
    vops = raw @ np.conj(np.swapaxes(raw, -1, -2)) * 1e-3
    return VopModel(vops, vops.mean(axis=0) * 0.3)


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


def test_a_prologue_is_a_window_of_its_own(system, model):
    seq = pp.Sequence(system)
    seq.add_block(pp.make_block_pulse(math.pi, duration=1e-3, system=system))
    seq.add_block(pp.make_delay(20e-3))
    body = pp.make_sinc_pulse(math.pi / 6, duration=1e-3, system=system)
    readout = pp.make_trapezoid("x", area=1000, duration=1e-3, system=system)
    for _ in range(4):
        seq.add_block(body)
        seq.add_block(readout)
        seq.add_block(pp.make_delay(5e-3))

    _, report = safety.check_sar(seq, model, drive_per_hz=1.0)

    assert report.tr_start > 1
    assert report.windows.first[0] == 1
    assert report.windows.last[0] == report.tr_start - 1


def test_a_sequence_that_does_not_repeat_is_one_window(system, model):
    seq = pp.Sequence(system)
    seq.add_block(pp.make_block_pulse(math.pi / 2, duration=1e-3, system=system))
    seq.add_block(pp.make_trapezoid("x", area=1000, duration=2e-3, system=system))

    _, report = safety.check_sar(seq, model, drive_per_hz=1.0)

    assert report.tr_size == 0
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
