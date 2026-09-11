"""PNS check: SAFE and chronaxie responses to the physical-axis slew."""

import copy
import math

import numpy as np
import pypulseq as upstream
import pytest
from pypulseq.utils.safe_pns_prediction import safe_example_hw

import pypulseqpp as pp
from pypulseqpp import safety
from pypulseqpp.safety import ChronaxieModel
from pypulseqpp.safety._pns import _pns

CHRONAXIE = ChronaxieModel(chronaxie=360e-6, rheobase=20.0, alpha=0.333)


def opts(raster=10e-6, library=pp):
    return library.Opts(
        max_grad=40,
        grad_unit="mT/m",
        max_slew=200,
        slew_unit="T/m/s",
        grad_raster_time=raster,
    )


@pytest.fixture
def system():
    return opts()


def played(system, *blocks):
    sequence = pp.Sequence(system)
    for block in blocks:
        sequence.add_block(*block)
    return sequence


def encoded(system):
    """Phase-encoded readouts with a sinusoidal gradient: trapezoids and a shape."""
    readout = pp.make_trapezoid("x", area=2000, system=system)
    n = 300
    wave = 1e5 * np.sin(np.pi * (np.arange(n) + 0.5) / n) ** 2
    shaped = pp.make_arbitrary_grad("z", wave, first=0.0, last=0.0, system=system)
    blocks = []
    for index in range(6):
        area = -1000 + 400 * index
        blocks.append([pp.make_trapezoid("y", area=area, duration=1e-3, system=system)])
        blocks.append([readout])
        blocks.append([shaped])
        blocks.append([pp.make_delay(2e-3)])
    return played(system, *blocks)


def trace(sequence, model, rotation=None):
    _, found = _pns(sequence, model, rotation, None, keep_trace=True)
    return found


def ramp(system, amplitude, rise, channel="x"):
    """One trapezoid whose rise is a rectangular slew of amplitude / rise."""
    return pp.make_trapezoid(
        channel,
        amplitude=amplitude * 1e-3 * system.gamma,
        rise_time=rise,
        fall_time=rise,
        flat_time=2e-3,
        system=system,
    )


# -- the two models ------------------------------------------------------------


def test_the_safe_response_is_upstreams(system, tmp_path):
    path = tmp_path / "encoded.seq"
    encoded(system).write(path)
    ours = pp.Sequence(system)
    ours.read(path)
    mirror = upstream.Sequence(system=opts(library=upstream))
    mirror.read(str(path))

    ok, norm, components, _ = mirror.calculate_pns(safe_example_hw(), do_plots=False)
    found = trace(ours, safe_example_hw())

    count = norm.size
    assert found["samples"] >= count
    for axis in range(3):
        np.testing.assert_allclose(
            found["trace_axes"][axis][:count],
            components[:, axis],
            rtol=1e-9,
            atol=1e-12 * components.max(),
        )
    assert found["norm"]["value"] == pytest.approx(norm.max(), rel=1e-9)
    assert (found["norm"]["value"] < 1.0) == ok


def test_the_chronaxie_response_is_the_convolution_over_the_whole_timeline(system):
    sequence = encoded(system)
    dt = system.grad_raster_time
    total = round(sequence.duration()[0] / dt)
    t = (np.arange(total) + 0.5) * dt
    c, s_min = CHRONAXIE.chronaxie, CHRONAXIE.rheobase / CHRONAXIE.alpha
    lags = np.arange(int(20 * c / dt) + 1) * dt
    kernel = c * dt / (s_min * (c + lags) * (c + lags + dt))

    found = trace(sequence, CHRONAXIE)

    for axis, channel in enumerate(sequence.waveforms()):
        g = np.interp(t, channel[0], channel[1], left=0.0, right=0.0)
        slew = np.diff(g, prepend=0.0) / dt / system.gamma
        expected = np.abs(np.convolve(slew, kernel)[:total])
        np.testing.assert_allclose(
            found["trace_axes"][axis], expected, rtol=1e-9, atol=1e-12
        )


def test_a_rectangular_slew_follows_the_strength_duration_curve(system):
    amplitude, rise = 20.0, 200e-6
    sequence = played(system, [ramp(system, amplitude, rise)])

    _, report = safety.check_pns(sequence, CHRONAXIE)

    slew = amplitude * 1e-3 / rise
    c = CHRONAXIE.chronaxie
    expected = slew * CHRONAXIE.alpha * rise / (CHRONAXIE.rheobase * (c + rise))
    assert report.peak.value == pytest.approx(expected, rel=0.02)


def test_the_chronaxie_response_does_not_move_when_the_raster_is_halved(system):
    fine = opts(5e-6)

    _, coarse = safety.check_pns(
        played(system, [ramp(system, 20.0, 200e-6)]), CHRONAXIE
    )
    _, halved = safety.check_pns(played(fine, [ramp(fine, 20.0, 200e-6)]), CHRONAXIE)

    assert halved.peak.value == pytest.approx(coarse.peak.value, rel=0.01)


def test_the_two_models_are_not_the_same_calculation(system):
    sequence = encoded(system)

    _, safe = safety.check_pns(sequence, safe_example_hw())
    _, chronaxie = safety.check_pns(sequence, CHRONAXIE)

    assert (safe.model, chronaxie.model) == ("safe", "chronaxie")
    assert safe.peak.value != pytest.approx(chronaxie.peak.value, rel=1e-3)


@pytest.mark.parametrize(
    "model", [safe_example_hw(), CHRONAXIE], ids=["safe", "chronaxie"]
)
def test_the_norm_is_the_root_sum_square_of_the_axes(system, model):
    found = trace(encoded(system), model)

    axes = np.stack(found["trace_axes"])
    np.testing.assert_allclose(found["trace_norm"], np.sqrt((axes**2).sum(axis=0)))


# -- the verdict ----------------------------------------------------------------


def test_a_response_over_threshold_is_refused(system):
    sequence = played(system, [ramp(system, 20.0, 200e-6)])

    within, report = safety.check_pns(sequence, CHRONAXIE)
    over, _ = safety.check_pns(
        sequence,
        CHRONAXIE._replace(rheobase=0.5 * report.peak.value * CHRONAXIE.rheobase),
    )

    assert within
    assert not over


def test_the_peak_names_the_block_that_plays_it(system):
    sequence = played(
        system,
        [ramp(system, 5.0, 400e-6)],
        [pp.make_delay(5e-3)],
        [ramp(system, 20.0, 100e-6, channel="y")],
        [pp.make_delay(5e-3)],
    )

    _, report = safety.check_pns(sequence, CHRONAXIE)

    starts = 2 * 400e-6 + 2e-3 + 5e-3
    assert report.peak.block == 3
    assert report.axes[1].block == 3
    assert report.axes[0].block == 1
    assert starts < report.peak.time < starts + 2 * 100e-6 + 2e-3


# -- axes and orientation --------------------------------------------------------


def test_a_turned_block_stimulates_the_axis_it_plays_on(system):
    on_x = ramp(system, 20.0, 200e-6, channel="x")
    on_y = ramp(system, 20.0, 200e-6, channel="y")

    _, turned = safety.check_pns(
        played(system, [on_x, pp.make_rotation(math.pi / 2)]), safe_example_hw()
    )
    _, plain = safety.check_pns(played(system, [on_y]), safe_example_hw())

    assert turned.axes[1].value == pytest.approx(plain.axes[1].value, rel=1e-9)
    assert turned.axes[0].value < 1e-9 * plain.axes[1].value


def test_the_prescription_is_turned_after_the_blocks_own_rotation(system):
    # The block turns x onto y; the prescription turns y onto z.
    on_x = ramp(system, 20.0, 200e-6, channel="x")
    on_z = ramp(system, 20.0, 200e-6, channel="z")
    prescription = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]

    _, turned = safety.check_pns(
        played(system, [on_x, pp.make_rotation(math.pi / 2)]),
        safe_example_hw(),
        rotation=prescription,
    )
    _, plain = safety.check_pns(played(system, [on_z]), safe_example_hw())

    assert turned.axes[2].value == pytest.approx(plain.axes[2].value, rel=1e-9)


# -- models from values and from files ----------------------------------------------


def test_a_mapping_stands_for_a_chronaxie_model(system):
    sequence = played(system, [ramp(system, 20.0, 200e-6)])

    _, named = safety.check_pns(sequence, CHRONAXIE)
    _, mapped = safety.check_pns(sequence, CHRONAXIE._asdict())

    assert mapped == named


def test_a_safe_model_reads_from_an_asc_file(system, tmp_path):
    hardware = safe_example_hw()
    lines = []
    for name in "xyz":
        axis = getattr(hardware, name)
        tag = name.upper()
        for i in range(3):
            lines.append(f"flGSWDTau{tag}[{i}] = {getattr(axis, f'tau{i + 1}')}")
            lines.append(f"flGSWDA{tag}[{i}] = {getattr(axis, f'a{i + 1}')}")
        lines.append(f"flGSWDStimulationLimit{tag} = {axis.stim_limit}")
        lines.append(f"flGSWDStimulationThreshold{tag} = {axis.stim_thresh}")
        lines.append(
            f"asGPAParameters[0].sGCParameters.flGScaleFactor{tag} = {axis.g_scale}"
        )
    path = tmp_path / "gradient.asc"
    path.write_text("\n".join(lines) + "\n")
    sequence = encoded(system)

    read = safety.read_safe_model(path)
    _, from_file = safety.check_pns(sequence, path)
    _, from_values = safety.check_pns(sequence, hardware)

    assert read.y.tau2 == hardware.y.tau2
    assert read.z.g_scale == hardware.z.g_scale
    assert from_file == from_values


def test_an_inconsistent_safe_model_is_refused(system):
    hardware = copy.deepcopy(safe_example_hw())
    hardware.y.a1 += 0.1

    with pytest.raises(ValueError, match=r"a1 \+ a2 \+ a3"):
        safety.check_pns(pp.Sequence(system), hardware)


def test_a_chronaxie_model_needs_positive_coefficients(system):
    with pytest.raises(ValueError, match="positive"):
        safety.check_pns(pp.Sequence(system), ChronaxieModel(0.0, 20.0))


def test_what_is_no_nerve_model_is_refused(system):
    with pytest.raises(TypeError, match="ChronaxieModel"):
        safety.check_pns(pp.Sequence(system), system)


def test_nothing_played_stimulates_nothing(system):
    ok, report = safety.check_pns(pp.Sequence(system), CHRONAXIE)

    assert ok
    assert report.samples == 0
    assert report.peak.value == 0.0
