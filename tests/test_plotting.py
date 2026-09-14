"""The plot subpackage, and the rasters and storage names a Sequence answers for."""

import math

import matplotlib
import numpy as np
import pytest

import pypulseqpp as pp
from pypulseqpp import plot
from pypulseqpp.plot._kspace import sampling_order

matplotlib.use("Agg")
import matplotlib.pyplot as plt

THICKNESS = 5e-3


@pytest.fixture
def system():
    return pp.Opts(
        max_grad=32,
        grad_unit="mT/m",
        max_slew=130,
        slew_unit="T/m/s",
        rf_ringdown_time=20e-6,
        rf_dead_time=100e-6,
        adc_dead_time=10e-6,
    )


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def gre(system, lines=16, fov=0.25):
    """A slice-selective gradient echo; one encode template scaled per line."""
    rf, gz, gzr = pp.make_sinc_pulse(
        math.pi / 12,
        duration=2e-3,
        slice_thickness=THICKNESS,
        apodization=0.5,
        time_bw_product=4,
        system=system,
        return_gz=True,
        use="excitation",
    )
    gx = pp.make_trapezoid("x", flat_area=lines / fov, flat_time=2e-3, system=system)
    adc = pp.make_adc(lines, duration=gx.flat_time, delay=gx.rise_time, system=system)
    gxpre = pp.make_trapezoid("x", area=-gx.area / 2, duration=1e-3, system=system)
    encode = pp.make_trapezoid(
        "y", area=(lines / 2) / fov, duration=1e-3, system=system
    )
    seq = pp.Sequence(system)
    for line in range(lines):
        seq.add_block(rf, gz)
        seq.add_block(
            gxpre, pp.scale_grad(encode, (line - lines / 2) / (lines / 2)), gzr
        )
        seq.add_block(gx, adc)
        seq.add_block(pp.make_delay(1e-3))
    return seq


def spin_echo(system):
    rf, gz, _ = pp.make_sinc_pulse(
        math.pi / 2,
        duration=2e-3,
        slice_thickness=THICKNESS,
        system=system,
        return_gz=True,
        use="excitation",
    )
    rf180, gz180, _ = pp.make_sinc_pulse(
        math.pi,
        duration=2e-3,
        slice_thickness=THICKNESS,
        system=system,
        return_gz=True,
        use="refocusing",
    )
    adc = pp.make_adc(64, duration=3e-3, system=system)
    seq = pp.Sequence(system)
    seq.add_block(rf, gz)
    seq.add_block(pp.make_delay(2e-3))
    seq.add_block(rf180, gz180)
    seq.add_block(adc)
    return seq


def radial(system, spokes=8, fov=0.25, samples=32):
    gx = pp.make_trapezoid("x", flat_area=samples / fov, flat_time=2e-3, system=system)
    adc = pp.make_adc(samples, duration=gx.flat_time, delay=gx.rise_time, system=system)
    pre = pp.make_trapezoid("x", area=-gx.area / 2, duration=1e-3, system=system)
    rf = pp.make_block_pulse(math.pi / 18, duration=0.3e-3, system=system)
    seq = pp.Sequence(system)
    for spoke in range(spokes):
        turn = pp.make_rotation(math.pi * spoke / spokes)
        seq.add_block(rf)
        seq.add_block(pre, turn)
        seq.add_block(gx, adc, turn)
    return seq


def scattered(figure):
    return [axis for axis in figure.axes if axis.get_xlabel().startswith("$k_")]


# -- the functional interface ---------------------------------------------------


def test_the_functional_plot_is_the_method(system, monkeypatch):
    seq = gre(system)
    calls = []
    monkeypatch.setattr(
        pp.Sequence, "plot", lambda self, *a, **k: calls.append((self, a, k)) or "drawn"
    )

    assert plot.plot(seq, tr_range=(1, 2)) == "drawn"
    assert calls == [(seq, (), {"tr_range": (1, 2)})]


def test_the_functional_paper_plot_is_the_method(system):
    seq = gre(system)

    functional = plot.paper_plot(seq, max_underlays=4)
    method = seq.paper_plot(max_underlays=4)

    assert (functional.tr, functional.underlays) == (method.tr, method.underlays)


def test_no_new_plotting_method_is_added_to_the_sequence():
    assert not hasattr(pp.Sequence, "plot_kspace")
    assert not hasattr(pp.Sequence, "plot_rf")


# -- k-space ------------------------------------------------------------------------


def test_a_planar_trajectory_is_drawn_in_its_plane(system):
    figure = plot.plot_kspace(gre(system), plot_now=False)

    (axis,) = scattered(figure)
    assert (axis.get_xlabel(), axis.get_ylabel()) == ("$k_x$ [1/m]", "$k_y$ [1/m]")


def test_the_drawn_path_is_the_calculated_trajectory(system):
    seq = gre(system)

    drawn = plot.plot_kspace(seq, plot_now=False).axes[0].lines[0].get_xydata()

    dense = np.asarray(seq.calculate_kspace()[1])
    assert np.array_equal(np.nan_to_num(drawn.T), np.nan_to_num(dense[:2]))


def test_the_samples_drawn_are_the_calculated_ones(system):
    seq = gre(system)

    offsets = scattered(plot.plot_kspace(seq, plot_now=False))[0].collections[0]

    samples = np.asarray(seq.calculate_kspace()[0])
    np.testing.assert_array_equal(offsets.get_offsets(), samples[:2].T)


def test_a_rotated_sequence_needs_no_special_argument(system):
    seq = radial(system)

    figure = plot.plot_kspace(seq, plot_now=False)

    samples = scattered(figure)[0].collections[0].get_offsets()
    # Spokes turned about z fill both in-plane axes.
    assert np.ptp(samples[:, 0]) > 0 and np.ptp(samples[:, 1]) > 0


def test_a_range_draws_only_its_samples(system):
    figure = plot.plot_kspace(gre(system, lines=16), tr_range=(1, 2), plot_now=False)

    assert len(scattered(figure)[0].collections[0].get_offsets()) == 2 * 16


def test_the_sampling_order_has_one_value_per_sample(system):
    seq = gre(system)

    shot, echo = sampling_order(seq, 1, seq.num_blocks)

    samples = np.asarray(seq.calculate_kspace()[0])
    assert shot.shape == echo.shape == (samples.shape[1],)


def test_colouring_draws_the_same_samples_as_not_colouring(system):
    plain = plot.plot_kspace(radial(system), plot_now=False)
    coloured = plot.plot_kspace(radial(system), color_by="shot", plot_now=False)

    assert np.array_equal(
        scattered(plain)[0].collections[0].get_offsets(),
        scattered(coloured)[0].collections[0].get_offsets(),
    )


def test_the_order_view_drops_a_panel_whose_index_never_varies(system):
    """One line per excitation has no echo axis, so there is no echo panel."""
    figure = plot.plot_kspace(gre(system), color_by="order", plot_now=False)

    assert len(scattered(figure)) == 1


def test_an_unknown_colour_index_is_refused(system):
    with pytest.raises(ValueError, match="color_by"):
        plot.plot_kspace(gre(system), color_by="slice", plot_now=False)


# -- RF profiles ---------------------------------------------------------------------


def test_a_pulse_is_named_by_the_job_it_does(system):
    seq = spin_echo(system)

    excitation = plot.plot_rf(seq, "excitation", plot_now=False)
    refocusing = plot.plot_rf(seq, "refocusing", plot_now=False)

    assert excitation.axes[1].get_ylabel() == r"$|M_{xy}|$"
    assert refocusing.axes[1].get_ylabel() == "refocusing efficiency"
    assert (
        excitation.axes[1].get_xlabel() == refocusing.axes[1].get_xlabel() == "z [mm]"
    )


def test_a_pulse_no_one_plays_is_refused(system):
    with pytest.raises(ValueError, match="no pulse used for 'inversion'"):
        plot.plot_rf(spin_echo(system), "inversion", plot_now=False)


def test_the_slice_profile_is_as_thick_as_the_pulse_was_designed(system):
    figure = plot.plot_rf(gre(system), plot_now=False)

    position, magnitude = figure.axes[1].lines[-1].get_xydata().T
    inside = position[magnitude >= 0.5 * magnitude.max()]
    assert inside.max() - inside.min() == pytest.approx(THICKNESS * 1e3, rel=0.1)


def test_a_bare_pulse_is_drawn_against_off_resonance(system):
    rf = pp.make_block_pulse(math.pi / 2, duration=0.5e-3, system=system)

    figure = plot.plot_rf(rf, plot_now=False)

    assert figure.axes[1].get_xlabel() == "off-resonance [Hz]"


def test_a_plane_is_drawn_as_two_heatmaps(system):
    figure = plot.plot_rf(gre(system), plane="zf", samples=31, plot_now=False)

    images = [image for axis in figure.axes for image in axis.get_images()]
    assert len(images) == 2
    assert images[0].get_array().shape == (31, 31)


def test_a_profile_along_an_axis_the_pulse_ignores_needs_the_window(system):
    with pytest.raises(ValueError, match="whole=True"):
        plot.plot_rf(gre(system), plane="x", plot_now=False)


# -- what a Sequence answers for ----------------------------------------------------------


def test_the_rasters_are_the_ones_the_sequence_was_designed_with():
    system = pp.Opts(
        grad_raster_time=20e-6,
        rf_raster_time=2e-6,
        adc_raster_time=200e-9,
        block_duration_raster=20e-6,
    )

    seq = pp.Sequence(system)

    assert seq.grad_raster_time == pytest.approx(20e-6)
    assert seq.rf_raster_time == pytest.approx(2e-6)
    assert seq.adc_raster_time == pytest.approx(200e-9)
    assert seq.block_duration_raster == pytest.approx(20e-6)


def test_a_file_carries_its_rasters(system, tmp_path):
    other = pp.Opts(grad_raster_time=20e-6, block_duration_raster=20e-6)
    written = pp.Sequence(other)
    written.add_block(pp.make_trapezoid("x", area=1000, duration=2e-3, system=other))
    written.write(tmp_path / "rasters.seq")

    read = pp.Sequence(system)
    read.read(tmp_path / "rasters.seq")

    assert read.grad_raster_time == pytest.approx(20e-6)


def test_the_rasters_cannot_be_set(system):
    with pytest.raises(AttributeError):
        pp.Sequence(system).grad_raster_time = 5e-6


@pytest.mark.parametrize("name", ["rf_library", "block_cache", "next_free_block_ID"])
def test_upstream_storage_is_refused_with_a_reason(system, name):
    with pytest.raises(AttributeError, match="compiled storage"):
        getattr(pp.Sequence(system), name)


def test_an_unknown_attribute_is_an_ordinary_attribute_error(system):
    seq = pp.Sequence(system)

    with pytest.raises(AttributeError, match="has no attribute 'nonsense'"):
        seq.nonsense  # noqa: B018
    assert not hasattr(seq, "nonsense")
