"""The system a design is held to.

Ported from the reference toolbox's `test_opts`, with the one place this
package deliberately answers differently: the raster defaults.
"""

import pypulseq as upstream
import pytest
from pypulseq.convert import convert

import pypulseqpp as pp

FIELDS = (
    "max_grad",
    "max_slew",
    "rise_time",
    "rf_dead_time",
    "rf_ringdown_time",
    "adc_dead_time",
    "adc_raster_time",
    "rf_raster_time",
    "grad_raster_time",
    "block_duration_raster",
    "gamma",
    "B0",
)


@pytest.mark.parametrize("field", FIELDS)
def test_a_system_says_what_it_can_do(field):
    assert hasattr(pp.Opts(), field)


def test_the_amplitude_and_slew_defaults_are_upstreams():
    system = pp.Opts()

    assert system.max_grad == pytest.approx(
        convert(from_value=40, from_unit="mT/m", to_unit="Hz/m"), rel=0.01
    )
    assert system.max_slew == pytest.approx(
        convert(from_value=170, from_unit="T/m/s", to_unit="Hz/m/s"), rel=0.01
    )
    assert system.gamma == pytest.approx(42576000, abs=1)
    assert pytest.approx(1.5, abs=0.01) == system.B0


def test_the_rasters_are_common_multiples_of_both_vendors():
    """20 us over GE's 4 and Siemens' 10; 2 us over Siemens' 1 and GE's 2."""
    system = pp.Opts()

    assert system.grad_raster_time == pytest.approx(20e-6)
    assert system.block_duration_raster == pytest.approx(20e-6)
    assert system.rf_raster_time == pytest.approx(2e-6)
    assert system.adc_raster_time == pytest.approx(2e-6)

    for raster in (system.grad_raster_time, system.block_duration_raster):
        assert raster % 4e-6 == pytest.approx(0.0, abs=1e-12)
        assert raster % 10e-6 == pytest.approx(0.0, abs=1e-12)
    for raster in (system.rf_raster_time, system.adc_raster_time):
        assert raster % 1e-6 == pytest.approx(0.0, abs=1e-12)
        assert raster % 2e-6 == pytest.approx(0.0, abs=1e-12)


def test_a_dead_time_belongs_to_a_chain_rather_than_to_the_format():
    system = pp.Opts()

    assert system.rf_dead_time == 0.0
    assert system.rf_ringdown_time == 0.0
    assert system.adc_dead_time == 0.0


def test_a_factory_asked_for_no_system_designs_against_these_rasters():
    """What the factories read and what Opts() says have to be one system."""
    gx = pp.make_trapezoid("x", area=1000, duration=1e-3)

    assert gx.rise_time % pp.Opts().grad_raster_time == pytest.approx(0.0, abs=1e-12)
    assert upstream.Opts.default.grad_raster_time == pp.Opts().grad_raster_time


def test_a_system_named_as_the_default_is_the_one_the_factories_read():
    ge = pp.Opts(grad_raster_time=4e-6, rf_raster_time=2e-6)
    try:
        ge.set_as_default()
        assert upstream.Opts.default.grad_raster_time == pytest.approx(4e-6)
    finally:
        pp.Opts.reset_default()

    assert upstream.Opts.default.grad_raster_time == pytest.approx(20e-6)


@pytest.mark.parametrize(
    ("limit", "unit", "asked"),
    [("max_grad", "mT/m", 30), ("max_slew", "T/m/s", 100)],
)
def test_a_limit_is_read_in_the_unit_it_was_given_in(limit, unit, asked):
    system = pp.Opts(**{limit: asked, f"{limit.split('_')[1]}_unit": unit})

    assert getattr(system, limit) == pytest.approx(
        convert(from_value=asked, from_unit=unit), rel=0.01
    )


def test_a_rise_time_is_a_slew_limit_said_another_way():
    system = pp.Opts(rise_time=250e-6, max_grad=40, grad_unit="mT/m")

    assert system.rise_time == pytest.approx(250e-6)
    assert system.max_slew == pytest.approx(system.max_grad / 250e-6, rel=0.01)


def test_two_systems_built_the_same_way_are_the_same_system():
    first, second = pp.Opts(), pp.Opts()

    assert (first.max_grad, first.max_slew, first.grad_raster_time) == (
        second.max_grad,
        second.max_slew,
        second.grad_raster_time,
    )


# -- headroom, which belongs to the design rather than to the scanner -------


def test_derating_leaves_the_scanner_saying_what_it_has():
    system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")

    derated = pp.apply_system_derates(system)

    assert derated is not system
    assert derated.max_grad == pytest.approx(pp.MAX_GRAD_DERATE * system.max_grad)
    assert derated.max_slew == pytest.approx(pp.MAX_SLEW_DERATE * system.max_slew)


def test_derating_twice_derates_once():
    system = pp.Opts(max_grad=40, grad_unit="mT/m")

    once = pp.apply_system_derates(system)
    twice = pp.apply_system_derates(once)

    assert twice.max_grad == pytest.approx(once.max_grad)


def test_a_ceiling_only_ever_lowers_a_limit():
    system = pp.Opts(max_grad=80, grad_unit="mT/m", max_slew=200, slew_unit="T/m/s")

    capped = pp.cap_system(system, max_grad=40, max_slew=150)
    above = pp.cap_system(system, max_grad=200, max_slew=500)

    assert capped is not system
    assert capped.max_grad == pytest.approx(
        convert(from_value=40, from_unit="mT/m", to_unit="Hz/m"), rel=1e-6
    )
    assert above.max_grad == pytest.approx(system.max_grad)
    assert above.max_slew == pytest.approx(system.max_slew)


def test_an_axis_left_unnamed_is_left_alone():
    system = pp.Opts(max_grad=80, grad_unit="mT/m", max_slew=200, slew_unit="T/m/s")

    capped = pp.cap_system(system, max_grad=40)

    assert capped.max_slew == pytest.approx(system.max_slew)
