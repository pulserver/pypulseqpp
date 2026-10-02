"""The prescription a shipped sequence exposes, and what an application resolves and times."""

import inspect

import numpy as np
import pytest
from zoo import FUNCTIONS, SMALL, app_class, chain, parameters

import pypulseqpp as pp
from pypulseqpp import sequences


def test_a_parameter_carries_its_type_default_unit_and_description():
    main = sequences.gre2D_sequence.main
    protocol = sequences.parameters(main)
    te, n_x = protocol["te"], protocol["n_x"]

    assert list(protocol) == [
        name for name in inspect.signature(main).parameters if name != "system"
    ]
    assert (te.type, te.default, te.optional, te.unit) == (float, 8e-3, True, "s")
    assert te.description.startswith("Echo time (s). ``None`` is as short as")
    assert (n_x.type, n_x.optional, n_x.unit) == (int, False, "")
    assert protocol["flip_angle_deg"].unit == "degrees"
    assert protocol["fov_x"].unit == protocol["fov_y"].unit == "m"


def test_the_choices_of_a_parameter_are_the_values_its_type_lists_in_order():
    protocol = sequences.parameters(sequences.fse3D_sequence.main)

    assert protocol["excitation"].choices == ("slab", "nonselective")
    assert protocol["wave"].choices == ("phase", "partition", "both")
    assert protocol["n_x"].choices == ()


def test_the_choices_of_a_function_parameter_are_the_values_its_type_lists():
    protocol = sequences.parameters(sequences.bssfp2D_sequence.main)

    assert protocol["gating"].choices == ("none", "retrospective", "prospective")
    assert protocol["n_x"].choices == ()


@pytest.mark.parametrize("name", sequences.ZOO)
def test_every_prescribed_parameter_is_described_and_a_string_lists_its_choices(
    name,
):
    for parameter in parameters(name).values():
        assert parameter.description, parameter.name
        assert parameter.type is not None, parameter.name
        assert parameter.type is not str or parameter.choices, parameter.name


def test_the_units_the_shipped_sequences_state_are_the_package_units():
    stated = {
        parameter.unit
        for name in sequences.ZOO
        for parameter in parameters(name).values()
    }

    assert stated <= {"", "m", "s", "Hz", "degrees", "T/m", "beats per minute"}


class Pause(sequences.SequenceApp):
    """A pause of one TR per repetition, the TR its own shortest when not asked for."""

    MAX_GRAD = 40.0
    MAX_SLEW = 150.0
    SHORTEST = 10e-3

    def init_sequence(self, n: int = 3, tr: float | None = None, label: str = "a"):
        self.n, self.tr, self.label = n, self.SHORTEST if tr is None else tr, label
        if self.tr < self.SHORTEST:
            raise ValueError("the TR is shorter than a pause")
        self.resolve(tr=self.tr)
        self.loops = 0

    def loop(self):
        self.loops += 1
        for _ in range(self.n):
            self.kernel()

    def kernel(self):
        self.seq.add_block(pp.make_delay(self.tr))


def test_a_resolved_value_replaces_the_requested_one():
    assert Pause().resolved == {"n": 3, "tr": 10e-3, "label": "a"}
    assert Pause(n=2, tr=20e-3).resolved == {"n": 2, "tr": 20e-3, "label": "a"}


def test_resolving_a_name_init_sequence_does_not_take_is_refused():
    with pytest.raises(TypeError, match="has no parameter te"):
        Pause().resolve(te=1e-3)


def test_a_prescription_the_design_refuses_raises_on_construction():
    with pytest.raises(ValueError, match="shorter than a pause"):
        Pause(tr=5e-3)


def test_a_stated_scan_time_is_reported_without_designing():
    class Stated(Pause):
        def init_sequence(self, n: int = 3, tr: float | None = None):
            super().init_sequence(n, tr)
            self.duration = n * self.tr

    app = Stated(n=4)

    assert app.scan_time() == pytest.approx(40e-3)
    assert not app.loops


def test_without_a_stated_scan_time_the_whole_chain_is_designed_and_timed():
    class Prescanned(Pause):
        def prescans(self):
            return {"calibration": self.kernel}

    app = Prescanned(n=4)

    assert app.scan_time() == pytest.approx(50e-3)
    assert app.loops == 1
    assert app.seq.duration()[0] == pytest.approx(40e-3)


@pytest.mark.parametrize("name", sequences.ZOO)
def test_the_scan_time_is_the_time_the_designed_chain_plays(name):
    app = app_class(name)(pp.Opts(), **SMALL[name])
    stated = app.scan_time()
    designed = [app.design(prescan).duration()[0] for prescan in app.prescans()]

    assert stated == pytest.approx(sum(designed) + app.design().duration()[0])


@pytest.mark.parametrize("name", sequences.ZOO)
def test_every_shipped_application_states_its_scan_time(name):
    assert app_class(name)(pp.Opts(), **SMALL[name]).duration is not None


#: The definition a shipped application records each prescribed parameter as.
RECORDED = {
    "te": "TE",
    "tr": "TR",
    "ti": "TI",
    "esp": "EchoSpacing",
    "slice_thickness": "SliceThickness",
    "slice_spacing": "SliceGap",
    "tr_periphery": "TRPeriphery",
    "etl_periphery": "EchoTrainLengthPeriphery",
    "n_blades": "NumBlades",
    "n_gain_calibration_readouts": "NumGainCalibrationReadouts",
}


@pytest.mark.parametrize("name", sequences.ZOO)
def test_the_resolved_prescription_is_what_the_file_records(name):
    """A multi-echo ``TE`` lists every echo: ``te``, then one ``echo_spacing`` apart."""
    app = app_class(name)(pp.Opts(), **SMALL[name])
    written, resolved = app.design().definitions, app.resolved
    recorded = {
        parameter: np.atleast_1d(written[key])
        for parameter, key in RECORDED.items()
        if parameter in resolved and key in written
    }

    assert "tr" in recorded
    for parameter, values in recorded.items():
        assert values[0] == pytest.approx(resolved[parameter]), parameter
    if resolved.get("echo_spacing") is not None:
        te = recorded["te"]
        assert te == pytest.approx(
            te[0] + resolved["echo_spacing"] * np.arange(len(te))
        )


@pytest.mark.parametrize("name", sequences.ZOO)
def test_the_resolved_prescription_resolves_to_itself(name):
    """Prescribing what a design resolved to designs it again."""
    app = app_class(name)(pp.Opts(), **SMALL[name])
    again = app_class(name)(pp.Opts(), **app.resolved)

    for parameter, value in app.resolved.items():
        assert again.resolved[parameter] == pytest.approx(value, rel=1e-9), parameter


@pytest.mark.parametrize("name", sequences.ZOO)
def test_every_acquisition_samples_at_the_resolved_receiver_bandwidth(name):
    app = app_class(name)(pp.Opts(), **SMALL[name])
    seq = app.design()
    blocks = (seq.get_block(i) for i in range(1, len(seq.block_events) + 1))
    (dwell,) = {float(block.adc.dwell) for block in blocks if block.adc is not None}

    assert 1 / dwell == pytest.approx(app.resolved["readout_bandwidth_hz"])


# -- a sequence function records its design in the file ----------------------


@pytest.mark.parametrize("name", FUNCTIONS)
def test_a_function_designs_the_definitions_it_wrote_again_from_them(name):
    """Prescribing what a design wrote down designs it again."""
    main = getattr(sequences, name).main
    written = chain(main(**SMALL[name]))[-1].definitions
    prescribed = {
        parameter: np.atleast_1d(written[key])[0]
        for parameter, key in RECORDED.items()
        if parameter in parameters(name) and key in written
    }
    again = chain(main(**{**SMALL[name], **prescribed}))[-1].definitions

    assert "tr" in prescribed
    for parameter, key in RECORDED.items():
        if parameter in prescribed:
            assert np.atleast_1d(again[key]) == pytest.approx(
                np.atleast_1d(written[key]), rel=1e-9
            ), parameter


@pytest.mark.parametrize("name", FUNCTIONS)
def test_every_acquisition_of_a_function_samples_at_one_receiver_bandwidth(name):
    built = chain(getattr(sequences, name).main(**SMALL[name]))
    blocks = (
        seq.get_block(i) for seq in built for i in range(1, len(seq.block_events) + 1)
    )
    dwells = {float(block.adc.dwell) for block in blocks if block.adc is not None}

    assert len(dwells) == 1
    assert min(dwells) > 0
