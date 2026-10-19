"""The receive-gain prescan every shipped sequence states: whole repetitions lasting the calibration duration."""

import numpy as np
import pytest
from zoo import SMALL, chain

import pypulseqpp as pp
from pypulseqpp import sequences

#: Prescriptions whose repetitions differ from the small one's: navigators
#: after each shot, trains of two lengths, slices packed into packets.
VARIANTS = {
    "epi2D_reference_frames": ("epi2D_sequence", {"n_frames": 3, "n_dummy": 2}),
    "mprage3D_navigator": ("mprage3D_sequence", {"navigator": True}),
    "fse3D_navigator": ("fse3D_sequence", {"navigator": True}),
    "fse3D_dual_region": (
        "fse3D_sequence",
        {"tr_periphery": None, "etl_periphery": 4},
    ),
    "epi2D_frames": ("epi2D_sequence", {"n_frames": 3}),
}

CASES = {
    **{f"{name}_small": (name, SMALL[name]) for name in sequences.ZOO},
    **{f"{name}_default": (name, {}) for name in sequences.ZOO},
    **{
        key: (name, {**SMALL[name], **changes})
        for key, (name, changes) in VARIANTS.items()
    },
}


def repetition_ends(seq):
    """Per repetition of the written file: readouts the prescan acquires from the first block to its end, whether it images, and its duration."""
    seq = seq.remove_duplicates()
    size, start = seq.repetition()
    assert start == 1
    acquires = np.asarray(seq._native.block_events())[:, 4] > 0
    exempt = seq.evaluate_labels(evolution="blocks").get("NOPRESCAN", 0)
    calibrates = acquires & ~np.broadcast_to(np.asarray(exempt) != 0, acquires.shape)
    navigator = np.broadcast_to(
        np.asarray(seq.evaluate_labels(evolution="adc").get("NAV", 0)) != 0,
        (int(acquires.sum()),),
    )
    imaging = np.zeros(acquires.size, dtype=bool)
    imaging[np.flatnonzero(acquires)[~navigator]] = True
    count = -(-acquires.size // size)
    ends = [int(calibrates[: (r + 1) * size].sum()) for r in range(count)]
    images = [bool(imaging[r * size : (r + 1) * size].any()) for r in range(count)]
    durations = np.asarray(seq._native.block_durations())
    lasts = [float(durations[r * size : (r + 1) * size].sum()) for r in range(count)]
    return ends, images, lasts


def expected_count(ends, images, lasts):
    """Readouts through the fewest imaging repetitions lasting the calibration duration, or all of them."""
    total = 0.0
    last = None
    for end, images_here, lasting in zip(ends, images, lasts, strict=True):
        if not images_here:
            continue
        total += lasting
        last = end
        if total >= sequences.GAIN_CALIBRATION_DURATION * (1 - 1e-9):
            break
    return last


@pytest.mark.parametrize("case", CASES)
def test_every_file_with_a_readout_states_whole_repetitions_lasting_the_calibration_duration(
    case,
):
    name, prescription = CASES[case]
    for seq in chain(getattr(sequences, name)(**prescription)):
        stated = seq.definitions.get("NumGainCalibrationReadouts")
        ends, images, lasts = repetition_ends(seq)
        if not ends or ends[-1] == 0:
            assert stated is None
            continue
        count = int(np.atleast_1d(stated)[0])
        assert count in ends, (count, sorted(set(ends))[:8])
        assert count == expected_count(ends, images, lasts), (
            count,
            ends[:8],
            lasts[:4],
        )


def test_a_prescan_counts_the_dummies_before_the_first_imaging_repetition():
    seq = sequences.gre2D_sequence(**{**SMALL["gre2D_sequence"], "n_dummy": 3})

    assert sequences.gain_calibration_readouts(seq, repetitions=1) == 1
    assert sequences.gain_calibration_readouts(seq, repetitions=4) == 4


def test_a_prescan_plays_as_many_repetitions_as_last_the_calibration_duration():
    seq = sequences.gre2D_sequence(**{**SMALL["gre2D_sequence"], "n_dummy": 3})
    tr = float(np.atleast_1d(seq.definitions["TR"])[0])
    needed = int(np.ceil(sequences.GAIN_CALIBRATION_DURATION / tr - 1e-9))

    assert sequences.gain_calibration_readouts(seq) == needed
    assert sequences.gain_calibration_readouts(seq, duration=3.5 * tr) == 4


def test_a_shot_that_acquires_only_navigators_is_not_an_imaging_repetition():
    prescription = {**SMALL["mprage3D_sequence"], "navigator": True, "n_dummy": 1}
    seq = sequences.mprage3D_sequence(**prescription)
    size, _ = seq.repetition()
    acquires = np.asarray(seq._native.block_events())[:, 4] > 0
    dummy, first = acquires[:size].sum(), acquires[size : 2 * size].sum()

    assert dummy > 0
    assert sequences.gain_calibration_readouts(seq, repetitions=1) == dummy + first


@pytest.mark.parametrize(
    "name", ["gre_multiecho2D_sequence", "gre_multiecho3D_sequence"]
)
def test_a_multi_echo_prescan_calibrates_on_the_first_echo_alone(name):
    seq = getattr(sequences, name)(**{**SMALL[name], "n_dummy": 0})
    adc = seq.evaluate_labels(evolution="adc")
    echoes = np.asarray(adc["ECO"])
    exempt = np.broadcast_to(np.asarray(adc.get("NOPRESCAN", 0)) != 0, echoes.shape)

    assert echoes.max() > 0
    np.testing.assert_array_equal(exempt, echoes > 0)
    assert sequences.gain_calibration_readouts(seq, repetitions=2) == 2


def test_a_prescan_of_no_repetition_is_refused():
    seq = sequences.gre2D_sequence(**SMALL["gre2D_sequence"])

    with pytest.raises(ValueError, match="at least one repetition"):
        sequences.gain_calibration_readouts(seq, repetitions=0)


def test_a_prescan_longer_than_the_sequence_is_refused():
    seq = sequences.gre2D_sequence(**{**SMALL["gre2D_sequence"], "n_dummy": 0})

    with pytest.raises(ValueError, match="fewer than the 17"):
        sequences.gain_calibration_readouts(seq, repetitions=17)


def test_a_sequence_without_readouts_has_no_imaging_repetition():
    seq = pp.Sequence(pp.Opts())
    for _ in range(3):
        seq.add_block(pp.make_block_pulse(np.pi / 2, duration=1e-3))
        seq.add_block(pp.make_delay(10e-3))

    with pytest.raises(ValueError, match="0 imaging repetitions"):
        sequences.gain_calibration_readouts(seq)
