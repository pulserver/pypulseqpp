"""The receive-gain prescan every shipped sequence states, as whole repetitions."""

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
    """Readouts acquired from the first block to the end of each repetition of the written file, and whether it images."""
    seq = seq.remove_duplicates()
    size, start = seq.repetition()
    assert start == 1
    acquires = np.asarray(seq._native.block_events())[:, 4] > 0
    navigator = np.broadcast_to(
        np.asarray(seq.evaluate_labels(evolution="adc").get("NAV", 0)) != 0,
        (int(acquires.sum()),),
    )
    imaging = np.zeros(acquires.size, dtype=bool)
    imaging[np.flatnonzero(acquires)[~navigator]] = True
    count = -(-acquires.size // size)
    ends = [int(acquires[: (r + 1) * size].sum()) for r in range(count)]
    images = [bool(imaging[: (r + 1) * size].any()) for r in range(count)]
    return ends, images


@pytest.mark.parametrize("case", CASES)
def test_every_file_with_a_readout_states_its_prescan_through_its_first_imaging_repetition(
    case,
):
    name, prescription = CASES[case]
    for seq in chain(getattr(sequences, name)(**prescription)):
        stated = seq.definitions.get("NumGainCalibrationReadouts")
        ends, images = repetition_ends(seq)
        if not ends or ends[-1] == 0:
            assert stated is None
            continue
        count = int(np.atleast_1d(stated)[0])
        assert count == ends[images.index(True)], (count, sorted(set(ends))[:8])


def test_a_prescan_counts_the_dummies_before_the_first_imaging_repetition():
    seq = sequences.gre2D_sequence(**{**SMALL["gre2D_sequence"], "n_dummy": 3})

    assert sequences.gain_calibration_readouts(seq) == 1
    assert sequences.gain_calibration_readouts(seq, repetitions=4) == 4


def test_a_shot_that_acquires_only_navigators_is_not_an_imaging_repetition():
    prescription = {**SMALL["mprage3D_sequence"], "navigator": True, "n_dummy": 1}
    seq = sequences.mprage3D_sequence(**prescription)
    size, _ = seq.repetition()
    acquires = np.asarray(seq._native.block_events())[:, 4] > 0
    dummy, first = acquires[:size].sum(), acquires[size : 2 * size].sum()

    assert dummy > 0
    assert sequences.gain_calibration_readouts(seq) == dummy + first


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
