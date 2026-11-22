"""Shuffled 3D sequences read their calibration region in wave-free reference trains, centre out."""

import numpy as np
import pytest

N_Y, N_Z, ACS_Y, ACS_Z = 16, 8, 4, 2

SHUFFLED = {
    "fse3D_sequence": {
        "n_x": 32,
        "n_y": N_Y,
        "n_z": N_Z,
        "etl": 8,
        "te": 10e-3,
        "tr": 300e-3,
    },
    "mprage3D_sequence": {
        "n_x": 32,
        "n_y": N_Y,
        "n_z": N_Z,
        "ti": 100e-3,
        "tr": 300e-3,
    },
}


def _labels(module: str) -> dict[str, np.ndarray]:
    import zoo

    import pypulseqpp as pp
    from pypulseqpp import sequences

    kwargs = {**SHUFFLED[module], "ry": 2, "rz": 2, "n_acs_y": ACS_Y, "n_acs_z": ACS_Z}
    seq = zoo.chain(
        getattr(sequences, module).main(pp.Opts(), **kwargs, ordering="shuffling")
    )[-1]
    labels = seq.evaluate_labels(evolution="adc")
    return {name: np.atleast_1d(np.asarray(value)) for name, value in labels.items()}


def _radius(lin: np.ndarray, par: np.ndarray) -> np.ndarray:
    return ((lin - N_Y // 2) / N_Y) ** 2 + ((par - N_Z // 2) / N_Z) ** 2


@pytest.mark.parametrize("module", sorted(SHUFFLED))
def test_the_reference_trains_cover_the_calibration_region_once(module):
    labels = _labels(module)
    ref = labels["REF"].astype(bool)
    views = sorted(
        zip(labels["LIN"][ref].tolist(), labels["PAR"][ref].tolist(), strict=True)
    )
    region = sorted(
        (y, z)
        for y in range(N_Y // 2 - ACS_Y // 2, N_Y // 2 + ACS_Y // 2)
        for z in range(N_Z // 2 - ACS_Z // 2, N_Z // 2 + ACS_Z // 2)
    )
    assert views == region


@pytest.mark.parametrize("module", sorted(SHUFFLED))
def test_reference_views_are_not_counted_as_imaging_calibration(module):
    labels = _labels(module)
    ref = labels["REF"].astype(bool)
    assert not labels["IMA"][ref].any()


def test_a_shuffled_fse_reference_train_reads_its_views_centre_out():
    labels = _labels("fse3D_sequence")
    ref = labels["REF"].astype(bool)
    radius = _radius(labels["LIN"][ref], labels["PAR"][ref])
    echoes = labels["ECO"][ref]
    by_echo = [radius[echoes == e].mean() for e in np.unique(echoes)]
    assert np.all(np.diff(by_echo) >= 0)
