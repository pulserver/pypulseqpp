"""Fit the PD target-signal parameters to the trains of Buonincontri et al., Fig. 1.

The fixture holds five trains read off the published figure (centre, three
intermediate shots and periphery). The central and the peripheral train are
fitted, each on its own parameters, by inverse EPG at the spacing of a 7 T
0.8 mm 3D FSE (3.416 ms, 4.162 ms to the first echo; the abstract states
none). The intermediate trains are then designed from the parameters
interpolated at their place in the transition, set by their length, and
compared, not fitted. Prints the parameters ``pypulseqpp._pd`` takes as
``PD_CENTRE`` and ``PD_PERIPHERY`` and the RMS difference of each train.

    python benchmarks/pd_buonincontri_fit.py
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from pypulseqpp._pd import PD_MAXIMUM_ANGLE_DEG, _pd_trains

ESP, FIRST = 3.416e-3, 4.162e-3
FIXTURE = Path(__file__).parents[1] / "tests" / "fixtures" / "buonincontri_figure1.json"
LOWER = [30.0, 0.02, -0.8, 0.02, 0.0, 0.3, 0.05, 0.1]
UPPER = [PD_MAXIMUM_ANGLE_DEG, 1.0, 0.8, 0.98, 1.5, 4.0, 1.0, 10.0]


def design(controls, length):
    angles, _ = _pd_trains([length], [controls], PD_MAXIMUM_ANGLE_DEG, ESP, FIRST)
    return np.rad2deg(angles[0])


def fit(published):
    published = np.asarray(published)
    best = None
    for onset in (0.3, 0.6, 0.8):
        for level in (0.25, 0.45):
            for start in (0.5, 0.8):
                for power in (0.8, 2.0):
                    found = least_squares(
                        lambda c: design(c, len(published)) - published,
                        [100.0, level, 0.0, onset, 0.5, power, start, 1.5],
                        bounds=(LOWER, UPPER),
                    )
                    if best is None or found.cost < best.cost:
                        best = found
    return best.x


def rms(controls, published):
    return float(np.sqrt(np.mean((design(controls, len(published)) - published) ** 2)))


def main():
    trains = {
        k: np.asarray(v)
        for k, v in json.loads(FIXTURE.read_text())["angles_deg"].items()
    }
    centre, periphery = fit(trains["centre"]), fit(trains["periphery"])
    centre, periphery = np.round(centre, 3), np.round(periphery, 3)
    print("PD_CENTRE =", tuple(float(v) for v in centre))
    print("PD_PERIPHERY =", tuple(float(v) for v in periphery))
    first, last = len(trains["centre"]), len(trains["periphery"])
    for name, published in trains.items():
        place = (first - len(published)) / (first - last)
        controls = centre + place * (periphery - centre)
        print(
            f"{name:10s} {len(published):3d} pulses  place {place:.3f}  RMS {rms(controls, published):5.2f} deg"
        )


if __name__ == "__main__":
    main()
