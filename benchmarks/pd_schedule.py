"""Time the PD train design of a centre/periphery transition on one CPU thread.

The transition is that of Buonincontri et al. (Proc. ISMRM 2025, abstract
566-05-007): ETL 45 to 20, TR 1800 to 150 ms, 586 shots, here with six
startup pulses at 3.416 ms spacing, 4.162 ms to the first echo. Every shot's
target signal is the cubic step between ``PD_CENTRE`` and ``PD_PERIPHERY``
and its train is designed by inverse EPG in one batched pass; the optional
forward-EPG fit of the tail onset is timed per endpoint.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/pd_schedule.py
"""

from __future__ import annotations

import json
from time import perf_counter

import numpy as np

from pypulseqpp import sequences
from pypulseqpp._pd import PD_CENTRE, PD_MAXIMUM_ANGLE_DEG, PD_PERIPHERY, _pd_trains

ESP, FIRST, STARTUP, SHOTS = 3.416e-3, 4.162e-3, 6, 586


def design(lengths, place):
    module = sequences.fse3D_sequence
    centre, periphery = np.array(PD_CENTRE), np.array(PD_PERIPHERY)
    _pd_trains(
        [45 + STARTUP, 20 + STARTUP],
        [centre, periphery],
        PD_MAXIMUM_ANGLE_DEG,
        ESP,
        FIRST,
    )
    controls = centre + module.cubic(place)[:, None] * (periphery - centre)
    return _pd_trains(lengths + STARTUP, controls, PD_MAXIMUM_ANGLE_DEG, ESP, FIRST)[0]


def fitted_endpoints():
    import torch

    from pypulseqpp._pd_opt import optimize_pd_controls

    torch.set_num_threads(1)
    module = sequences.fse3D_sequence
    seconds = []
    for length, tr, controls in ((45, 1.8, PD_CENTRE), (20, 0.15, PD_PERIPHERY)):
        echo_map = np.arange(length)[None, :] * np.ones((4, 1), dtype=np.int32)
        start = perf_counter()
        optimize_pd_controls(
            length + STARTUP,
            np.array(controls),
            maximum=PD_MAXIMUM_ANGLE_DEG,
            startup=STARTUP,
            esp=ESP,
            first_echo=FIRST,
            tr=tr,
            centre=STARTUP + 8,
            echo_map=echo_map.astype(np.int32),
            t1_ms=module.DESIGN_T1_MS,
            t2_ms=module.DESIGN_T2_MS,
            b1_scales=module.DESIGN_B1_SCALES,
            contrast_pair=module.DESIGN_CONTRAST,
        )
        seconds.append(perf_counter() - start)
    return seconds


def main():
    place = np.linspace(0, 1, SHOTS)
    step = sequences.fse3D_sequence.cubic(place)
    lengths = np.rint(45 + (20 - 45) * step).astype(int)
    timings = []
    for _ in range(5):
        start = perf_counter()
        trains = design(lengths, place)
        timings.append(perf_counter() - start)
    report = {
        "shots": SHOTS,
        "pulses_per_train": int(trains.shape[1]),
        "train_design_seconds_min": round(min(timings), 4),
        "train_design_seconds_median": round(float(np.median(timings)), 4),
        "peak_angle_deg": round(float(np.rad2deg(trains.max())), 2),
    }
    try:
        report["tail_onset_fit_seconds_per_endpoint"] = [
            round(s, 2) for s in fitted_endpoints()
        ]
    except ImportError:
        report["tail_onset_fit_seconds_per_endpoint"] = "blochsim not installed"
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
