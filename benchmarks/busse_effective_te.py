"""Time contrast-TE selection and control fits for S13 geometry on one CPU thread.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/busse_effective_te.py --output /tmp/busse_effective_te.json

Fits use the same selected centre, with or without the effective-TE constraint.
Native model setup and PSF forms are warmed before timing. Each timed fit starts
with an empty schedule cache. This is a host CPU benchmark, not a scan result.
"""

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch

import pypulseqpp as pp
from pypulseqpp import sequences
from pypulseqpp._busse import _schedule
from pypulseqpp._busse_opt import optimize_controls, psf_weights
from pypulseqpp._fse_contrast import _select_busse_centre


def main(output):
    torch.set_num_threads(1)
    m = sequences.fse3D_sequence
    etl = 80
    startup = 6
    esp = 0.003416
    first = 0.004162
    controls = (25.0, 70.0, 120.0)
    start = perf_counter()
    c = _select_busse_centre(
        etl, etl, startup, controls, controls, 1.3, esp, first, 0.054026
    )
    selection = perf_counter() - start
    angles = pp.make_busse_schedule(etl + startup, centre_echo=startup + c)
    effective = float(
        pp.calc_fse_effective_te(angles, esp, first_echo_time=first)[startup + c]
    )
    cal, img = pp.make_cartesian_plane_sampling(
        (276, 140), (2, 2), (24, 24), caipi_shift=1, elliptical=True, train_length=etl
    )
    sampled = sorted({*cal, *img})
    coords = (np.asarray(sampled) - [138, 70]) / [276, 140]
    plane = np.full((276, 140), -1, dtype=np.int32)
    for train in pp.make_pclo_order(coords, etl, center_echo=c, pad=True):
        for e, i in enumerate(train):
            if i is not None:
                plane[sampled[i]] = e
    psf_weights(plane, etl)
    protocol = {
        "length": etl + startup,
        "minimum_flip_angle": np.deg2rad(25),
        "centre_flip_angle": np.deg2rad(70),
        "maximum_flip_angle": np.deg2rad(120),
        "centre_echo": startup + c,
        "startup_echoes": startup,
        "initial_decay": 1.3,
        "esp": esp,
        "first_echo_time": first,
        "tr": 3.431,
        "echo_map": plane,
        "t1_ms": m.DESIGN_T1_MS,
        "t2_ms": m.DESIGN_T2_MS,
        "b1_scales": m.DESIGN_B1_SCALES,
        "contrast_pair": m.DESIGN_CONTRAST,
    }
    # Warm native model setup before either timed condition.
    optimize_controls(**protocol)
    report = {
        "requested_ms": 54.026,
        "physical_ms": (first + (startup + c) * esp) * 1000,
        "prescribed_effective_ms": effective * 1000,
        "centre_acquired_echo_1based": c + 1,
        "selection_seconds": selection,
    }
    for preserve in (False, True):
        times = []
        for _ in range(3):
            _schedule.cache_clear()
            begin = perf_counter()
            fit = optimize_controls(
                **protocol, effective_te=effective if preserve else None
            )
            times.append(perf_counter() - begin)
        report["preserve" if preserve else "unconstrained"] = {
            "median_seconds": median(times),
            "runs_seconds": times,
            "achieved_effective_ms": float(
                pp.calc_fse_effective_te(fit, esp, first_echo_time=first)[startup + c]
            )
            * 1000,
            "controls_deg": np.rad2deg(fit[[np.argmin(fit), startup + c, -1]]).tolist(),
        }
    print(json.dumps(report, indent=2), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
