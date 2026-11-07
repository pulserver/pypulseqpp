"""Illustrate Buonincontri's transition and ordering using a Busse family.

The abstract's Fig. 1 states ETL 45/20, TR 1800/150 ms, TE 28 ms, CAIPI 4,
and 586 shots, but not ESP or numerical PDvar* controls. This reconstruction
uses our prescribed-PSS Busse family (25/70/120 degrees, six startup pulses),
a common ESP of 3.416 ms and S13 first echo timing. Treating TE 28 ms as
physical timing places the 70-degree centre at the second acquired echo;
this produces a steep early rise. The abstract does not establish that TE
convention. This is a transition-method demonstration, not a reconstruction
of the published PDvar* RF trains. Peripheral TR is raised to
the duration of the common padded GE layout; the exact scan-time cost is
reported. Fig. 2 uses an explicitly illustrative 192 x 160 CAIPI plane, since
the full view list and sorting details are not supplied in the abstract.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/buonincontri_transition.py --output /tmp/buonincontri
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import pypulseqpp as pp
from pypulseqpp import sequences
from pypulseqpp._busse import _make_busse_schedules
from pypulseqpp._busse_opt import optimize_controls

ESP = 0.003416
FIRST = 0.004162
STARTUP = 6
CENTRE = 1


def hardware():
    return pp.cap_system(
        pp.apply_system_derates(
            pp.Opts(
                gamma=42575900.0,
                B0=7.001091,
                max_grad=100,
                grad_unit="mT/m",
                max_slew=115,
                slew_unit="T/m/s",
                max_b1=25,
                b1_unit="uT",
                rf_dead_time=232e-6,
                rf_ringdown_time=4e-6,
                rf_raster_time=2e-6,
                grad_raster_time=4e-6,
                adc_raster_time=2e-6,
                block_duration_raster=4e-6,
            )
        ),
        max_grad=80,
        max_slew=200,
    )


def maps(order, sampled, shape):
    shot_map = np.full(shape, np.nan)
    echo_map = np.full(shape, np.nan)
    used = []
    for shot, train in enumerate(order):
        for echo, index in enumerate(train):
            if index is not None:
                shot_map[sampled[index]] = shot / max(len(order) - 1, 1)
                echo_map[sampled[index]] = echo
                used.append(index)
    assert sorted(used) == list(range(len(sampled)))
    return shot_map, echo_map


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    module = sequences.fse3D_sequence
    # One common padded timing layout; the small encoding plane only avoids
    # constructing a volume while measuring its minimum TR.
    timing = module(
        hardware(),
        n_x=276,
        n_y=16,
        n_z=8,
        fov_x=0.21,
        fov_y=0.1596,
        fov_z=0.112,
        readout_bandwidth_hz=500000,
        crusher_cycles=0.5,
        rf_overlap=True,
        esp=ESP,
        etl=45,
        etl_periphery=20,
        te=None,
        tr=None,
        n_dummy=0,
        flip_modulation="busse",
        busse_centre_echo=CENTRE,
    )
    minimum_tr = float(timing.definitions["TR"][0])
    assert timing.check_timing()[0]
    peripheral_tr = max(0.15, minimum_tr)
    count = 586
    u = np.linspace(0, 1, count)
    step = module.cubic(u)
    lengths = np.rint(45 - 25 * step).astype(int)
    paper_tr = 1.8 - 1.65 * step
    feasible_tr = 1.8 + (peripheral_tr - 1.8) * step
    shape = (192, 160)
    calibration, imaging = pp.make_cartesian_plane_sampling(
        shape, (2, 2), (24, 24), caipi_shift=1, elliptical=True, train_length=1
    )
    sampled = sorted({*calibration, *imaging})
    coordinates = (np.asarray(sampled) - np.array(shape) // 2) / shape
    endpoints = []
    fit_times = []
    for endpoint, (length, period) in enumerate(((45, 1.8), (20, peripheral_tr))):
        order = pp.make_pclo_order(coordinates, length, center_echo=CENTRE, pad=True)
        _, echo_map = maps(order, sampled, shape)
        plane = np.where(np.isfinite(echo_map), echo_map, -1).astype(np.int32)
        begin = perf_counter()
        fitted = optimize_controls(
            length + STARTUP,
            minimum_flip_angle=np.deg2rad(25),
            centre_flip_angle=np.deg2rad(70),
            maximum_flip_angle=np.deg2rad(120),
            centre_echo=STARTUP + CENTRE,
            startup_echoes=STARTUP,
            initial_decay=1.3,
            esp=ESP,
            first_echo_time=FIRST,
            tr=period,
            echo_map=plane,
            t1_ms=module.DESIGN_T1_MS,
            t2_ms=module.DESIGN_T2_MS,
            b1_scales=module.DESIGN_B1_SCALES,
            contrast_pair=module.DESIGN_CONTRAST,
            objective="contrast" if endpoint == 0 else "psf",
        )
        fit_times.append(perf_counter() - begin)
        endpoints.append(np.rad2deg(fitted[[np.argmin(fitted), STARTUP + CENTRE, -1]]))
    controls = endpoints[0] + step[:, None] * (endpoints[1] - endpoints[0])
    begin = perf_counter()
    profiles = np.rad2deg(
        _make_busse_schedules(lengths + STARTUP, controls, STARTUP + CENTRE, 1.3)
    )
    construction = perf_counter() - begin
    assert np.all(
        profiles[np.arange(profiles.shape[1])[None, :] >= (lengths + STARTUP)[:, None]]
        == 0
    )
    report = {
        "source": "Buonincontri et al., ISMRM 2025, abstract 566-05-007, Figures 1-2",
        "scope": "Method reproduction with Busse replacement for PDvar*, not an exact numerical reproduction of the proprietary RF train or scan results.",
        "assumptions": {
            "ESP_ms": ESP * 1000,
            "first_echo_ms": FIRST * 1000,
            "startup": STARTUP,
            "initial_controls_deg": [25, 70, 120],
            "initial_decay_echoes": 1.3,
            "illustrative_view_matrix": shape,
            "586_shots": "Stated in the abstract, used to illustrate the normalized transition; not inferred from the incomplete published sampling description.",
        },
        "minimum_padded_TR_ms": minimum_tr * 1000,
        "published_TR_endpoints_ms": [1800, 150],
        "feasible_TR_endpoints_ms": [1800, peripheral_tr * 1000],
        "TE_ms": (FIRST + ESP * (STARTUP + CENTRE)) * 1000,
        "ETL_endpoints": [45, 20],
        "shots": count,
        "published_transition_seconds": float(paper_tr.sum()),
        "feasible_transition_seconds": float(feasible_tr.sum()),
        "extra_seconds": float((feasible_tr - paper_tr).sum()),
        "extra_percent": float((feasible_tr.sum() / paper_tr.sum() - 1) * 100),
        "optimized_endpoint_controls_deg": [x.tolist() for x in endpoints],
        "CPU_endpoint_fit_seconds": fit_times,
        "CPU_batched_586_trains_seconds": construction,
    }
    print(json.dumps(report, indent=2), flush=True)
    (output / "buonincontri_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    figure = plt.figure(figsize=(13, 9), layout="constrained")
    grid = figure.add_gridspec(5, 2, width_ratios=(1.05, 1.8))
    tr_axis = figure.add_subplot(grid[:2, 0])
    tr_axis.plot(u, feasible_tr * 1000, label="Our padded GE layout")
    tr_axis.plot(u, paper_tr * 1000, "--", color="0.55", label="Published TR endpoints")
    tr_axis.axhline(minimum_tr * 1000, color="0.65", linestyle=":")
    tr_axis.set(
        xlabel="Shot position: centre → periphery",
        ylabel="TR (ms)",
        title=f"Cubic TR transition · 586 shots\nPadding costs {report['extra_seconds']:.2f} s ({report['extra_percent']:.2f}%)",
    )
    tr_axis.legend(fontsize=9)
    length_axis = figure.add_subplot(grid[2:4, 0])
    length_axis.plot(u, lengths, label="Acquired ETL")
    length_axis.axhline(
        45, linestyle="--", color="0.55", label="Common acquired-echo slots"
    )
    length_axis.set(
        xlabel="Shot position: centre → periphery",
        ylabel="Echoes",
        title="ETL 45 → 20 · six startup pulses in every train",
    )
    length_axis.legend(fontsize=9)
    note = figure.add_subplot(grid[4, 0])
    note.axis("off")
    note.text(
        0,
        0.9,
        f"ESP fixed: {ESP * 1000:.3f} ms\nPhysical TE: {report['TE_ms']:.3f} ms (nearest to 28 ms)\nIndependent CPU fits: {fit_times[0]:.2f} / {fit_times[1]:.2f} s\n586 intermediate trains: {construction:.2f} s",
        va="top",
    )
    for row, shot in enumerate(np.rint(np.linspace(0, count - 1, 5)).astype(int)):
        axis = figure.add_subplot(grid[row, 1])
        last = lengths[shot] + STARTUP
        axis.axvspan(0.5, STARTUP + 0.5, color="0.92")
        axis.axvspan(last + 0.5, 51.5, color="#e8eef4")
        axis.plot(np.arange(1, 52), profiles[shot], color="#24669b")
        axis.axvline(STARTUP + CENTRE + 1, linestyle=":", color="0.6")
        axis.set(
            xlim=(0.5, 51.5),
            ylim=(-5, 130),
            ylabel="Flip (°)",
            title=f"Shot {shot}: acquired ETL {lengths[shot]}, TR {feasible_tr[shot] * 1000:.1f} ms",
        )
        axis.grid(alpha=0.15)
        if row == 4:
            axis.set_xlabel(
                "RF slot: 6 startup + up to 45 acquired; blue shading = zero RF padding"
            )
    figure.suptitle(
        "Buonincontri Figure 1 transition using our Busse / inverse-EPG trains\nPublished endpoints; common ESP and padded GE layout · RF shapes are Busse, not published PDvar*"
    )
    figure.savefig(output / "buonincontri_figure1_busse.png", dpi=160)
    variable_lengths, _, place = module.shot_parameters(
        len(sampled), 45, 20, 1.8, peripheral_tr
    )
    orders = [
        (
            "A. Linear ordering",
            pp.make_linear_order(coordinates[:, ::-1], 45, pad=True),
        ),
        (
            "B. Conventional radial wedges",
            pp.make_radial_order(coordinates, 45, pad=True),
        ),
        (
            "C. Buonincontri adapted radial",
            module.deal_trains(coordinates, variable_lengths, place, CENTRE, 45),
        ),
    ]
    figure, axes = plt.subplots(3, 2, figsize=(10.5, 12), layout="constrained")
    for row, (name, order) in enumerate(orders):
        shot_map, echo_map = maps(order, sampled, shape)
        for column, (values, title, vmax) in enumerate(
            ((shot_map, "Shot position", 1), (echo_map, "Acquired echo index", 44))
        ):
            axis = axes[row, column]
            artist = axis.imshow(
                values.T,
                origin="lower",
                cmap="turbo",
                vmin=0,
                vmax=vmax,
                extent=(-shape[0] / 2, shape[0] / 2, -shape[1] / 2, shape[1] / 2),
                interpolation="nearest",
            )
            axis.set(title=f"{name} · {title}", xlabel="ky index", ylabel="kz index")
            figure.colorbar(artist, ax=axis, shrink=0.8)
    figure.suptitle(
        "Buonincontri Figure 2: ordering mechanism reproduced with our code\nSame illustrative CAIPI 2x2, shift 1 + 24x24 ACS support in every panel\nA/B: identical 45-echo trains · C: 45→20 echoes and cubic TR transition"
    )
    figure.savefig(output / "buonincontri_figure2_ordering.png", dpi=160)
    np.savez(
        output / "buonincontri_profiles.npz",
        profiles_deg=profiles,
        acquired_ETL=lengths,
        TR_published=paper_tr,
        TR_feasible=feasible_tr,
        controls_deg=controls,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
