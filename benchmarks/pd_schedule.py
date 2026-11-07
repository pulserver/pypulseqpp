"""Measure PD inverse/forward EPG and a padded centre/periphery transition.

The PD signal prescription follows Mugler et al. (ISMRM 1999, abstract 1631).
The endpoint ETL/TR values follow the supplied Buonincontri abstract; its
proprietary PDvar* controls and startup convention are unspecified. This is
an independent design with S13 timing, not a numerical vendor reproduction.
Run on one CPU thread, excluding imports and native model initialization.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/pd_schedule.py --output /tmp/pd
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
from blochsim._execution import execution
from blochsim.simulators import FSESimulator
from buonincontri_transition import ESP, FIRST, STARTUP, hardware, maps

import pypulseqpp as pp
from pypulseqpp import sequences
from pypulseqpp._busse_opt import psf_weights
from pypulseqpp._pd import _make_pd_schedules, _pd_schedule
from pypulseqpp._pd_opt import optimize_pd_controls


def reference_model(length, tissues):
    class TimedFSE(FSESimulator):
        def layout(self, **kwargs):
            parts = super().layout(**kwargs)
            extra = FIRST - ESP
            return [
                (time + (extra / 2 if i == 1 else extra if i > 1 else 0), op)
                for i, (time, op) in enumerate(parts)
            ]

    return TimedFSE(
        ESP=ESP * 1e3,
        states=2 * length + 1,
        T1=[x[0] for x in tissues],
        T2=[x[1] for x in tissues],
    )


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    module = sequences.fse3D_sequence
    # Initialize native EPG independently of the timed design calls.
    with execution("cpu"):
        reference_model(26, [(1000, 100)]).simulate(
            flip=torch.full((26,), 120.0), TR=1800, repetitions=8
        )
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
        flip_modulation="pd",
        pd_effective_te=0.028,
        pd_final_angle_deg=140,
        pd_final_angle_periphery_deg=100,
    )
    assert timing.check_timing()[0]
    centre = int(timing.definitions["PDCentreEcho"][0])
    effective = float(timing.definitions["EffectiveEchoTime"][0])
    minimum_tr = float(timing.definitions["TR"][0])
    peripheral_tr = max(0.15, minimum_tr)
    shape = (192, 160)
    cal, img = pp.make_cartesian_plane_sampling(
        shape, (2, 2), (24, 24), caipi_shift=1, elliptical=True, train_length=1
    )
    sampled = sorted({*cal, *img})
    coordinates = (np.asarray(sampled) - np.array(shape) // 2) / shape
    _pd_schedule.cache_clear()
    controls, levels, timings, metrics = [], [], [], []
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for endpoint, (acquired, period, final) in enumerate(
        ((45, 1.8, 140), (20, peripheral_tr, 100))
    ):
        order = pp.make_pclo_order(coordinates, acquired, center_echo=centre, pad=True)
        _, echo_map = maps(order, sampled, shape)
        plane = np.where(np.isfinite(echo_map), echo_map, -1).astype(np.int32)
        begin = perf_counter()
        fitted, angles, latent = optimize_pd_controls(
            acquired + STARTUP,
            [120, 140, final, 0.7],
            startup=STARTUP,
            esp=ESP,
            first_echo=FIRST,
            tr=period,
            centre=STARTUP + centre,
            echo_map=plane,
            t1_ms=module.DESIGN_T1_MS,
            t2_ms=module.DESIGN_T2_MS,
            b1_scales=module.DESIGN_B1_SCALES,
            contrast_pair=module.DESIGN_CONTRAST,
            objective="contrast" if endpoint == 0 else "psf",
            effective_te=effective if endpoint == 0 else None,
        )
        timings.append(perf_counter() - begin)
        controls.append(fitted)
        levels.append(latent)
        baseline, _ = _pd_schedule(
            acquired + STARTUP, 120, 140, final, 0.7, STARTUP, ESP, FIRST, 1.0, 0.1, 1.3
        )
        with execution("cpu"):
            model = reference_model(
                acquired + STARTUP, [(1000, 60), (1000, 100), (1000, 150)]
            )
            signals = (
                model.simulate(
                    flip=torch.as_tensor(np.rad2deg(np.stack([baseline, angles]))),
                    TR=period * 1e3,
                    repetitions=8,
                )
                .abs()
                .numpy()
            )
        energy_form, moment_form = psf_weights(plane, acquired)

        def rms(signal, energy_form=energy_form, moment_form=moment_form):
            signal = signal[..., STARTUP:]
            energy = np.einsum("bti,ij,btj->bt", signal, energy_form, signal)
            moment = np.einsum("bti,ij,btj->bt", signal, moment_form, signal)
            return np.sqrt(np.maximum(moment, 0) / np.maximum(energy, 1e-20))

        radius = rms(signals)
        metrics.append(
            {
                "RF_energy_ratio": float(np.sum(angles**2) / np.sum(baseline**2)),
                "reference_tissue_PSF_RMS_baseline": radius[0].tolist(),
                "reference_tissue_PSF_RMS_fitted": radius[1].tolist(),
            }
        )
        for tissue, t2 in enumerate((60, 100, 150)):
            axes[endpoint, 0].plot(
                np.arange(1, acquired + STARTUP + 1),
                signals[1, tissue],
                label=f"T2 {t2} ms",
            )
        axes[endpoint, 0].axvspan(0.5, STARTUP + 0.5, color="0.92")
        axes[endpoint, 0].set(
            title=f"{'Centre' if endpoint == 0 else 'Periphery'} · ETL {acquired}, TR {period * 1000:.1f} ms",
            xlabel="RF / echo number including startup",
            ylabel="Steady-state magnitude / M0",
        )
        axes[endpoint, 0].legend()
        axes[endpoint, 1].plot(np.rad2deg(baseline), "--", label="Prescribed")
        axes[endpoint, 1].plot(np.rad2deg(angles), label="Forward EPG fit")
        axes[endpoint, 1].set(
            title=f"CPU endpoint fit {timings[-1]:.3f} s",
            xlabel="RF index, zero based",
            ylabel="Angle (°)",
        )
        axes[endpoint, 1].legend()
    figure.suptitle(
        "PD prescribed signal design · fixed ESP 3.416 ms · CPU forward EPG refinement"
    )
    figure.savefig(output / "pd_endpoints.png", dpi=160)
    count = 586
    step = module.cubic(np.linspace(0, 1, count))
    lengths = np.rint(45 - 25 * step).astype(int)
    paper_tr = 1.8 - 1.65 * step
    feasible_tr = 1.8 + (peripheral_tr - 1.8) * step
    transition_controls = controls[0] + step[:, None] * (controls[1] - controls[0])
    seed = levels[0] + step[:, None] * (levels[1] - levels[0])
    np.savez(
        output / "pd_transition_inputs.npz",
        lengths=lengths + STARTUP,
        controls=transition_controls,
        seed=seed,
    )
    begin = perf_counter()
    profiles = _make_pd_schedules(
        lengths + STARTUP, transition_controls, STARTUP, ESP, FIRST, seed=seed
    )
    construction = perf_counter() - begin
    assert np.all(profiles[np.arange(51)[None, :] >= (lengths + STARTUP)[:, None]] == 0)
    np.savez(
        output / "pd_profiles.npz",
        profiles_deg=np.rad2deg(profiles),
        acquired_ETL=lengths,
        TR_published=paper_tr,
        TR_feasible=feasible_tr,
        controls_deg=transition_controls,
    )
    figure, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    raw_times = []
    for ax, etl in zip(axes.flat, (20, 40, 45, 80, 120, 160), strict=True):
        _pd_schedule.cache_clear()
        begin = perf_counter()
        angles = pp.make_pd_schedule(etl + STARTUP, ESP, first_echo_time=FIRST)
        elapsed = perf_counter() - begin
        raw_times.append({"acquired_ETL": etl, "inverse_seconds_cold": elapsed})
        ax.plot(np.arange(1, len(angles) + 1), np.rad2deg(angles))
        ax.axvspan(0.5, STARTUP + 0.5, color="0.92")
        ax.set(
            title=f"ETL {etl}{' · S13 length' if etl == 80 else ''} | inverse EPG {elapsed * 1000:.0f} ms",
            xlabel="RF pulse number including startup",
            ylabel="Angle (°)",
            ylim=(0, 155),
        )
        ax.grid(alpha=0.15)
    figure.suptitle(
        "Independent PD trains · first 120°, maximum 140°, final 100°\nFixed ESP 3.416 ms; first echo 4.162 ms · CPU, one thread, imports excluded"
    )
    figure.savefig(output / "pd_multiple_etls.png", dpi=160)
    report = {
        "design": "Independent Mugler-style prescribed-signal PD, not a PDvar* reconstruction",
        "ESP_ms": ESP * 1000,
        "first_echo_ms": FIRST * 1000,
        "startup": STARTUP,
        "reference_T1_T2_ms": [1000, 100],
        "requested_effective_TE_ms": 28,
        "prescribed_effective_TE_ms": effective * 1000,
        "fitted_effective_TE_ms": float(
            pp.calc_fse_effective_te(profiles[0], ESP, first_echo_time=FIRST)[
                STARTUP + centre
            ]
            * 1000
        ),
        "physical_centre_TE_ms": (FIRST + ESP * (STARTUP + centre)) * 1000,
        "centre_acquired_zero_based": centre,
        "endpoint_controls": np.asarray(controls).tolist(),
        "CPU_endpoint_fit_seconds": timings,
        "CPU_batched_586_trains_seconds": construction,
        "CPU_total_fit_and_transition_seconds": sum(timings) + construction,
        "CPU_cold_inverse": raw_times,
        "endpoint_metrics": metrics,
        "minimum_padded_TR_ms": minimum_tr * 1000,
        "extra_padding_seconds": float((feasible_tr - paper_tr).sum()),
        "extra_padding_percent": float((feasible_tr.sum() / paper_tr.sum() - 1) * 100),
        "model_limits": "Fully relaxed inverse EPG; endpoint steady-state refinement on representative PCLO maps, not a joint variable-TR simulation. Instantaneous CPMG pulses, no finite RF profiles or scanner reconstruction.",
    }
    (output / "pd_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
