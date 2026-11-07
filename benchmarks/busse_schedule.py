"""Compare one prescribed-PSS family across ETLs and time CPU control fits.

Run with the design/plot extras installed::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/busse_schedule.py --output /tmp/busse

Imports and native model initialization are excluded from fit timings. The
first setup includes the PSF matrices. Timings are medians of three cold
schedule constructions or three fits; no constructor cache is reused between
fits. All fits use the same initial 25/70/120 degree prescription.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from blochsim._execution import execution
from blochsim.simulators import FSESimulator

import pypulseqpp as pp
from pypulseqpp import sequences
from pypulseqpp._busse import _schedule
from pypulseqpp._busse_opt import optimize_controls, psf_weights


def echo_map(acquired):
    """Actual adaptive radial view order for the S13 matrix and acceleration."""
    module = sequences.fse3D_sequence
    cal, image = pp.make_cartesian_plane_sampling(
        (276, 140),
        (2, 2),
        (24, 24),
        caipi_shift=1,
        elliptical=True,
        train_length=acquired,
    )
    sampled = sorted({*cal, *image})
    coordinates = (np.asarray(sampled) - [138, 70]) / [276, 140]
    lengths, _, place = module.shot_parameters(
        len(sampled), acquired, acquired, 3.431, 3.431
    )
    order = module.deal_trains(
        coordinates, lengths, np.zeros_like(place), (acquired - 1) // 2, acquired
    )
    result = np.full((276, 140), -1, dtype=np.int32)
    for train in order:
        for echo, index in enumerate(train):
            if index is not None:
                result[sampled[index]] = echo
    assert result[138, 70] == (acquired - 1) // 2
    return result


def validate_fit(prescribed, fitted, plane, centre):
    """Re-evaluate with twice the state count and repetition count."""
    length = len(fitted)
    module = sequences.fse3D_sequence

    class TimedFSE(FSESimulator):
        def layout(
            self, *, flip, ESP, phases=0.0, exc_flip=90.0, exc_phase=90.0, TR=1e6
        ):
            program = super().layout(
                flip=flip,
                ESP=ESP,
                phases=phases,
                exc_flip=exc_flip,
                exc_phase=exc_phase,
                TR=TR,
            )
            return [
                (time + (0.000746 / 2 if i == 1 else 0.000746 if i > 1 else 0), op)
                for i, (time, op) in enumerate(program)
            ]

    with execution("cpu"):
        model = TimedFSE(
            ESP=3.416,
            states=4 * length + 1,
            T1=list(module.DESIGN_T1_MS) * 3,
            T2=list(module.DESIGN_T2_MS) * 3,
            B1=np.repeat(module.DESIGN_B1_SCALES, 3).tolist(),
        )
        signal = (
            model.simulate(
                flip=torch.as_tensor(
                    np.rad2deg(np.asarray([prescribed, fitted])), dtype=torch.float64
                ),
                TR=3431.0,
                repetitions=16,
            )
            .abs()
            .numpy()
        )
        lower = (
            TimedFSE(
                ESP=3.416,
                states=2 * length + 1,
                T1=list(module.DESIGN_T1_MS) * 3,
                T2=list(module.DESIGN_T2_MS) * 3,
                B1=np.repeat(module.DESIGN_B1_SCALES, 3).tolist(),
            )
            .simulate(
                flip=torch.as_tensor(
                    np.rad2deg(np.asarray([prescribed, fitted])), dtype=torch.float64
                ),
                TR=3431.0,
                repetitions=8,
            )
            .abs()
            .numpy()
        )
    discrepancy = float(np.linalg.norm(signal - lower) / np.linalg.norm(signal))
    assert discrepancy < 0.005
    samples = signal.reshape(2, 3, 3, length)
    contrast = samples[:, :, 2, centre] - samples[:, :, 0, centre]
    contrast_ratio = contrast[1] / contrast[0]
    assert contrast_ratio.min() >= 0.95
    energy, moment = psf_weights(plane, length - 6)
    data = signal[:, :, 6:]
    radius = np.sqrt(
        np.einsum("bti,ij,btj->bt", data, moment, data)
        / np.einsum("bti,ij,btj->bt", data, energy, data)
    ).mean(1)
    assert radius[1] <= radius[0] * 1.0001
    return {
        "high_fidelity_relative_signal_error": discrepancy,
        "high_fidelity_contrast_fraction_per_B1": contrast_ratio.tolist(),
        "prescribed_psf_rms_voxels": float(radius[0]),
        "optimized_psf_rms_voxels": float(radius[1]),
    }


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    # Initialize the native simulator before measuring construction or fitting.
    with execution("cpu"):
        FSESimulator(ESP=3.416, T1=1200.0, T2=35.0, states=16).simulate(
            flip=[120.0] * 8
        )
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[1] / "tests/fixtures/busse_figure1.json"
        ).read_text()
    )
    canonical = pp.make_busse_schedule(106)
    error = float(np.max(abs(np.rad2deg(canonical) - fixture["angles_deg"])))
    assert error < 1e-8
    report = {
        "cpu": next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        "threads": 1,
        "figure1_max_error_deg": error,
        "prescription": "25/70/120 degrees; first/final 120; six startup; acquired midpoint; initial decay 1.3 echoes",
        "timing": "Median of three, imports/native initialization excluded. Fit includes baseline model but excludes PSF setup. Constructor cache cleared each run.",
        "ESP_ms": 3.416,
        "first_echo_ms": 4.162,
        "TR_s": 3.431,
        "ensemble_T1_ms": list(sequences.fse3D_sequence.DESIGN_T1_MS),
        "ensemble_T2_ms": list(sequences.fse3D_sequence.DESIGN_T2_MS),
        "ensemble_B1": list(sequences.fse3D_sequence.DESIGN_B1_SCALES),
        "ETLs": {},
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    saved = {"figure1": canonical}
    for acquired, ax in zip((40, 80, 120, 160), axes.flat, strict=True):
        length = acquired + 6
        centre = 6 + (acquired - 1) // 2
        construction = []
        for _ in range(3):
            _schedule.cache_clear()
            start = perf_counter()
            prescribed = pp.make_busse_schedule(length)
            construction.append(perf_counter() - start)
        start = perf_counter()
        plane = echo_map(acquired)
        psf_weights(plane, acquired)
        setup = perf_counter() - start
        timings = []
        for _ in range(3):
            _schedule.cache_clear()
            start = perf_counter()
            fitted = optimize_controls(
                length,
                minimum_flip_angle=np.deg2rad(25),
                centre_flip_angle=np.deg2rad(70),
                maximum_flip_angle=np.deg2rad(120),
                centre_echo=centre,
                startup_echoes=6,
                initial_decay=1.3,
                esp=0.003416,
                first_echo_time=0.004162,
                tr=3.431,
                echo_map=plane,
                t1_ms=sequences.fse3D_sequence.DESIGN_T1_MS,
                t2_ms=sequences.fse3D_sequence.DESIGN_T2_MS,
                b1_scales=sequences.fse3D_sequence.DESIGN_B1_SCALES,
                contrast_pair=sequences.fse3D_sequence.DESIGN_CONTRAST,
            )
            timings.append(perf_counter() - start)
        minimum, middle, final = np.rad2deg(fitted[[np.argmin(fitted), centre, -1]])
        np.testing.assert_allclose(
            fitted,
            pp.make_busse_schedule(
                length,
                minimum_flip_angle=np.deg2rad(minimum),
                centre_flip_angle=np.deg2rad(middle),
            ),
            atol=1e-8,
        )
        record = {
            "constructor_seconds": median(construction),
            "psf_setup_seconds": setup,
            "fit_seconds": median(timings),
            "fit_runs_seconds": timings,
            "minimum_deg": float(minimum),
            "centre_deg": float(middle),
            "final_deg": float(final),
            "centre_RF_1based": centre + 1,
            "TE_ms": 4.162 + centre * 3.416,
            "energy_fraction_of_prescribed": float(sum(fitted**2) / sum(prescribed**2)),
        }
        record.update(validate_fit(prescribed, fitted, plane, centre))
        report["ETLs"][acquired] = record
        print(acquired, record, flush=True)
        saved[f"prescribed_{acquired}"] = prescribed
        saved[f"optimized_{acquired}"] = fitted
        x = np.arange(1, length + 1)
        ax.axvspan(0.5, 6.5, color="0.92")
        ax.axvline(centre + 1, color="0.6", linestyle=":")
        ax.plot(
            x,
            np.rad2deg(prescribed),
            color="#2266aa",
            label="Same Figure 1 prescription",
        )
        ax.plot(
            x,
            np.rad2deg(fitted),
            color="#c65f25",
            linestyle="--",
            label="Forward-EPG control fit",
        )
        ax.set_title(
            f"Acquired ETL {acquired}{' (S13 matrix)' if acquired == 80 else ''} · CPU fit {median(timings):.2f} s"
        )
        ax.text(
            0.98,
            0.55,
            f"Fit: {minimum:.1f}° / {middle:.1f}° / {final:.1f}°\nCentre RF {centre + 1} · TE {record['TE_ms']:.1f} ms",
            transform=ax.transAxes,
            ha="right",
        )
        ax.set_xlabel("RF pulse (including six startup pulses)")
        ax.set_ylabel("Refocusing angle (°)")
        ax.set_ylim(15, 125)
        ax.grid(alpha=0.18)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(
        "One prescribed-PSS / inverse-EPG constructor across all ETLs\nFirst/final 120° · acquired midpoint fixed · ESP 3.416 ms affects forward EPG only"
    )
    fig.tight_layout()
    fig.savefig(output / "shared_busse_trains.png", dpi=160)
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.plot(
        np.arange(1, 107),
        fixture["angles_deg"],
        color="0.6",
        linewidth=4,
        label="Working Figure 1 reference",
    )
    axis.plot(
        np.arange(1, 107),
        np.rad2deg(canonical),
        color="#2266aa",
        label=f"Shared constructor (max error {error:.1e}°)",
    )
    axis.axvspan(0.5, 6.5, color="0.92")
    axis.axvline(56, color="0.6", linestyle=":")
    axis.set(
        xlabel="Refocusing pulse (including startup)",
        ylabel="Refocusing angle (°)",
        title="Figure 1 parity · 100 acquired + six startup · 25° / 70° / 120°",
    )
    axis.legend()
    axis.grid(alpha=0.18)
    figure.tight_layout()
    figure.savefig(output / "shared_busse_figure1.png", dpi=160)
    np.savez(output / "shared_busse_trains.npz", **saved)
    (output / "shared_busse_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
