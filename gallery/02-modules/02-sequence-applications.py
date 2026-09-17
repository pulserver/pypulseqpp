"""
==========================================
Sequence applications and the command line
==========================================

Turning modules and a scan loop into a
:class:`~pypulseqpp.sequences.SequenceApp`, the contract every sequence the
package ships is written against.

An application separates three things a script mixes together: the
prescription, in ``init_sequence``; the sampling order, in ``loop``; and one
repetition's blocks, in ``kernel``. The prescription is also the command-line
interface, which is derived from the signature and the ``Parameters`` section
of ``init_sequence`` rather than declared a second time.
"""

# sphinx_gallery_start_ignore
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAGE_WIDTH = 8.6  # inches, the width of the documentation column

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "figure.figsize": (PAGE_WIDTH, 3.4),
        "savefig.dpi": 110,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
    }
)
# sphinx_gallery_end_ignore
import contextlib

import numpy as np

import pypulseqpp as pp
from pypulseqpp import cli, safety, sequences

# %%
# The application
# ---------------
#
# ``MAX_GRAD`` and ``MAX_SLEW`` are the gradient ceilings the sequence is
# designed under; every application states both, and the system limits are
# capped to them. The remaining class attributes are the settings a user does
# not prescribe.
#
# ``kernel`` adds the blocks of one repetition, ``loop`` calls it once per
# repetition in play order, and ``finalize`` writes the definitions a
# reconstruction reads the prescription from.


class InversionRecoveryGre(sequences.SequenceApp):
    """Segmented inversion-prepared 2D Cartesian gradient echo.

    One adiabatic inversion opens each shot, which then reads ``etl``
    consecutive phase-encode lines under RF spoiling.
    """

    NAME = "ir_gre_2d"
    MAX_GRAD = 32.0
    MAX_SLEW = 130.0
    RF_SPOILING_INCREMENT_DEG = 117.0
    SPOILING_CYCLES = 4.0

    def init_sequence(
        self,
        fov: float = 220e-3,
        n_x: int = 128,
        n_y: int = 96,
        slice_thickness: float = 5e-3,
        flip_angle_deg: float = 8.0,
        ti: float = 900e-3,
        etl: int = 32,
    ) -> None:
        """Design the inversion, the excitation, the readout and the shots.

        Parameters
        ----------
        fov : float, optional
            Field of view along both encoded axes (m).
        n_x, n_y : int, optional
            Matrix size along the readout and the phase encode.
        slice_thickness : float, optional
            Slice thickness (m).
        flip_angle_deg : float, optional
            Excitation flip angle (degrees).
        ti : float, optional
            Inversion time (s), from the inversion to the first excitation of
            the shot.
        etl : int, optional
            Phase-encode lines read after each inversion.
        """
        system = self.system
        self.fov, self.matrix = fov, (n_x, n_y)
        self.slice_thickness = slice_thickness

        self.inversion = sequences.InversionPreparation(system, duration_s=10e-3)
        self.excitation = sequences.SpatialSelectiveExcitation(
            system, flip_angle_deg, slice_thickness, duration_s=3e-3
        )
        self.readout = sequences.LineReadout2D(
            system,
            self.excitation.rf,
            self.excitation.gz,
            self.excitation.gz_reph,
            fov=(fov, fov),
            matrix=(n_x, n_y),
            te=None,
            spoiling_cycles=self.SPOILING_CYCLES,
        )
        self.recovery = pp.make_delay(
            pp.round_to_raster(ti, system.block_duration_raster)
        )
        self.shots = np.array_split(np.arange(n_y), max(1, n_y // etl))
        self.repetition_time = self.readout.duration

    def loop(self) -> None:
        """Play every shot: the inversion, the recovery interval, then its lines."""
        phase, increment = 0.0, 0.0
        for shot_index, shot in enumerate(self.shots):
            for block in self.inversion.blocks:
                self.seq.add_block(*block)
            self.seq.add_block(self.recovery)
            for line in shot:
                increment += np.deg2rad(self.RF_SPOILING_INCREMENT_DEG)
                phase = (phase + increment) % (2 * np.pi)
                self.kernel(int(line), shot_index, phase)

    def kernel(self, line: int, shot: int, phase: float) -> None:
        """One excitation reading phase-encode line ``line``."""
        excitation, readout, seq = self.excitation, self.readout, self.seq
        excitation.rf.phase_offset = phase
        readout.adc.phase_offset = phase
        step = (line - self.matrix[1] // 2) / (self.matrix[1] / 2)

        seq.add_block(excitation.rf, excitation.gz, *self.labels(LIN=line, SEG=shot))
        seq.add_block(
            readout.gx_pre, pp.scale_grad(readout.gy_pre, step), readout.gz_reph
        )
        seq.add_block(readout.gx, readout.adc)
        seq.add_block(readout.gx_spoil, pp.scale_grad(readout.gy_rew, step))

    def finalize(self) -> None:
        """Record the prescription and the k-space geometry as definitions."""
        n_x, n_y = self.matrix
        for key, value in {
            "FOV": [self.fov, self.fov, self.slice_thickness],
            "Matrix": [n_x, n_y, 1],
            "Name": self.NAME,
            "TE": self.readout.echo_time,
            "TR": self.repetition_time,
            "kSpaceCenterLine": n_y // 2,
            "kSpaceCenterSample": self.readout.center_sample,
        }.items():
            self.seq.set_definition(key=key, value=value)


# %%
# Designing
# ---------
#
# Constructing the application designs it; nothing is played until
# :meth:`~pypulseqpp.sequences.SequenceApp.design`, which starts a new sequence,
# runs the loop and then ``finalize``.

application = InversionRecoveryGre(ti=1100e-3, etl=24)
seq = application.design()
print(f"{seq.num_blocks} blocks, {seq.duration()[0]:.2f} s")
print(f"timing: {'ok' if seq.check_timing()[0] else 'failed'}")
print(f"TE {seq.get_definition('TE')[0] * 1e3:.2f} ms")

seq.paper_plot(time_range=[0.0, 1.3])

# %%
# :meth:`~pypulseqpp.sequences.SequenceApp.protocol` returns the prescription
# with the defaults the application declares, which is what the command line
# and a protocol editor present.

for name, default in application.protocol().items():
    print(f"{name:18} {default}")

# %%
# A setting under a different value
# ---------------------------------
#
# A class attribute is a setting a user does not prescribe. A subclass that
# changes one, and nothing else, is the same sequence designed under a
# different limit — here a gradient slew ceiling low enough to bring the
# stimulation estimate of :doc:`../01-basics/02-analysis-and-checks` back under
# its threshold.


from pypulseq.utils.safe_pns_prediction import safe_example_hw


class GentleInversionRecoveryGre(InversionRecoveryGre):
    """The same sequence, designed under a lower gradient slew ceiling."""

    MAX_SLEW = 60.0


for application in (InversionRecoveryGre(), GentleInversionRecoveryGre()):
    designed = application.design()
    _, report = safety.check_pns(designed, safe_example_hw())
    print(
        f"{type(application).__name__:32} "
        f"max_slew {application.system.max_slew / application.system.gamma:6.1f} T/m/s "
        f"pns {100 * report.peak.value:5.1f}% of threshold"
    )

# %%
# Writing the scan
# ----------------
#
# :meth:`~pypulseqpp.sequences.SequenceApp.write` designs the chain and writes
# it: the prescans listed by
# :meth:`~pypulseqpp.sequences.SequenceApp.prescans` as separate files ahead of
# the main one, each naming the next as its ``NextSequence``, so every file
# stays one repeating unit.

# sphinx_gallery_start_ignore
import tempfile
from pathlib import Path

directory = tempfile.TemporaryDirectory()
base = Path(directory.name)
# sphinx_gallery_end_ignore
written = InversionRecoveryGre().write(base / "ir_gre.seq")
for path in written:
    print(f"{Path(path).name}: {Path(path).stat().st_size / 1024:.0f} kB")
# sphinx_gallery_start_ignore
directory.cleanup()
# sphinx_gallery_end_ignore

# %%
# The command line
# ----------------
#
# ``main = <App>.main`` is the module-level entry point every shipped sequence
# exposes. :func:`~pypulseqpp.cli.run` derives the options from its signature,
# which is ``init_sequence``'s, and the help text from that method's
# ``Parameters`` section, so a prescription is documented once.

main = InversionRecoveryGre.main
with contextlib.suppress(SystemExit):  # --help exits, as argparse does
    cli.run(main, ["--help"])
