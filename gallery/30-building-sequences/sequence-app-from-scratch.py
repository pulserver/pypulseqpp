"""
==============================
A SequenceApp from scratch
==============================

A complete sequence separates three things: the prescription a user sets, the
order the views are sampled in, and the blocks of one repetition.
:class:`~pypulseqpp.sequences.SequenceApp` is the contract that keeps them
apart, and every sequence the package ships is written against it. The
architecture is described in
:doc:`/explanations/design/sequence-application`.

This page implements a slice-selective, RF-spoiled two-dimensional gradient
echo against that contract.
"""

# %%
# The three methods
# -----------------
#
# ``init_sequence`` receives the prescription and designs the modules and the
# sampling order from it. ``loop`` calls ``kernel`` once per repetition.
# ``finalize`` records the definitions a reconstruction reads. Settings a user
# does not prescribe are class attributes, and ``MAX_GRAD`` and ``MAX_SLEW``
# have no default, so every application states them.

import numpy as np

import pypulseqpp as pp
import pypulseqpp.sequences as design
from pypulseqpp.sequences import SequenceApp


class SpoiledGradientEcho(SequenceApp):
    """RF-spoiled 2D Cartesian gradient echo, one line per repetition."""

    NAME = "tour_gre_2d"
    MAX_GRAD = 40.0
    MAX_SLEW = 150.0
    PULSE_DURATION = 3e-3
    RF_SPOILING_INCREMENT_DEG = 117.0

    def init_sequence(
        self,
        fov: float = 220e-3,
        matrix: int = 128,
        slice_thickness: float = 5e-3,
        flip_angle_deg: float = 12.0,
        te: float | None = None,
        tr: float | None = None,
    ) -> None:
        """Design the excitation, the readout and the line order.

        Parameters
        ----------
        fov : float, optional
            Isotropic field of view (m).
        matrix : int, optional
            Matrix size along both encoded axes.
        slice_thickness : float, optional
            Slice thickness (m).
        flip_angle_deg : float, optional
            Excitation flip angle (degrees).
        te : float | None, optional
            Echo time (s). ``None`` is as short as the readout admits.
        tr : float | None, optional
            Repetition time (s). ``None`` is as short as possible.
        """
        self.matrix = matrix
        self.excitation = design.SpatialSelectiveExcitation(
            self.system,
            flip_angle_deg=flip_angle_deg,
            thickness_m=slice_thickness,
            duration_s=self.PULSE_DURATION,
        )
        self.readout = design.LineReadout2D(
            self.system,
            self.excitation.rf,
            self.excitation.gz,
            self.excitation.gz_reph,
            fov=(fov, fov),
            matrix=(matrix, matrix),
            te=te,
            tr=tr,
        )
        self.lines = list(range(matrix))
        self.phases = np.deg2rad(self.RF_SPOILING_INCREMENT_DEG) * np.cumsum(
            np.arange(len(self.lines) + 1)
        )
        self.fov, self.slice_thickness = fov, slice_thickness

    def loop(self) -> None:
        """Play every phase-encode line in order."""
        for index, line in enumerate(self.lines):
            self.kernel(line, float(self.phases[index]))

    def kernel(self, line: int, phase: float) -> None:
        """One repetition: excitation, encoding, acquisition, spoiling."""
        readout = self.readout
        readout.rf.phase_offset = phase % (2 * np.pi)
        readout.adc.phase_offset = phase % (2 * np.pi)
        step = (line - self.matrix // 2) / (self.matrix / 2)
        self.seq.add_block(readout.rf, readout.gz, *self.labels(LIN=line))
        self.seq.add_block(
            readout.gx_pre, pp.scale_grad(readout.gy_pre, step), readout.gz_reph
        )
        self.seq.add_block(readout.gx, readout.adc)
        self.seq.add_block(readout.gx_spoil, pp.scale_grad(readout.gy_rew, step))
        if getattr(readout, "wait_tr", None) is not None:
            self.seq.add_block(readout.wait_tr)

    def finalize(self) -> None:
        """Record the geometry a reconstruction reads."""
        self.seq.set_definition("FOV", [self.fov, self.fov, self.slice_thickness])
        self.seq.set_definition("Matrix", [self.matrix, self.matrix, 1])
        self.seq.set_definition("Name", self.NAME)


# %%
# Designing it
# ------------
#
# Constructing the application runs ``init_sequence`` and therefore designs it;
# ``design`` starts a fresh sequence, runs ``loop``, then ``finalize``.

app = SpoiledGradientEcho(matrix=128)
seq = app.design()
print(f"{seq.num_blocks} blocks, {seq.duration()[0]:.2f} s")
print("timing:", seq.check_timing()[0])
print("definitions:", sorted(seq.definitions))

# %%
# Sequence diagram
# ----------------

seq.paper_plot(tr=32)

# %%
# Acquisition order
# -----------------
#
# ``kernel`` writes a ``LIN`` label on every acquisition, so the order the loop
# played is read back from the sequence rather than reconstructed.

pp.plot.plot_kspace(seq, color_by="order", plane="xy", show_trajectory=False)

# %%
# The command line follows from the signature
# -------------------------------------------
#
# ``main`` is derived from the class, and its parameters are those of
# ``init_sequence``, so a subclass gets a command-line interface without
# declaring one.

main = SpoiledGradientEcho.main
print(str(__import__("inspect").signature(main))[:120], "...")
