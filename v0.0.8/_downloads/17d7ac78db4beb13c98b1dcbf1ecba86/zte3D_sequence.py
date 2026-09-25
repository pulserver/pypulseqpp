"""
===================
3D zero echo time
===================

A short non-selective excitation is applied while the radial readout gradient is
already at amplitude. Acquisition begins after the transmit/receive dead time,
without gradient-echo formation; spoiling suppresses residual transverse
magnetisation between repetitions. Contrast depends on TR, flip angle, RF
bandwidth, and very short-T2 decay. ZTE is used for anatomical imaging of
short-T2 tissues and other minimal-TE applications.
"""

# sphinx_gallery_start_ignore
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 110,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    }
)


# sphinx_gallery_end_ignore

# %%
# Nyquist angular sampling
# ------------------------
#
# ``ceil(pi * n**2)`` half-spoke directions over the sphere, which sample its
# surface at the Nyquist spacing.

import pypulseqpp as pp
from pypulseqpp.sequences import zte3D_sequence

baseline = zte3D_sequence(n=64, n_dummy=0)
print(f"{baseline.num_blocks} blocks, {baseline.duration()[0]:.2f} s")
print(
    f"{int(baseline.get_definition('NumShots')[0])} shots, "
    f"TR {baseline.get_definition('TR')[0] * 1e6:.0f} us"
)

# %%
# Sequence diagram
# ----------------
#
# The automatically detected repetition is one shell of half-spoke
# directions. The readout gradient reaches amplitude before the hard RF event,
# and the ADC window starts after the transmit/receive dead time. The solid trace is a
# representative repetition; shaded traces show other gradient encodes.

baseline.paper_plot()

# %%
# Sampling order
# --------------
#
# The half-spokes over the three k-space axes. Each starts at the centre of
# k-space and runs outward to the surface of the sampled sphere.

pp.plot.plot_kspace(baseline, color_by="shot")

# %%
# Angular undersampling
# ---------------------
#
# The Nyquist set of directions is divided into shells, each a copy of the
# first rotated about ``z``. ``r`` acquires one shell in every ``r``: at
# ``r = 2`` the number of acquisitions is halved and the azimuthal spacing
# between acquired shells is doubled. Angular undersampling produces streak
# artefacts from the k-space periphery rather than the fold-over of an
# undersampled Cartesian acquisition.

alternative = zte3D_sequence(n=64, r=2, n_dummy=0)

# sphinx_gallery_start_ignore
print(f"{'':16} {'blocks':>8} {'duration (s)':>13} {'acquisitions':>13}")
for name, seq in (("every shell", baseline), ("every second", alternative)):
    print(
        f"{name:16} {seq.num_blocks:8d} {seq.duration()[0]:13.2f} "
        f"{seq._native.num_adc():13d}"
    )
# sphinx_gallery_end_ignore

# %%
pp.plot.plot_kspace(alternative, color_by="shot")

# %%
