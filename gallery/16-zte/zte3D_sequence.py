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
# Baseline
# --------
#
# Half-spokes turned over a sphere, at enough views to sample its surface.

import pypulseqpp as pp
from pypulseqpp.sequences import zte3D_sequence

baseline = zte3D_sequence(n_x=64, n_views=None, n_dummy=0)
print(f"{baseline.num_blocks} blocks, {baseline.duration()[0]:.2f} s")
print(
    f"{int(baseline.get_definition('NumShots')[0])} shots, "
    f"TR {baseline.get_definition('TR')[0] * 1e6:.0f} us"
)

# %%
# Sequence diagram
# ----------------
#
# The automatically detected repetition contains one complete set of
# half-spoke directions. The readout gradient precedes the hard RF event, and the ADC window starts
# after the transmit/receive dead time. The solid trace is a
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
# Fewer views
# -----------
#
# ``n_views`` sets the half-spokes per shell. The default balances their angular
# spacing against the spacing between shells for the requested matrix. Halving
# this count shortens the scan and undersamples one angular direction, producing
# streaking rather than Cartesian aliasing.

alternative = zte3D_sequence(n_x=64, n_views=33, n_dummy=0)

# sphinx_gallery_start_ignore
print(f"{'':16} {'blocks':>8} {'duration (s)':>13} {'acquisitions':>13}")
for name, seq in (("balanced", baseline), ("half the views", alternative)):
    print(
        f"{name:16} {seq.num_blocks:8d} {seq.duration()[0]:13.2f} "
        f"{seq._native.num_adc():13d}"
    )
# sphinx_gallery_end_ignore

# %%
pp.plot.plot_kspace(alternative, color_by="shot")

# %%
