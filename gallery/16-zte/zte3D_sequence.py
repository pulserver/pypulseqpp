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

baseline = zte3D_sequence(n_x=64, n_views=300, n_dummy=0)
print(f"{baseline.num_blocks} blocks, {baseline.duration()[0]:.2f} s")
print(
    f"{int(baseline.get_definition('NumShots')[0])} shots, "
    f"TR {baseline.get_definition('TR')[0] * 1e6:.0f} us"
)

# %%
# Sequence diagram
# ----------------
#
# Every half-spoke carries its own gradient direction, so the repeating unit
# the diagram would otherwise draw is a whole set of directions. A window of a
# few milliseconds shows the unit that matters: the gradient is already on when
# the hard pulse plays, the ADC opens as soon as the transmitter has settled,
# and the amplitude steps to the next direction between spokes.

baseline.paper_plot(time_range=(0, 2e-3))

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
# ``n_views`` sets how many half-spokes are played. Fewer of them shortens the
# scan and undersamples the surface of the sphere, which shows as streaks rather
# than as aliasing. Both configurations here play far fewer views than the
# matrix asks for, so that the individual spokes stay visible on the page.

alternative = zte3D_sequence(n_x=64, n_views=120, n_dummy=0)

# sphinx_gallery_start_ignore
print(f"{'':16} {'blocks':>8} {'duration (s)':>13} {'acquisitions':>13}")
for name, seq in (("Nyquist", baseline), ("half the views", alternative)):
    print(
        f"{name:16} {seq.num_blocks:8d} {seq.duration()[0]:13.2f} "
        f"{seq._native.num_adc():13d}"
    )
# sphinx_gallery_end_ignore

# %%
pp.plot.plot_kspace(alternative, color_by="shot")

# %%
