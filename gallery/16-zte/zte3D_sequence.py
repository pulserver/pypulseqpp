"""
===================
3D zero echo time
===================

The readout gradient is already at amplitude when the hard pulse is
transmitted, so acquisition begins as soon as the receiver is available and
the echo time is a few tens of microseconds. What the pulse cannot excite
during the gradient, and what the dead time costs at the centre of k-space,
are the price of it.
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


def safety_table(rows):
    """Print a check, its verdict and its peak, one per line."""
    print(f"{'check':26} {'result':8} {'peak':>22}")
    for name, ok, peak in rows:
        print(f"{name:26} {'pass' if ok else 'FAIL':8} {peak:>22}")


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
# half-spoke directions. The gradient is already on when each hard pulse plays,
# and the ADC window opens after the transmit dead time. The solid trace is a
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
# Safety checks
# -------------
#
# A passing check does not establish that a sequence is safe to run on a
# scanner or on a subject. The nerve model below is a demonstration, not a
# scanner's.

from pypulseqpp import safety

model = safety.ChronaxieModel(chronaxie=334e-6, rheobase=23.4, alpha=0.333)
grad_ok, grad = safety.check_max_grad(baseline)
slew_ok, slew = safety.check_max_slew(baseline)
cont_ok, cont = safety.check_grad_continuity(baseline)
pns_ok, pns = safety.check_pns(baseline, model)

# sphinx_gallery_start_ignore
safety_table(
    [
        (
            "gradient amplitude",
            grad_ok,
            f"{grad.per_axis.value / baseline.system.gamma * 1e3:.1f} mT/m",
        ),
        (
            "slew rate",
            slew_ok,
            f"{slew.per_axis.value / baseline.system.gamma:.0f} T/m/s",
        ),
        (
            "gradient continuity",
            cont_ok,
            f"{len(cont.discontinuities)} discontinuities",
        ),
        ("peripheral nerve stimulation", pns_ok, f"{pns.peak.value:.2f} of threshold"),
    ],
)
# sphinx_gallery_end_ignore
