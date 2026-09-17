"""
===================================================
Echo train length, signal envelope and point spread
===================================================

A three-dimensional fast spin echo reads one ``(line, partition)`` view per
refocused echo, so the amplitude the train has left at echo :math:`m` becomes
the weight of whichever view that echo reads. The ordering is the map from echo
index to k-space position, and the weighting it produces is a filter applied to
the image: its inverse Fourier transform is the point-spread function of the
acquisition.

This example takes the refocusing schedule and echo spacing out of designed
sequences, simulates the echo amplitudes with an extended phase graph, assembles
the k-space weighting from the sequence's own view labels, and measures the
width of the resulting point-spread function against the train length and the
acquisition time.
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


def envelope_figure(envelopes, esp_ms):
    """Draw each tissue's echo amplitude against the echo time."""
    figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.0))
    for name, amplitude in envelopes.items():
        axis.plot(
            esp_ms * np.arange(1, amplitude.size + 1),
            amplitude,
            marker="o",
            ms=3,
            label=name,
        )
    axis.set_xlabel("echo time (ms)")
    axis.set_ylabel("echo amplitude")
    axis.set_ylim(bottom=0)
    axis.legend(frameon=False)
    figure.tight_layout()
    return figure


def weighting_figure(panels):
    """Draw the k-space weighting each ordering produces."""
    figure, axes = plt.subplots(
        1, len(panels), figsize=(PAGE_WIDTH, 3.6), squeeze=False
    )
    for axis, (name, weight) in zip(axes[0], panels.items(), strict=True):
        drawn = axis.imshow(weight.T, origin="lower", cmap="magma", vmin=0.0)
        axis.set_title(name)
        axis.set_xlabel("line index")
    axes[0][0].set_ylabel("partition index")
    figure.colorbar(drawn, ax=axes[0], label="echo amplitude", shrink=0.85)
    return figure


def psf_figure(profiles):
    """Draw the point-spread profiles against the ideal one."""
    figure, axis = plt.subplots(figsize=(PAGE_WIDTH, 3.4))
    for name, (offsets, profile) in profiles.items():
        axis.plot(offsets, profile, label=name)
    axis.axhline(0.5, color="0.6", lw=0.8)
    axis.set_xlim(-4, 4)
    axis.set_xlabel("offset (pixels)")
    axis.set_ylabel("point-spread amplitude")
    axis.legend(frameon=False)
    figure.tight_layout()
    return figure


def table(rows, tissues):
    """Print one line per train length, with a width column per tissue."""
    heads = "".join(f"{name:>18}" for name in tissues)
    print(f"{'train':>7}  {'shots':>7}  {'duration':>10}{heads}")
    for row in rows:
        widths = "".join(f"{row['fwhm'][name]:12.3f} px" for name in tissues)
        print(f"{row['etl']:7d}  {row['shots']:7d}  {row['duration']:8.1f} s{widths}")
    print(f"{'no decay':>7}  {'':7}  {'':10}{IDEAL_FWHM:12.3f} px")


# sphinx_gallery_end_ignore
import numpy as np
import torchsim

from pypulseqpp import sequences

#: The field of view and matrix every configuration below is designed at.
PRESCRIPTION = {
    "fov_x": 200e-3,
    "fov_y": 200e-3,
    "fov_z": 200e-3,
    "n_x": 64,
    "n_y": 48,
    "n_z": 48,
}

#: Relaxation times (ms) of the two tissues the envelopes are simulated for,
#: cartilage and synovial fluid.
TISSUES = {"cartilage": (1200.0, 35.0), "synovial fluid": (3600.0, 250.0)}

# %%
# The echo envelope
# -----------------
#
# The design writes its refocusing schedule and echo spacing into the sequence,
# so the envelope is simulated from what will be played rather than from what
# was asked for. ``flip_modulation="optimized"`` designs the schedule with
# torchsim against a prescribed contrast, which leaves the leading echoes below
# 180 degrees and the rest of the train at full refocusing.

reference = sequences.fse3D_sequence(
    **PRESCRIPTION,
    te=None,
    tr=1.0,
    etl=32,
    refocusing_angle_deg=180.0,
    ordering="radial",
    flip_modulation="optimized",
)
angles = np.asarray(reference.get_definition("RefocusingFlipAngles"))
esp_ms = 1e3 * reference.get_definition("EchoSpacing")[0]

envelopes = {
    name: np.abs(np.asarray(torchsim.fse_sim(flip=angles, ESP=esp_ms, T1=t1, T2=t2)))
    for name, (t1, t2) in TISSUES.items()
}

# %%
# Over a train this long the envelope follows the transverse relaxation of each
# tissue: the schedule shapes the first few echoes, and from there the
# amplitude is set by :math:`T_2` and the echo spacing. The two tissues
# therefore reach the end of the train with very different amplitudes, and the
# weighting below is the short-:math:`T_2` one.

# sphinx_gallery_start_ignore
envelope_figure(envelopes, esp_ms)
# sphinx_gallery_end_ignore

# %%
# From the echo index to a k-space weighting
# ------------------------------------------
#
# The view labels record which line and partition each acquisition read and at
# which echo, so the weighting is assembled from the sequence rather than from
# the ordering rule. Radial ordering sorts views by their distance from the
# centre of k-space, which makes the weighting a radially symmetric filter.


def weighting(seq, amplitude):
    """The amplitude each acquired view is read with, as a ``(line, partition)`` map."""
    labels = seq.evaluate_labels(evolution="adc")
    weight = np.zeros((PRESCRIPTION["n_y"], PRESCRIPTION["n_z"]))
    weight[labels["LIN"], labels["PAR"]] = amplitude[labels["ECO"]]
    return weight


shuffled = sequences.fse3D_sequence(
    **PRESCRIPTION,
    te=None,
    tr=1.0,
    etl=32,
    ordering="shuffling",
    flip_modulation="optimized",
)

cartilage = envelopes["cartilage"]
orderings = {
    "radial": weighting(reference, cartilage),
    "shuffled": weighting(shuffled, cartilage),
}

# sphinx_gallery_start_ignore
weighting_figure(orderings)
# sphinx_gallery_end_ignore

# %%
# The weighting decreases outward from the centre, and the elliptical edge is
# the corner of k-space the design leaves unsampled.
#
# The measurement below assumes a weighting of this shape, so it applies to
# distance ordering and not to every ordering the design offers. Under
# ``ordering="shuffling"`` the centre of k-space is read at many different
# echoes, and no single amplitude weights it; the printed echo indices below
# show the difference. A shuffled acquisition is reconstructed jointly over the
# decay rather than as one image with one point-spread function, which is a
# different subject from this one.

# sphinx_gallery_start_ignore
radial_echoes = reference.evaluate_labels(evolution="adc")
shuffled_echoes = shuffled.evaluate_labels(evolution="adc")


def central_echoes(labels, radius=4):
    """Echo indices that read views within ``radius`` of the centre."""
    centre = 0.5 * (PRESCRIPTION["n_y"] - 1), 0.5 * (PRESCRIPTION["n_z"] - 1)
    inside = np.hypot(labels["LIN"] - centre[0], labels["PAR"] - centre[1]) <= radius
    return np.unique(labels["ECO"][inside])


print(
    "echoes reading the central 4-pixel disc:"
    f"\n  radial    {central_echoes(radial_echoes)}"
    f"\n  shuffled  {central_echoes(shuffled_echoes)}"
)
# sphinx_gallery_end_ignore

# %%
# Train length against point spread
# ---------------------------------
#
# The train length sets how many excitations the volume takes and how far the
# envelope has decayed by the edge of k-space. The point-spread function is the
# inverse Fourier transform of the weighting, zero-padded so its width can be
# measured between samples, and its width is quoted against the width the same
# sampled region gives with no decay at all.


#: Factor the weighting is zero-padded by before its transform, so the
#: point-spread function is sampled finely enough to interpolate a width on.
PAD = 16


def point_spread(weight, pad=PAD):
    """Central profile of the point-spread function, in pixels of offset."""
    size = weight.shape[0]
    padded = np.zeros((size * pad, size * pad))
    corner = (size * pad - size) // 2
    padded[corner : corner + size, corner : corner + size] = weight
    psf = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(padded))))
    profile = psf[:, psf.shape[1] // 2]
    profile = profile / profile.max()
    offsets = (np.arange(profile.size) - profile.size // 2) / pad
    return offsets, profile


def full_width(offsets, profile):
    """Width of the profile at half its maximum, in pixels."""
    spacing = offsets[1] - offsets[0]
    peak = int(np.argmax(profile))
    edges = []
    for step in (-1, 1):
        index = peak
        while 0 < index < profile.size - 1 and profile[index] > 0.5:
            index += step
        below, above = profile[index], profile[index - step]
        edges.append(offsets[index] - step * spacing * (0.5 - below) / (above - below))
    return edges[1] - edges[0]


# %%
# The train lengths differ in nothing else: the same views are read, at the same
# echo spacing, with the schedule each length is designed with.

rows = []
profiles = {}
for etl in (8, 16, 32, 48):
    seq = sequences.fse3D_sequence(
        **PRESCRIPTION,
        te=None,
        tr=1.0,
        etl=etl,
        refocusing_angle_deg=180.0,
        ordering="radial",
        flip_modulation="optimized",
    )
    schedule = np.asarray(seq.get_definition("RefocusingFlipAngles"))
    spacing_ms = 1e3 * seq.get_definition("EchoSpacing")[0]
    widths = {}
    for name, (t1, t2) in TISSUES.items():
        amplitude = np.abs(
            np.asarray(torchsim.fse_sim(flip=schedule, ESP=spacing_ms, T1=t1, T2=t2))
        )
        offsets, profile = point_spread(weighting(seq, amplitude))
        widths[name] = full_width(offsets, profile)
        if name == "cartilage":
            profiles[f"{etl} echoes"] = (offsets, profile)
    rows.append(
        {
            "etl": etl,
            "shots": int(seq.get_definition("NumShots")[0]),
            "duration": seq.duration()[0],
            "fwhm": widths,
        }
    )

# sphinx_gallery_start_ignore
IDEAL_OFFSETS, IDEAL_PROFILE = point_spread(
    weighting(reference, np.ones_like(cartilage))
)
IDEAL_FWHM = full_width(IDEAL_OFFSETS, IDEAL_PROFILE)
profiles["no decay"] = (IDEAL_OFFSETS, IDEAL_PROFILE)
table(rows, TISSUES)
psf_figure(profiles)
# sphinx_gallery_end_ignore

# %%
# Every acquired view is read at every train length, so the acquisition time
# is inversely proportional to the train length while the width is not. The
# early steps exchange a large fraction of the acquisition time for a small
# fraction of width and the late ones exchange comparable fractions of both,
# which places the useful operating point where the envelope is still close to
# its maximum when the ordering reaches the edge of k-space. Where that falls
# is set by the echo spacing and by the :math:`T_2` the sharpness is wanted
# for, which is why the two tissues are not blurred alike by the same train.
# The profiles drawn are the short-:math:`T_2` ones.
