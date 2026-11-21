"""Spoiler and RF-spoiling audit of the shipped non-Cartesian sequences.

Builds each sequence, walks its blocks repetition by repetition, and reports
for each one:

* the end-of-TR spoiler: the gradient blocks after the last block that drives
  an in-plane (x or y) gradient, their logical moment, whether they carry a
  rotation, and whether their physical moment is the same in every TR;
* the net physical moment one TR leaves, from the excitation centre to the
  next excitation, and its spread across TRs (the quantity gradient spoiling
  needs constant);
* for spin echoes, the crusher moment on each side of the refocusing centre;
* the RF-spoiling increment recovered from the excitation phases, per slice,
  and whether the ADC phase follows the RF phase.

Usage: python spoiler_audit.py [out.json]
"""

from __future__ import annotations

import json
import sys

import numpy as np

from pypulseqpp import sequences

AXES = ("gx", "gy", "gz")

SPECS = [
    # name, kwargs, family
    ("gre_radial2D", dict(n_slices=2, n_dummy=3), "gre"),
    ("gre_spiral2D", dict(n_slices=2, n_dummy=3), "gre"),
    ("gre_propeller2D", dict(n_slices=2, n_dummy=3), "gre"),
    ("gre_stack_of_stars3D", dict(n_z=8, n_dummy=3), "gre"),
    ("gre_stack_of_spirals3D", dict(n_z=8, n_dummy=3), "gre"),
    ("gre_stack_of_blades3D", dict(n_z=8, n_dummy=3), "gre"),
    ("mprage_stack_of_stars3D", dict(n_z=8), "gre"),
    ("mprage_stack_of_spirals3D", dict(n_z=8), "gre"),
    ("se_radial2D", dict(n_slices=2, n_dummy=1), "se"),
    ("se_spiral2D", dict(n_slices=2, n_dummy=1), "se"),
    ("se_propeller2D", dict(n_slices=2, n_dummy=1), "se"),
    ("se_epi_propeller2D", dict(n_slices=2, n_dummy=1), "se"),
    ("se_stack_of_stars3D", dict(n_z=8, n_dummy=1), "se"),
    ("se_stack_of_spirals3D", dict(n_z=8, n_dummy=1), "se"),
    ("se_stack_of_blades3D", dict(n_z=8, n_dummy=1), "se"),
    ("zte3D", dict(n=64, n_dummy=1), "zte"),
    # Cartesian references for the spoiler moment.
    ("gre2D", dict(n_slices=2, n_dummy=3), "gre"),
    ("gre3D", dict(n_z=8, n_dummy=3), "gre"),
]


def quat_to_matrix(q):
    w, x, y, z = (float(v) for v in q)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def grad_points(g):
    """Corner times (s) and amplitudes (Hz/m) of a gradient event."""
    if g is None:
        return np.zeros(0), np.zeros(0)
    if g.type == "trap":
        d, r, f, fa = (
            float(g.delay),
            float(g.rise_time),
            float(g.flat_time),
            float(g.fall_time),
        )
        a = float(g.amplitude)
        return (
            np.array([d, d + r, d + r + f, d + r + f + fa]),
            np.array([0.0, a, a, 0.0]),
        )
    tt = np.asarray(g.tt, dtype=float) + float(g.delay)
    wf = np.asarray(g.waveform, dtype=float)
    first = float(getattr(g, "first", wf[0]))
    last = float(getattr(g, "last", wf[-1]))
    shape_dur = float(getattr(g, "shape_dur", tt[-1] - float(g.delay)))
    t = np.concatenate([[float(g.delay)], tt, [float(g.delay) + shape_dur]])
    a = np.concatenate([[first], wf, [last]])
    keep = np.concatenate([[True], np.diff(t) > 0])
    return t[keep], a[keep]


def area_between(g, t0, t1):
    t, a = grad_points(g)
    if t.size == 0:
        return 0.0
    t0, t1 = max(t0, float(t[0])), min(t1, float(t[-1]))
    if t1 <= t0:
        return 0.0
    grid = np.unique(np.clip(np.concatenate([t, [t0, t1]]), t0, t1))
    vals = np.interp(grid, t, a, left=0.0, right=0.0)
    return float(np.trapezoid(vals, grid))


def block_area(g):
    if g is None:
        return 0.0
    t, _ = grad_points(g)
    return area_between(g, -1.0, float(t[-1]) + 1.0)


def decode(seq):
    rows = []
    n = len(seq.block_events)
    for i in range(1, n + 1):
        b = seq.get_block(i)
        rot = getattr(b, "rotation", None)
        R = np.eye(3) if rot is None else quat_to_matrix(rot.quaternion)
        logical = np.array([block_area(getattr(b, ax)) for ax in AXES])
        rows.append(
            dict(
                index=i,
                block=b,
                rf_use=None if b.rf is None else str(b.rf.use),
                adc=b.adc is not None,
                rotated=rot is not None and not np.allclose(R, np.eye(3)),
                has_rot_event=rot is not None,
                R=R,
                logical=logical,
                physical=R @ logical,
                inplane=any(
                    getattr(b, ax) is not None
                    and np.abs(block_area(getattr(b, ax))) >= 0
                    and _nonzero(getattr(b, ax))
                    for ax in ("gx", "gy")
                ),
            )
        )
    return rows


def _nonzero(g):
    t, a = grad_points(g)
    return bool(np.any(np.abs(a) > 1e-6))


def split_trs(rows):
    starts = [k for k, r in enumerate(rows) if r["rf_use"] == "excitation"]
    trs = []
    for j, s in enumerate(starts):
        e = starts[j + 1] if j + 1 < len(starts) else len(rows)
        trs.append(rows[s:e])
    return trs


def tail_blocks(tr):
    """Gradient blocks after the last block with in-plane gradient or ADC."""
    last = max((k for k, r in enumerate(tr) if r["inplane"] or r["adc"]), default=None)
    if last is None:
        return []
    return [
        r
        for r in tr[last + 1 :]
        if np.any(np.abs(r["logical"]) > 1e-3) and r["rf_use"] is None
    ]


def net_moment(tr, family):
    """Physical moment from the excitation centre to the end of the TR (1/m).

    For a spin echo, measured from the last refocusing centre instead.
    """
    exc = tr[0]
    ref_idx = [k for k, r in enumerate(tr) if r["rf_use"] == "refocusing"]
    if family == "se" and ref_idx:
        k0 = ref_idx[-1]
        anchor = tr[k0]
    else:
        k0 = 0
        anchor = exc
    b = anchor["block"]
    tc = float(b.rf.delay) + float(b.rf.center)
    first = np.array(
        [
            area_between(getattr(b, ax), tc, 1.0) if getattr(b, ax) is not None else 0.0
            for ax in AXES
        ]
    )
    total = anchor["R"] @ first
    for r in tr[k0 + 1 :]:
        total = total + r["physical"]
    return total


def _split(r, tc):
    b = r["block"]
    before = np.array(
        [
            area_between(getattr(b, ax), -1.0, tc)
            if getattr(b, ax) is not None
            else 0.0
            for ax in AXES
        ]
    )
    after = np.array(
        [
            area_between(getattr(b, ax), tc, 1.0) if getattr(b, ax) is not None else 0.0
            for ax in AXES
        ]
    )
    return r["R"] @ before, r["R"] @ after


def crushers(tr):
    """Crusher moment on each side of every refocusing centre (1/m).

    A refocusing block with its own gradient holds its crushers; a bare
    refocusing block takes them from the gradient-only blocks next to it.
    """
    out = []
    for k, r in enumerate(tr):
        if r["rf_use"] != "refocusing":
            continue
        b = r["block"]
        tc = float(b.rf.delay) + float(b.rf.center)
        before, after = _split(r, tc)
        rotated = r["rotated"]
        if not np.any(np.abs(r["logical"]) > 1e-3):

            def side(blocks):
                total = np.zeros(3)
                for q in blocks:
                    if (
                        q["rf_use"]
                        or q["adc"]
                        or q["inplane"]
                        or not np.any(np.abs(q["logical"]) > 1e-3)
                    ):
                        break
                    total = total + q["physical"]
                    nonlocal rotated
                    rotated = rotated or q["rotated"]
                return total

            before = side(tr[k - 1 :: -1] if k else [])
            after = side(tr[k + 1 :])
        out.append(dict(before=before.tolist(), after=after.tolist(), rotated=rotated))
    return out


def moment_at_adc(tr):
    """Physical moment at the start of the first ADC, signed through refocusing."""
    exc = tr[0]
    b = exc["block"]
    _, m = _split(exc, float(b.rf.delay) + float(b.rf.center))
    for r in tr[1:]:
        if r["adc"]:
            t0 = float(r["block"].adc.delay)
            pre, _ = _split(r, t0)
            return m + pre
        if r["rf_use"] == "refocusing":
            bb = r["block"]
            pre, post = _split(r, float(bb.rf.delay) + float(bb.rf.center))
            m = -(m + pre) + post
        else:
            m = m + r["physical"]
    return None


def rf_spoiling(trs):
    """Quadratic increment per slice (deg) and ADC-follows-RF check."""
    per_slice = {}
    adc_ok = True
    for tr in trs:
        rf = tr[0]["block"].rf
        phi = float(rf.phase_offset) + 2 * np.pi * float(rf.freq_offset) * float(
            rf.center
        )
        key = round(float(rf.freq_offset), 3)
        per_slice.setdefault(key, []).append(phi)
        adcs = [r["block"].adc for r in tr if r["adc"]]
        for adc in adcs:
            d = np.angle(np.exp(1j * (float(adc.phase_offset) - phi)))
            if abs(d) > 1e-6:
                adc_ok = False
    incs = []
    for phases in per_slice.values():
        p = np.unwrap(np.asarray(phases))
        if p.size >= 3:
            dd = np.diff(p, 2)
            incs.extend(np.rad2deg(np.angle(np.exp(1j * dd))).tolist())
    incs = np.asarray(incs)
    if incs.size == 0:
        return dict(
            increment_deg=None, constant=None, adc_follows=adc_ok, slices=len(per_slice)
        )
    inc = float(np.mod(incs[0], 360))
    const = bool(np.allclose(np.mod(incs - incs[0] + 180, 360) - 180, 0, atol=1e-3))
    if np.allclose(incs, 0, atol=1e-6):
        inc = 0.0
    return dict(
        increment_deg=inc, constant=const, adc_follows=adc_ok, slices=len(per_slice)
    )


def audit(name, kwargs, family):
    fn = getattr(sequences, name + "_sequence")
    fn = getattr(fn, "main", fn)
    seq = fn(**kwargs)
    rows = decode(seq)
    trs = split_trs(rows)
    # Skip a leading prep (e.g. MPRAGE inversion lives in its own blocks before
    # an excitation, which split_trs leaves in the previous TR's tail).
    tails = [tail_blocks(tr) for tr in trs]
    sig = []
    for t in tails:
        sig.append(
            tuple(
                (
                    tuple(np.round(r["logical"], 3)),
                    r["has_rot_event"],
                    tuple(np.round(r["physical"], 3)),
                )
                for r in t
            )
        )
    uniq = {}
    for s in sig:
        uniq[s] = uniq.get(s, 0) + 1
    common = max(uniq, key=uniq.get) if uniq else ()
    nets = np.array([net_moment(tr, family) for tr in trs])
    # Exclude the last TR of each train/volume whose tail is followed by a prep.
    spread = (nets.max(axis=0) - nets.min(axis=0)).tolist() if len(nets) else None
    at_adc = [moment_at_adc(tr) for tr in trs]
    at_adc = np.array([m for m in at_adc if m is not None])
    if at_adc.size:
        z_echo = float(at_adc[np.argmin(np.abs(at_adc[:, 2])), 2])
    else:
        z_echo = None
    res = dict(
        name=name,
        family=family,
        n_tr=len(trs),
        spoiler_variants=len(uniq),
        spoiler_common=[
            dict(logical=list(l), rotation_event=r, physical=list(p))
            for l, r, p in common
        ],
        spoiler_variant_counts=sorted(uniq.values(), reverse=True),
        spoiler_any_rotated=any(r["rotated"] for t in tails for r in t),
        net_moment_median=np.median(nets, axis=0).tolist(),
        net_moment_spread=spread,
        rf=rf_spoiling(trs),
        crushers=crushers(trs[min(len(trs) - 1, 2)]) if family == "se" else [],
        z_moment_at_adc_min=z_echo,
    )
    # A few deviant spoiler signatures for the report.
    res["spoiler_examples"] = [
        dict(
            count=c,
            blocks=[
                dict(logical=list(l), rotation_event=r, physical=list(p))
                for l, r, p in s
            ],
        )
        for s, c in sorted(uniq.items(), key=lambda kv: -kv[1])[:4]
    ]
    return res


def main(argv):
    out = argv[1] if len(argv) > 1 else "spoiler_audit.json"
    results = []
    for name, kwargs, family in SPECS:
        try:
            r = audit(name, kwargs, family)
        except Exception as exc:  # report and continue
            r = dict(name=name, family=family, error=f"{type(exc).__name__}: {exc}")
        results.append(r)
        print(
            json.dumps(
                {
                    k: r.get(k)
                    for k in (
                        "name",
                        "n_tr",
                        "spoiler_variants",
                        "spoiler_variant_counts",
                        "spoiler_common",
                        "spoiler_any_rotated",
                        "net_moment_median",
                        "net_moment_spread",
                        "rf",
                        "crushers",
                        "z_moment_at_adc_min",
                        "error",
                    )
                },
                default=float,
            )
        )
    with open(out, "w") as fh:
        json.dump(results, fh, indent=1, default=float)


if __name__ == "__main__":
    main(sys.argv)
