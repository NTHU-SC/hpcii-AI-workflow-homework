#!/usr/bin/env python3
"""Look at a heat output file (512x512 little-endian float64).

    python3 tools/show.py FIELD.out                  # summary + FIELD.png
    python3 tools/show.py FIELD.out -o plate.png     # choose the image name
    python3 tools/show.py A.out --diff B.out         # compare two outputs

The plot puts the bottom edge of the plate (row 0) at the bottom and the left
edge (column 0) on the left. With --diff it plots log10 |A - B| and reports
the largest difference and where it is.

Needs NumPy and matplotlib.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

N = 512
TOL = 1e-6


def read_field(path):
    a = np.fromfile(path, dtype="<f8")
    if a.size != N * N:
        sys.exit(f"{path}: expected {N * N} float64 values, found {a.size}")
    return a.reshape(N, N)


def summary(name, T):
    bad = int((~np.isfinite(T)).sum())
    print(f"{name}: {T.size} values, min {np.nanmin(T):.6f} K, max {np.nanmax(T):.6f} K, "
          f"mean {np.nanmean(T):.6f} K" + (f", {bad} NaN/Inf" if bad else ""))
    cols = (0, 64, 128, 192, 256, 320, 384, 448, 511)
    print("  row\\col " + "".join(f"{i:>9d}" for i in cols))
    for j in (511, 384, 256, 128, 0):
        print(f"  {j:>7d} " + "".join(f"{T[j, i]:9.2f}" for i in cols))


def plot(data, title, label, out, cmap, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=120)
    im = ax.imshow(data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("column i (0 = left edge)")
    ax.set_ylabel("row j (0 = bottom edge)")
    fig.colorbar(im, ax=ax, label=label)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("field", help="output file of heat (512x512 float64)")
    ap.add_argument("--diff", metavar="OTHER", help="compare with another output file")
    ap.add_argument("-o", "--out", help="PNG to write (default: next to FIELD, .png)")
    ap.add_argument("--range", nargs=2, type=float, metavar=("LO", "HI"),
                    help="colour range in K (default: min and max of the field)")
    a = ap.parse_args()

    T = read_field(a.field)
    if not a.diff:
        summary(a.field, T)
        out = a.out or os.path.splitext(a.field)[0] + ".png"
        lo, hi = a.range or (None, None)
        plot(T, os.path.basename(a.field), "temperature (K)", out, "inferno", lo, hi)
        print(f"image: {out}")
        return

    U = read_field(a.diff)
    d = np.abs(T - U)
    j, i = np.unravel_index(np.nanargmax(np.where(np.isnan(d), np.inf, d)), d.shape)
    print(f"max |A - B| = {d[j, i]:.3e} K at row {j}, col {i} (A = {T[j, i]:.9f}, B = {U[j, i]:.9f})")
    print(f"cells differing at all: {int((d != 0).sum())}; "
          f"cells beyond {TOL:g} K: {int((~(d <= TOL)).sum())}")
    out = a.out or os.path.splitext(a.field)[0] + ".diff.png"
    with np.errstate(divide="ignore"):
        logd = np.log10(d)
    shown = np.ma.masked_invalid(logd)
    vmax = max(float(shown.max()) if shown.count() else -15.0, -15.0)
    plot(shown, f"|{os.path.basename(a.field)} - {os.path.basename(a.diff)}|",
         "log10 |A - B|  (K)   (white = identical)", out, "viridis", -16.0, vmax)
    print(f"image: {out}")


if __name__ == "__main__":
    main()
