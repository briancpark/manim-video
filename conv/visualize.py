#!/usr/bin/env python3
"""Visualize the conv2d test bed:
  1) perf.png  - GFLOP/s and speedup from ./conv2d --sweep (sweep.csv)
  2) maps.png  - real image filtered by the *actual* NEON kernel (./conv_filter)

Run via `make viz`, or directly after `./conv2d --sweep > sweep.csv`.
"""
import csv
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = __file__.rsplit("/", 1)[0]
FILTER_BIN = f"{HERE}/conv_filter"


# ---------------------------------------------------------------- perf chart
def plot_perf(csv_path="sweep.csv", out="perf.png"):
    rows = list(csv.DictReader(open(csv_path)))
    ks = sorted({int(r["K"]) for r in rows})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for k in ks:
        sub = [r for r in rows if int(r["K"]) == k]
        ch = [int(r["Cin"]) for r in sub]
        ax1.plot(ch, [float(r["gflops_neon"]) for r in sub], "-o", label=f"neon {k}x{k}")
        ax1.plot(ch, [float(r["gflops_naive"]) for r in sub], "--x", color="gray",
                 alpha=0.5, label="naive" if k == ks[0] else None)
        ax2.plot(ch, [float(r["speedup"]) for r in sub], "-o", label=f"{k}x{k}")

    ax1.set(title="Throughput (M1 Pro, single core)", xlabel="channels (Cin=Cout)",
            ylabel="GFLOP/s", xscale="log", xticks=ch)
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.grid(alpha=0.3); ax1.legend(fontsize=8)

    ax2.set(title="NEON speedup vs scalar", xlabel="channels (Cin=Cout)",
            ylabel="x faster", xscale="log", xticks=ch)
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.axhline(1, color="k", lw=0.8); ax2.grid(alpha=0.3); ax2.legend(title="kernel")

    fig.tight_layout(); fig.savefig(out, dpi=110)
    print(f"wrote {out}")


# ----------------------------------------------------------- feature maps
def run_filter(img, kernel):
    """Apply one kernel to a single-channel image using the real NEON binary."""
    H, W = img.shape
    KH, KW = kernel.shape
    hdr = np.array([1, 1, H, W, KH, KW], dtype=np.int32)
    with open("/tmp/conv_in.bin", "wb") as f:
        f.write(hdr.tobytes())
        f.write(img.astype(np.float32).tobytes())
        f.write(kernel.astype(np.float32).tobytes())
    subprocess.run([FILTER_BIN, "/tmp/conv_in.bin", "/tmp/conv_out.bin"], check=True)
    out = np.fromfile("/tmp/conv_out.bin", dtype=np.float32)
    return out.reshape(H - KH + 1, W - KW + 1)


def make_test_image(n=256):
    """Synthetic grayscale: circles, a bar, and a gradient — lots of edges."""
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    img = 0.3 + 0.3 * (x / n)                      # gradient
    img += 0.4 * (((x - 90) ** 2 + (y - 90) ** 2) < 45 ** 2)   # disc
    img += 0.5 * ((np.abs(x - 175) < 30) & (np.abs(y - 170) < 55))  # bar
    ring = ((x - 180) ** 2 + (y - 80) ** 2)
    img += 0.4 * ((ring < 50 ** 2) & (ring > 38 ** 2))         # ring
    return np.clip(img, 0, 1)


def plot_maps(out="maps.png"):
    img = make_test_image()
    kernels = {
        "Sobel X": np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], np.float32),
        "Sobel Y": np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32),
        "Gaussian blur": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 16,
        "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32),
    }
    fig, axes = plt.subplots(1, len(kernels) + 1, figsize=(3 * (len(kernels) + 1), 3.2))
    axes[0].imshow(img, cmap="gray"); axes[0].set_title("input"); axes[0].axis("off")
    for ax, (name, k) in zip(axes[1:], kernels.items()):
        res = run_filter(img, k)
        cmap = "gray" if name in ("Gaussian blur", "Sharpen") else "RdBu"
        ax.imshow(res, cmap=cmap); ax.set_title(name); ax.axis("off")
    fig.suptitle("conv2d_neon applied to a real image", y=1.02)
    fig.tight_layout(); fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    plot_perf()
    plot_maps()
