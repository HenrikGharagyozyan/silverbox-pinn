"""Sanity check of the Silverbox excerpt: shapes, test RMS, and basic plots."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import FS, load_silverbox  # noqa: E402

FIG_DIR = ROOT / "figures"


def main() -> None:
    d = load_silverbox()
    FIG_DIR.mkdir(exist_ok=True)

    print(f"dt = {d.dt:.6e} s, fs = {FS} Hz")
    print(f"u: {d.u.shape}, y: {d.y.shape}, t: {d.t.shape}")
    print(f"u_train: {d.u_train.shape}, y_train: {d.y_train.shape}")
    print(f"u_test:  {d.u_test.shape}, y_test:  {d.y_test.shape}")
    print(f"removed means: u_mean = {d.u_mean:.6e} V, y_mean = {d.y_mean:.6e} V")
    print(f"centered means: u = {d.u.mean():.1e}, y = {d.y.mean():.1e}")
    print(f"train duration = {d.n_train * d.dt:.3f} s, test duration = {d.n_test * d.dt:.3f} s")
    print(f"max |y| = {np.abs(d.y).max():.4f} V, max |u| = {np.abs(d.u).max():.4f} V")
    print(f"RMS y_train = {np.sqrt(np.mean(d.y_train**2)):.5f} V")
    print(f"RMS y_test  = {np.sqrt(np.mean(d.y_test**2)):.5f} V  (zero-prediction baseline)")

    t_split = d.n_train * d.dt

    # Time series of u and y with the train/test boundary marked.
    for name, sig in (("u", d.u), ("y", d.y)):
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(d.t, sig, lw=0.5)
        ax.axvline(t_split, color="k", ls="--", lw=1, label="train | test")
        ax.set_xlabel("t [s]")
        ax.set_ylabel(f"{name} [V]")
        ax.set_title(f"Centered {name}(t) on the excerpt [42650:46746]")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"data_{name}.png", dpi=150)
        plt.close(fig)

    # One-sided amplitude spectrum of y (whole excerpt, Hann window).
    n = d.y.size
    win = np.hanning(n)
    Y = np.fft.rfft(d.y * win)
    f = np.fft.rfftfreq(n, d.dt)
    amp = 2.0 * np.abs(Y) / win.sum()
    f_peak = f[np.argmax(amp)]
    print(f"dominant frequency of y = {f_peak:.2f} Hz")

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.semilogy(f, amp, lw=0.7)
    ax.axvline(f_peak, color="r", ls=":", lw=1, label=f"peak {f_peak:.1f} Hz")
    ax.set_xlabel("f [Hz]")
    ax.set_ylabel("|Y(f)| [V]")
    ax.set_xlim(0, FS / 2)
    ax.set_title("Amplitude spectrum of centered y (Hann window)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "data_y_spectrum.png", dpi=150)
    plt.close(fig)

    print(f"figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
