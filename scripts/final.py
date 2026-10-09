"""Final evaluation of the kept model (results/best.pt), no training.

One free run of the whole excerpt by the rules of TASK.md, both RMSE values,
the test plot (measured vs predicted, error below), and the learned force
alpha(y, 0, 0) against the Duffing force -b y - c y^3 of the baseline.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_silverbox  # noqa: E402
from src.evaluate import evaluate, load_model, plot_test  # noqa: E402
from src.integrator import DTYPE  # noqa: E402

TITLE = "3x64 SiLU"


def plot_force(model, coeffs: dict, y_train_max: float, y_test_max: float, path: Path) -> None:
    ys = np.linspace(-0.25, 0.25, 501)
    z = torch.zeros(len(ys), dtype=DTYPE)
    with torch.no_grad():
        a_nn = model(torch.tensor(ys, dtype=DTYPE), z, z).numpy()
    a_duff = -coeffs["b"] * ys - coeffs["c"] * ys**3

    fig, axes = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True, height_ratios=[3, 2])
    ax = axes[0]
    ax.plot(ys, a_duff, lw=1.5, label=r"Duffing $-b\,y - c\,y^3$")
    ax.plot(ys, a_nn, lw=1.5, ls="--", label=r"network $\alpha_\theta(y, 0, 0)$")
    ax.set_ylabel(r"acceleration [V/s$^2$]")
    ax.set_title(f"{TITLE}: learned restoring force at v = 0, u = 0")
    ax2 = axes[1]
    ax2.plot(ys, a_nn - a_duff, lw=1.2, color="C3")
    ax2.axhline(0.0, color="k", lw=0.5)
    ax2.set_ylabel(r"network $-$ Duffing")
    ax2.set_xlabel("y [V]")
    for a in axes:
        for s in (-1, 1):
            a.axvline(s * y_train_max, color="k", ls=":", lw=1)
            a.axvline(s * y_test_max, color="0.6", ls="-.", lw=1)
    axes[0].plot([], [], color="k", ls=":", label=f"train range |y| = {y_train_max:.3f} V")
    axes[0].plot([], [], color="0.6", ls="-.", label=f"test max |y| = {y_test_max:.3f} V")
    axes[0].legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    data = load_silverbox()
    model = load_model(ROOT / "results" / "best.pt")
    s = evaluate(model, data)
    n_params = sum(p.numel() for p in model.parameters())
    y_train_max = float(np.abs(data.y_train).max())
    y_test_max = float(np.abs(data.y_test).max())
    baseline = json.loads((ROOT / "results" / "baseline_duffing.json").read_text())["final"]

    print(f"model: {TITLE}, {n_params} weights")
    print(f"free-run RMSE train = {s['rmse_train'] * 1e3:.3f} mV")
    print(f"free-run RMSE test  = {s['rmse_test'] * 1e3:.3f} mV")
    print(f"zero prediction on test = {np.sqrt(np.mean(data.y_test**2)) * 1e3:.3f} mV")
    print(
        f"Duffing baseline: train = {baseline['rmse_train'] * 1e3:.3f} mV, "
        f"test = {baseline['rmse_test'] * 1e3:.3f} mV"
    )
    print(f"max |y|: train = {y_train_max:.4f} V, test = {y_test_max:.4f} V")

    plot_test(data, s["y_hat"], s["rmse_test"], TITLE, ROOT / "figures" / "final_test.png")
    plot_force(model, baseline["coeffs"], y_train_max, y_test_max, ROOT / "figures" / "final_force.png")

    out = {
        "model": TITLE,
        "n_params": n_params,
        "rmse_train": s["rmse_train"],
        "rmse_test": s["rmse_test"],
        "rmse_test_zero": float(np.sqrt(np.mean(data.y_test**2))),
        "y_train_max": y_train_max,
        "y_test_max": y_test_max,
    }
    (ROOT / "results" / "final.json").write_text(json.dumps(out, indent=2))
    print("saved figures/final_test.png, figures/final_force.png, results/final.json")


if __name__ == "__main__":
    main()
