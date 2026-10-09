"""Free-run evaluation of a trained acceleration network.

One free run over all 4096 samples by the rules of TASK.md (see src/metrics.py):
start at the first sample with the measured y and a finite-difference velocity,
drive with the measured u only, keep the state across the train/test boundary.
Usage:
    python -m src.evaluate results/<name>/model.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from src.data import SilverboxData, load_silverbox
from src.metrics import free_run_scores
from src.model import AccelMLP, Scales

ROOT = Path(__file__).resolve().parent.parent


def evaluate(model: torch.nn.Module, data: SilverboxData) -> dict:
    """Free-run RMSE on train and test, plus the predicted trajectory y_hat."""
    model.eval()
    s = free_run_scores(model, data)
    model.train()
    return s


def load_model(path: str | Path) -> AccelMLP:
    ckpt = torch.load(path, weights_only=False)
    cfg = ckpt["config"]
    model = AccelMLP(Scales(**ckpt["scales"]), tuple(cfg["hidden"]), cfg["activation"])
    model.load_state_dict(ckpt["state_dict"])
    return model


def plot_test(data: SilverboxData, y_hat: np.ndarray, rmse_test: float, title: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = data.n_train
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, height_ratios=[3, 1])
    axes[0].plot(data.t[n:], data.y_test, lw=0.8, label="measured")
    axes[0].plot(data.t[n:], y_hat[n:], lw=0.8, ls="--", label="free run")
    axes[0].set_ylabel("y [V]")
    axes[0].set_title(f"{title}: test segment, RMSE = {rmse_test * 1e3:.3f} mV")
    axes[0].legend(loc="upper right")
    axes[1].plot(data.t[n:], (y_hat[n:] - data.y_test) * 1e3, lw=0.7, color="C3")
    axes[1].set_ylabel("error [mV]")
    axes[1].set_xlabel("t [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    args = parser.parse_args()
    ckpt_path = Path(args.checkpoint)
    data = load_silverbox()
    model = load_model(ckpt_path)
    s = evaluate(model, data)
    print(f"free-run RMSE train = {s['rmse_train'] * 1e3:.3f} mV, test = {s['rmse_test'] * 1e3:.3f} mV")
    name = ckpt_path.parent.name
    out = ROOT / "figures" / f"{name}_test.png"
    plot_test(data, s["y_hat"], s["rmse_test"], name, out)
    print(f"figure saved to {out}")


if __name__ == "__main__":
    main()
