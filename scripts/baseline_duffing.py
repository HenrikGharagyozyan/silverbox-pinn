"""Physical baseline: Duffing equation (1) with four trainable coefficients.

    y'' + a y' + b y + c y^3 = g u(t)

1. Initial a, b, c, g by linear least squares on finite-difference derivatives
   of the training record.
2. Refinement by gradient descent through RK4 rollouts on the training record:
   first short windows started from measured states, then the full free run
   of the training record.
3. Free-run RMSE on train and test as defined in TASK.md.
"""

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_silverbox  # noqa: E402
from src.integrator import DTYPE, rollout  # noqa: E402
from src.metrics import fd_velocity_central, fd_velocity_start, free_run_scores  # noqa: E402

NAMES = ("a", "b", "c", "g")
SEED = 0


def least_squares_init(y: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """Fit y'' = -a y' - b y - c y^3 + g u on central differences."""
    ydd = (y[2:] - 2.0 * y[1:-1] + y[:-2]) / dt**2
    yd = fd_velocity_central(y, dt)
    ym, um = y[1:-1], u[1:-1]
    X = np.stack([-yd, -ym, -(ym**3), um], axis=1)
    theta, *_ = np.linalg.lstsq(X, ydd, rcond=None)
    return theta


class Duffing(torch.nn.Module):
    """Acceleration f(y, v, u) = -a v - b y - c y^3 + g u.

    The coefficients differ by orders of magnitude (b ~ 1e5, a ~ 1e1), so each
    one is stored as a dimensionless factor times a fixed scale.
    """

    def __init__(self, theta0: np.ndarray):
        super().__init__()
        scale = np.where(np.abs(theta0) > 0, np.abs(theta0), 1.0)
        self.register_buffer("scale", torch.tensor(scale, dtype=DTYPE))
        self.w = torch.nn.Parameter(torch.tensor(theta0 / scale, dtype=DTYPE))

    def coeffs(self) -> torch.Tensor:
        return self.w * self.scale

    def forward(self, y, v, u):
        return self.accel()(y, v, u)

    def accel(self):
        """Closure with the coefficients computed once, for use inside a rollout."""
        a, b, c, g = self.coeffs().unbind()
        return lambda y, v, u: -a * v - b * y - c * y**3 + g * u


def train_windows(model, y, u, dt, window, batch, steps, lr, rng):
    """Rollouts of `window` samples from random measured states in the train record."""
    yt = torch.tensor(y, dtype=DTYPE)
    ut = torch.tensor(u, dtype=DTYPE)
    vt = torch.tensor(fd_velocity_central(y, dt), dtype=DTYPE)  # v at indices 1..N-2
    y_scale = float(np.sqrt(np.mean(y**2)))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    idx = torch.arange(window)
    for step in range(steps):
        starts = torch.tensor(rng.integers(1, len(y) - window, size=batch))
        rows = starts[:, None] + idx[None, :]
        y_hat = rollout(model.accel(), yt[starts], vt[starts - 1], ut[rows], dt)
        loss = torch.mean(((y_hat - yt[rows]) / y_scale) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            print(f"  [win {window}] step {step:4d}  loss {loss.item():.4e}")


def train_full(model, y, u, dt, steps, lr):
    """Free run of the whole training record from the first sample (same start as the metric)."""
    yt = torch.tensor(y, dtype=DTYPE)
    ut = torch.tensor(u, dtype=DTYPE)
    y0 = torch.tensor(y[0], dtype=DTYPE)
    v0 = torch.tensor(fd_velocity_start(y, dt), dtype=DTYPE)
    y_scale = float(np.sqrt(np.mean(y**2)))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = (np.inf, model.w.detach().clone())
    for step in range(steps):
        y_hat = rollout(model.accel(), y0, v0, ut, dt)
        loss = torch.mean(((y_hat - yt) / y_scale) ** 2)
        if loss.item() < best[0]:
            best = (loss.item(), model.w.detach().clone())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 10 == 0 or step == steps - 1:
            print(f"  [full] step {step:4d}  loss {loss.item():.4e}")
    with torch.no_grad():
        model.w.copy_(best[1])


def report(tag, model, data):
    s = free_run_scores(model.accel(), data)
    coeffs = dict(zip(NAMES, model.coeffs().detach().numpy().tolist()))
    print(f"{tag}: " + ", ".join(f"{k}={v:.6g}" for k, v in coeffs.items()))
    print(f"{tag}: free-run RMSE train = {s['rmse_train']:.5f} V, test = {s['rmse_test']:.5f} V")
    return s, coeffs


def main() -> None:
    torch.set_num_threads(1)  # tiny tensors: threading only adds overhead
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    data = load_silverbox()
    dt = data.dt
    y_tr, u_tr = data.y_train, data.u_train

    t0 = time.time()
    theta0 = least_squares_init(y_tr, u_tr, dt)
    model = Duffing(theta0)
    s_ls, c_ls = report("LS init", model, data)

    train_windows(model, y_tr, u_tr, dt, window=64, batch=64, steps=400, lr=1e-2, rng=rng)
    train_windows(model, y_tr, u_tr, dt, window=512, batch=16, steps=100, lr=3e-3, rng=rng)
    report("windows", model, data)

    train_full(model, y_tr, u_tr, dt, steps=40, lr=1e-3)
    s, coeffs = report("final", model, data)
    print(f"total time {time.time() - t0:.1f} s")
    print(f"zero-prediction baseline on test: {np.sqrt(np.mean(data.y_test**2)):.5f} V")

    out = {
        "ls_init": {"coeffs": c_ls, "rmse_train": s_ls["rmse_train"], "rmse_test": s_ls["rmse_test"]},
        "final": {"coeffs": coeffs, "rmse_train": s["rmse_train"], "rmse_test": s["rmse_test"]},
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "baseline_duffing.json").write_text(json.dumps(out, indent=2))

    n = data.n_train
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(data.t[n:], data.y_test, lw=0.8, label="measured")
    ax.plot(data.t[n:], s["y_hat"][n:], lw=0.8, ls="--", label="Duffing free run")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("y [V]")
    ax.set_title(f"Duffing baseline on test segment, RMSE = {s['rmse_test'] * 1e3:.2f} mV")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "baseline_duffing_test.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
