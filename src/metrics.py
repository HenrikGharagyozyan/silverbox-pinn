"""Free-run evaluation exactly as in the "The metric" section of TASK.md.

One rollout of the whole excerpt, started at the first sample with the measured
position and a finite-difference velocity, driven by the measured u only. The
state is carried across the train/test boundary; y is never read after t = 0
(besides y[0:3] for the initial velocity).
"""

import numpy as np
import torch

from src.data import SilverboxData
from src.integrator import DTYPE, Accel, rollout


def fd_velocity_start(y: np.ndarray, dt: float) -> float:
    """Second-order one-sided finite difference of y at the first sample."""
    return float((-3.0 * y[0] + 4.0 * y[1] - y[2]) / (2.0 * dt))


def fd_velocity_central(y: np.ndarray, dt: float) -> np.ndarray:
    """Second-order central difference of y at the interior samples 1..N-2."""
    return (y[2:] - y[:-2]) / (2.0 * dt)


@torch.no_grad()
def free_run(f: Accel, data: SilverboxData) -> np.ndarray:
    """Free-run prediction of y over the whole excerpt (train + test)."""
    y0 = torch.tensor(data.y[0], dtype=DTYPE)
    v0 = torch.tensor(fd_velocity_start(data.y, data.dt), dtype=DTYPE)
    u = torch.tensor(data.u, dtype=DTYPE)
    return rollout(f, y0, v0, u, data.dt).numpy()


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def free_run_scores(f: Accel, data: SilverboxData) -> dict:
    """Free-run RMSE on the 3072 train samples and on the 1024 test samples."""
    y_hat = free_run(f, data)
    n = data.n_train
    return {
        "rmse_train": rmse(y_hat[:n], data.y[:n]),
        "rmse_test": rmse(y_hat[n:], data.y[n:]),
        "y_hat": y_hat,
    }
