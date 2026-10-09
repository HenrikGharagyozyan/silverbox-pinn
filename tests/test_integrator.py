"""Convergence test of the RK4 rollout on a forced, damped linear oscillator.

    y'' + a y' + b y = g u(t),  u(t) = u0 + u1 t

The input is linear in time, so the midpoint rule u(t + dt/2) = (u[n] + u[n+1]) / 2
is exact and the only error left is the RK4 truncation error, which must scale
as dt^4. Run with `pytest tests/` or `python tests/test_integrator.py`.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.integrator import DTYPE, rollout  # noqa: E402

A, B, G = 2.0, 40.0, 3.0  # underdamped: omega_d = sqrt(B - A^2 / 4)
U0, U1 = 0.5, -0.2
Y0, V0 = 1.0, 0.0
T_END = 2.0


def exact(t: np.ndarray) -> np.ndarray:
    # Particular solution y_p = P + Q t.
    q = G * U1 / B
    p = (G * U0 - A * q) / B
    # Homogeneous part exp(-s t) (C1 cos(w t) + C2 sin(w t)) fitted to (Y0, V0).
    s = A / 2.0
    w = np.sqrt(B - s**2)
    c1 = Y0 - p
    c2 = (V0 - q + s * c1) / w
    return p + q * t + np.exp(-s * t) * (c1 * np.cos(w * t) + c2 * np.sin(w * t))


def f(y, v, u):
    return -A * v - B * y + G * u


def max_error(n_steps: int) -> float:
    dt = T_END / n_steps
    t = np.arange(n_steps + 1) * dt
    u = torch.tensor(U0 + U1 * t, dtype=DTYPE)
    y = rollout(f, torch.tensor(Y0, dtype=DTYPE), torch.tensor(V0, dtype=DTYPE), u, dt)
    return float(np.max(np.abs(y.numpy() - exact(t))))


def test_rk4_fourth_order():
    steps = [50, 100, 200, 400, 800]
    errs = np.array([max_error(n) for n in steps])
    orders = np.log2(errs[:-1] / errs[1:])
    print("errors:", errs)
    print("observed orders:", orders)
    assert np.all(np.abs(orders - 4.0) < 0.2), orders


def test_batched_matches_unbatched():
    n, dt = 100, 0.01
    u = torch.randn(3, n, dtype=DTYPE)
    y0 = torch.tensor([0.1, -0.2, 0.3], dtype=DTYPE)
    v0 = torch.tensor([1.0, 0.0, -1.0], dtype=DTYPE)
    yb = rollout(f, y0, v0, u, dt)
    for i in range(3):
        yi = rollout(f, y0[i], v0[i], u[i], dt)
        assert torch.allclose(yb[i], yi)


if __name__ == "__main__":
    test_rk4_fourth_order()
    test_batched_matches_unbatched()
    print("ok")
