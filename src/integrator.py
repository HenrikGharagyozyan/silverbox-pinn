"""Classical RK4 integration of the second-order system y' = v, v' = f(y, v, u).

The input u is only known at the sample times. Inside a step the RK4 stages
need u at t_n, t_n + dt/2 and t_n + dt; the midpoint value is taken as the
mean of u[n] and u[n+1] (linear interpolation between samples).

All tensors are float64. Shapes are batched: y, v, u are (B,) per time step
and the input sequence is (B, N). Unbatched (N,) inputs are also accepted.
"""

from typing import Callable

import torch

DTYPE = torch.float64

# f(y, v, u) -> acceleration, all tensors of the same shape.
Accel = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def rk4_step(
    f: Accel,
    y: torch.Tensor,
    v: torch.Tensor,
    u0: torch.Tensor,
    u1: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance (y, v) by one step dt, with u0 = u(t_n) and u1 = u(t_n + dt)."""
    um = 0.5 * (u0 + u1)

    k1y = v
    k1v = f(y, v, u0)

    k2y = v + 0.5 * dt * k1v
    k2v = f(y + 0.5 * dt * k1y, k2y, um)

    k3y = v + 0.5 * dt * k2v
    k3v = f(y + 0.5 * dt * k2y, k3y, um)

    k4y = v + dt * k3v
    k4v = f(y + dt * k3y, k4y, u1)

    y_next = y + dt / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
    v_next = v + dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    return y_next, v_next


def rollout(
    f: Accel,
    y0: torch.Tensor,
    v0: torch.Tensor,
    u_seq: torch.Tensor,
    dt: float,
    return_v: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Integrate from (y0, v0) over the input sequence u_seq.

    y0, v0: (B,) or scalar; u_seq: (B, N) or (N,).
    Returns y of shape (B, N) (or (N,) for unbatched input) with y[..., 0] = y0,
    and optionally v of the same shape.
    """
    unbatched = u_seq.dim() == 1
    u_seq = torch.as_tensor(u_seq, dtype=DTYPE)
    if unbatched:
        u_seq = u_seq.unsqueeze(0)
    batch, n = u_seq.shape
    y = torch.as_tensor(y0, dtype=DTYPE).reshape(-1).expand(batch)
    v = torch.as_tensor(v0, dtype=DTYPE).reshape(-1).expand(batch)

    ys, vs = [y], [v]
    for k in range(n - 1):
        y, v = rk4_step(f, y, v, u_seq[:, k], u_seq[:, k + 1], dt)
        ys.append(y)
        vs.append(v)

    y_traj = torch.stack(ys, dim=1)
    v_traj = torch.stack(vs, dim=1)
    if unbatched:
        y_traj, v_traj = y_traj[0], v_traj[0]
    return (y_traj, v_traj) if return_v else y_traj
