"""Acceleration network alpha_theta(y, v, u) for the state equation (2) of TASK.md.

    y' = v,  v' = alpha_theta(y, v, u)

A plain fully connected network in float64. Inputs are divided by fixed scales
(std of y, v, u on the training record, v from central differences) and the
output is multiplied by a fixed acceleration scale (std of the second
difference of y on the training record). The scales are buffers, not weights.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src.integrator import DTYPE
from src.metrics import fd_velocity_central

ACTIVATIONS = {"tanh": nn.Tanh, "silu": nn.SiLU}


@dataclass
class Scales:
    y: float
    v: float
    u: float
    acc: float

    @classmethod
    def from_train(cls, y: np.ndarray, u: np.ndarray, dt: float) -> "Scales":
        v = fd_velocity_central(y, dt)
        acc = (y[2:] - 2.0 * y[1:-1] + y[:-2]) / dt**2
        return cls(y=float(y.std()), v=float(v.std()), u=float(u.std()), acc=float(acc.std()))


class AccelMLP(nn.Module):
    def __init__(self, scales: Scales, hidden: tuple[int, ...] = (32, 32), activation: str = "tanh"):
        super().__init__()
        act = ACTIVATIONS[activation]
        layers: list[nn.Module] = []
        width_in = 3
        for width in hidden:
            layers += [nn.Linear(width_in, width), act()]
            width_in = width
        layers.append(nn.Linear(width_in, 1))
        self.net = nn.Sequential(*layers).to(DTYPE)

        self.register_buffer("in_scale", torch.tensor([scales.y, scales.v, scales.u], dtype=DTYPE))
        self.register_buffer("out_scale", torch.tensor(scales.acc, dtype=DTYPE))

    def forward(self, y: torch.Tensor, v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        x = torch.stack([y, v, u], dim=-1) / self.in_scale
        return self.net(x).squeeze(-1) * self.out_scale
