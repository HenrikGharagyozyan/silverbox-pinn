"""Training of the acceleration network on the training record only.

Stages (each can be switched off in the config):
  1) pretrain: regression of alpha(y, v, u) on the second difference of y,
     with v from central differences. A rough start: the second difference is
     biased at ~9 samples per period.
  2) rollout:  Adam on RK4 rollouts of random windows from the train record,
     started at the measured y and the central-difference velocity; the window
     length K grows by a schedule; lr decays (cosine); gradient clipping.
  3) finetune: Adam (and optionally L-BFGS) on the full free run of the train
     record from the first sample, the same start as the metric.

After each enabled stage the free-run RMSE on train and test is computed with
src/metrics.py. Usage:
    python -m src.train                      # default config
    python -m src.train --config cfg.json    # overrides from a JSON file
"""

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from src.data import SilverboxData, load_silverbox
from src.evaluate import evaluate
from src.integrator import DTYPE, rollout
from src.metrics import fd_velocity_central, fd_velocity_start
from src.model import AccelMLP, Scales

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    name: str = "mlp_2x32_tanh"
    seed: int = 0
    threads: int = 1
    # network
    hidden: tuple[int, ...] = (32, 32)
    activation: str = "tanh"
    # stage 1: regression on the second difference
    pretrain: bool = True
    pretrain_steps: int = 3000
    pretrain_lr: float = 3e-3
    # stage 2: rollouts of random windows, schedule of (K, steps)
    rollout: bool = True
    rollout_schedule: list[tuple[int, int]] = field(
        default_factory=lambda: [(32, 400), (128, 300), (512, 150)]
    )
    rollout_batch: int = 256
    rollout_lr: float = 3e-3
    rollout_lr_final: float = 1e-4
    grad_clip: float = 1.0
    # stage 3: full free run of the training record
    finetune: bool = True
    finetune_steps: int = 40
    finetune_lr: float = 1e-4
    finetune_lbfgs_steps: int = 0

    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        raw = json.loads(Path(path).read_text())
        if "hidden" in raw:
            raw["hidden"] = tuple(raw["hidden"])
        if "rollout_schedule" in raw:
            raw["rollout_schedule"] = [tuple(x) for x in raw["rollout_schedule"]]
        return cls(**raw)


def _y_loss(y_hat: torch.Tensor, y: torch.Tensor, y_scale: float) -> torch.Tensor:
    return torch.mean(((y_hat - y) / y_scale) ** 2)


def stage_pretrain(model: AccelMLP, data: SilverboxData, cfg: Config) -> None:
    y, u, dt = data.y_train, data.u_train, data.dt
    acc = (y[2:] - 2.0 * y[1:-1] + y[:-2]) / dt**2
    yt = torch.tensor(y[1:-1], dtype=DTYPE)
    vt = torch.tensor(fd_velocity_central(y, dt), dtype=DTYPE)
    ut = torch.tensor(u[1:-1], dtype=DTYPE)
    at = torch.tensor(acc, dtype=DTYPE)
    scale = float(model.out_scale)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.pretrain_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.pretrain_steps, eta_min=cfg.pretrain_lr * 0.03)
    for step in range(cfg.pretrain_steps):
        loss = torch.mean(((model(yt, vt, ut) - at) / scale) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == cfg.pretrain_steps - 1:
            print(f"  [pretrain] step {step:5d}  loss {loss.item():.4e}")


def stage_rollout(model: AccelMLP, data: SilverboxData, cfg: Config, rng: np.random.Generator) -> None:
    y, u, dt = data.y_train, data.u_train, data.dt
    yt = torch.tensor(y, dtype=DTYPE)
    ut = torch.tensor(u, dtype=DTYPE)
    vt = torch.tensor(fd_velocity_central(y, dt), dtype=DTYPE)  # v at indices 1..N-2
    y_scale = float(y.std())

    total = sum(steps for _, steps in cfg.rollout_schedule)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.rollout_lr)

    def lr_at(i: int) -> float:  # cosine decay over the whole stage
        c = 0.5 * (1.0 + math.cos(math.pi * i / max(total - 1, 1)))
        return cfg.rollout_lr_final + (cfg.rollout_lr - cfg.rollout_lr_final) * c

    i = 0
    for window, steps in cfg.rollout_schedule:
        idx = torch.arange(window)
        for step in range(steps):
            for group in opt.param_groups:
                group["lr"] = lr_at(i)
            i += 1
            starts = torch.tensor(rng.integers(1, len(y) - window, size=cfg.rollout_batch))
            rows = starts[:, None] + idx[None, :]
            y_hat = rollout(model, yt[starts], vt[starts - 1], ut[rows], dt)
            loss = _y_loss(y_hat, yt[rows], y_scale)
            if not torch.isfinite(loss):
                print(f"  [K={window}] step {step}: non-finite loss, step skipped")
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            if step % 50 == 0 or step == steps - 1:
                print(f"  [K={window:4d}] step {step:4d}  lr {lr_at(i - 1):.2e}  loss {loss.item():.4e}")


def stage_finetune(model: AccelMLP, data: SilverboxData, cfg: Config) -> None:
    y, u, dt = data.y_train, data.u_train, data.dt
    yt = torch.tensor(y, dtype=DTYPE)
    ut = torch.tensor(u, dtype=DTYPE)
    y0 = torch.tensor(y[0], dtype=DTYPE)
    v0 = torch.tensor(fd_velocity_start(y, dt), dtype=DTYPE)
    y_scale = float(y.std())

    def closure_loss() -> torch.Tensor:
        return _y_loss(rollout(model, y0, v0, ut, dt), yt, y_scale)

    best_loss, best_state = math.inf, None

    def keep_best(loss: float) -> None:
        nonlocal best_loss, best_state
        if math.isfinite(loss) and loss < best_loss:
            best_loss = loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    opt = torch.optim.Adam(model.parameters(), lr=cfg.finetune_lr)
    for step in range(cfg.finetune_steps):
        loss = closure_loss()
        keep_best(loss.item())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if step % 10 == 0 or step == cfg.finetune_steps - 1:
            print(f"  [finetune adam] step {step:3d}  loss {loss.item():.4e}")

    if cfg.finetune_lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(), lr=1.0, max_iter=cfg.finetune_lbfgs_steps, line_search_fn="strong_wolfe"
        )

        def closure():
            lbfgs.zero_grad()
            loss = closure_loss()
            loss.backward()
            keep_best(loss.item())
            return loss

        lbfgs.step(closure)
        print(f"  [finetune lbfgs] best loss {best_loss:.4e}")

    with torch.no_grad():
        loss = closure_loss().item()
    keep_best(loss)
    if best_state is not None:
        model.load_state_dict(best_state)


def train(cfg: Config) -> dict:
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.threads)
    rng = np.random.default_rng(cfg.seed)
    data = load_silverbox()

    scales = Scales.from_train(data.y_train, data.u_train, data.dt)
    model = AccelMLP(scales, cfg.hidden, cfg.activation)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"config: {asdict(cfg)}")
    print(f"scales: {scales}, parameters: {n_params}")

    stages = [
        ("pretrain", cfg.pretrain, lambda: stage_pretrain(model, data, cfg)),
        ("rollout", cfg.rollout, lambda: stage_rollout(model, data, cfg, rng)),
        ("finetune", cfg.finetune, lambda: stage_finetune(model, data, cfg)),
    ]
    history = []
    for name, enabled, run in stages:
        if not enabled:
            continue
        t0 = time.time()
        run()
        seconds = time.time() - t0
        s = evaluate(model, data)
        history.append({"stage": name, "seconds": seconds, "rmse_train": s["rmse_train"], "rmse_test": s["rmse_test"]})
        print(
            f"== after {name}: {seconds:.1f} s, free-run RMSE "
            f"train = {s['rmse_train'] * 1e3:.3f} mV, test = {s['rmse_test'] * 1e3:.3f} mV"
        )

    out_dir = ROOT / "results" / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(cfg), "scales": asdict(scales), "state_dict": model.state_dict()}, out_dir / "model.pt")
    summary = {"config": asdict(cfg), "n_params": n_params, "history": history}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="JSON file with Config overrides")
    args = parser.parse_args()
    cfg = Config.from_json(args.config) if args.config else Config()
    summary = train(cfg)
    print("\nstage      time [s]   train [mV]   test [mV]")
    for h in summary["history"]:
        print(f"{h['stage']:9s} {h['seconds']:9.1f} {h['rmse_train'] * 1e3:12.3f} {h['rmse_test'] * 1e3:11.3f}")


if __name__ == "__main__":
    main()
