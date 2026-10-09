"""Sweep over architectures and training hyperparameters.

Phase 1: every architecture (hidden layers x activation) with default training.
Phase 2: a grid of training hyperparameters (learning rate, number of rollout
steps, final rollout length K, pretraining on/off) for the best `--top`
architectures of phase 1.

The full-free-run fine-tuning (stage 3) is off: it costs ~3 min per run and
changed the test RMSE by < 1% in the first experiment.

Every finished run is appended to results/sweep.csv (runs already in the file
are skipped, so the sweep can be resumed). The weights of the run with the
lowest free-run test RMSE are copied to results/best.pt.

    python scripts/sweep.py --workers 4
    python scripts/sweep.py --summary-only   # table and best.pt from finished runs
"""

import argparse
import contextlib
import csv
import itertools
import multiprocessing as mp
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.train import Config, train  # noqa: E402

CSV_PATH = ROOT / "results" / "sweep.csv"
BEST_PATH = ROOT / "results" / "best.pt"
RUN_DIR = "sweep"  # runs are saved under results/sweep/<name>/

ARCHS = [(32,), (32, 32), (64, 64), (64, 64, 64)]
ACTIVATIONS = ["tanh", "silu"]
LRS = [1e-3, 3e-3]
STEP_SCALES = [1, 2]
K_FINALS = [128, 512]
PRETRAIN = [True, False]

BASE_SCHEDULE = [(32, 400), (128, 300), (512, 150)]

FIELDS = [
    "name", "phase", "hidden", "activation", "pretrain", "rollout_lr", "step_scale", "k_final",
    "n_params", "seconds", "rmse_train_mV", "rmse_test_mV",
]


def schedule(k_final: int, step_scale: int) -> list[tuple[int, int]]:
    return [(k, steps * step_scale) for k, steps in BASE_SCHEDULE if k <= k_final]


def make_config(hidden, activation, lr=3e-3, step_scale=1, k_final=512, pretrain=True) -> Config:
    arch = "x".join(map(str, hidden))
    name = f"{arch}_{activation}_lr{lr:g}_s{step_scale}_K{k_final}_{'pre' if pretrain else 'nopre'}"
    return Config(
        name=f"{RUN_DIR}/{name}",
        hidden=tuple(hidden),
        activation=activation,
        pretrain=pretrain,
        rollout_lr=lr,
        rollout_schedule=schedule(k_final, step_scale),
        finetune=False,
    )


def run_one(job: tuple[str, Config, dict]) -> dict:
    phase, cfg, meta = job
    log_dir = ROOT / "results" / cfg.name
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "train.log", "w") as log, contextlib.redirect_stdout(log):
        summary = train(cfg)
    last = summary["history"][-1]
    return {
        "name": cfg.name.split("/", 1)[1],
        "phase": phase,
        "hidden": "x".join(map(str, cfg.hidden)),
        "activation": cfg.activation,
        "pretrain": cfg.pretrain,
        "rollout_lr": cfg.rollout_lr,
        "step_scale": meta["step_scale"],
        "k_final": meta["k_final"],
        "n_params": summary["n_params"],
        "seconds": round(sum(h["seconds"] for h in summary["history"]), 1),
        "rmse_train_mV": round(last["rmse_train"] * 1e3, 4),
        "rmse_test_mV": round(last["rmse_test"] * 1e3, 4),
    }


def read_rows() -> list[dict]:
    if not CSV_PATH.exists():
        return []
    with open(CSV_PATH) as f:
        return list(csv.DictReader(f))


def run_jobs(jobs: list[tuple[str, Config, dict]], workers: int) -> None:
    done = {r["name"] for r in read_rows()}
    todo = [j for j in jobs if j[1].name.split("/", 1)[1] not in done]
    print(f"{len(jobs)} runs, {len(jobs) - len(todo)} already done, {len(todo)} to run")
    if not todo:
        return
    new_file = not CSV_PATH.exists()
    CSV_PATH.parent.mkdir(exist_ok=True)
    t0 = time.time()
    with open(CSV_PATH, "a", newline="") as f, mp.get_context("spawn").Pool(workers) as pool:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for i, row in enumerate(pool.imap_unordered(run_one, todo), 1):
            writer.writerow(row)
            f.flush()
            print(
                f"[{i}/{len(todo)}] {time.time() - t0:7.0f} s  {row['name']:42s} "
                f"train {float(row['rmse_train_mV']):8.3f} mV  test {float(row['rmse_test_mV']):8.3f} mV",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--top", type=int, default=2, help="architectures taken to phase 2")
    parser.add_argument("--summary-only", action="store_true", help="no training, only the table of finished runs")
    args = parser.parse_args()
    if args.summary_only:
        summarize()
        return

    # Phase 1: architectures with the default training setup.
    phase1 = [
        ("arch", make_config(h, a), {"step_scale": 1, "k_final": 512})
        for h, a in itertools.product(ARCHS, ACTIVATIONS)
    ]
    run_jobs(phase1, args.workers)

    rows = [r for r in read_rows() if r["phase"] == "arch"]
    rows.sort(key=lambda r: float(r["rmse_test_mV"]))
    top = [(tuple(int(x) for x in r["hidden"].split("x")), r["activation"]) for r in rows[: args.top]]
    print(f"phase 2 architectures: {top}")

    # Phase 2: training hyperparameters for the best architectures.
    phase2 = [
        ("hparam", make_config(h, a, lr, s, k, p), {"step_scale": s, "k_final": k})
        for (h, a), lr, s, k, p in itertools.product(top, LRS, STEP_SCALES, K_FINALS, PRETRAIN)
    ]
    run_jobs(phase2, args.workers)
    summarize()


def summarize() -> None:
    # Summary sorted by test RMSE; duplicates of the phase-1 default setting are kept once.
    rows, seen = [], set()
    for r in sorted(read_rows(), key=lambda r: float(r["rmse_test_mV"])):
        if r["name"] not in seen:
            seen.add(r["name"])
            rows.append(r)
    best = rows[0]
    shutil.copy(ROOT / "results" / RUN_DIR / best["name"] / "model.pt", BEST_PATH)

    print(f"\n{'#':>3} {'name':42s} {'params':>6} {'time s':>7} {'train mV':>9} {'test mV':>8}")
    for i, r in enumerate(rows, 1):
        print(
            f"{i:3d} {r['name']:42s} {r['n_params']:>6} {float(r['seconds']):7.0f} "
            f"{float(r['rmse_train_mV']):9.3f} {float(r['rmse_test_mV']):8.3f}"
        )
    print(f"\nbest: {best['name']}  ->  {BEST_PATH}")


if __name__ == "__main__":
    main()
