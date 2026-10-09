# silverbox-pinn

A neural acceleration model for the Silverbox oscillator. The state is (y, v) with y' = v and v' = alpha_theta(y, v, u). The network alpha_theta is a fully connected network, and the state is integrated with RK4 at the sampling step.

The full write-up is in [report.md](report.md) and [report.pdf](report.pdf).

| model | free-run RMSE train | free-run RMSE test |
|---|---|---|
| zero prediction | 53.0 mV | 60.5 mV |
| 3x64 SiLU network (kept, `results/best.pt`) | 1.029 mV | 2.273 mV |
| Duffing equation, 4 coefficients (reference) | 0.954 mV | 0.996 mV |

## Data

The data come from the Silverbox benchmark by Wigren and Schoukens (ECC 2013), available from <https://www.nonlinearbenchmark.org/>.

1. Download the Silverbox data set.
2. Take the file `SNLS80mV.mat` from it.
3. Put it at `data/SNLS80mV.mat`.

The `data/` directory is not tracked by git.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt   # numpy, scipy, torch, matplotlib, pytest, markdown
```

Python 3.10+ is required. Everything runs on CPU.

## Reproducing the results

All commands run from the repository root.

```bash
# data check: shapes, test RMS, figures/data_*.png
.venv/bin/python scripts/check_data.py

# RK4 convergence test (order 4)
.venv/bin/python -m pytest tests -q

# kept model: one free run from results/best.pt, no training (~5 s)
#   -> both RMSE values, figures/final_test.png, figures/final_force.png
.venv/bin/python scripts/final.py

# Duffing reference with 4 trainable coefficients (~3 min)
.venv/bin/python scripts/baseline_duffing.py

# retrain the kept configuration from scratch (~15 min)
#   -> results/3x64_silu_retrain/
.venv/bin/python -m src.train --config configs/best.json
.venv/bin/python -m src.evaluate results/3x64_silu_retrain/model.pt

# sweep over architectures and hyperparameters
#   resumes from results/sweep.csv; one run takes 6-23 min
.venv/bin/python scripts/sweep.py --workers 4
.venv/bin/python scripts/sweep.py --summary-only   # table only, copies the best run to results/best.pt

# build report.pdf from report.md (needs Google Chrome or Chromium)
.venv/bin/python scripts/build_report.py
```

Training uses seed 0 and a single thread per run. A retrain on the same machine and library versions reproduces the numbers above.

## Layout

```
src/data.py            excerpt [42650:46746], centering, train/test split
src/integrator.py      batched float64 RK4 and rollout
src/metrics.py         free-run metric from TASK.md
src/model.py           acceleration MLP with fixed input/output scales
src/train.py           pretraining, rollout training, optional full free-run finetune
src/evaluate.py        free-run evaluation and test plot of a checkpoint
scripts/               data check, Duffing baseline, sweep, final evaluation, report build
configs/best.json      configuration of the kept model
results/               sweep table, checkpoints, JSON summaries
figures/               all plots
```
