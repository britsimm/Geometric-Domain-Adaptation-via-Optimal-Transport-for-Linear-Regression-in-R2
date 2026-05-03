"""Experiment 1: Rotation.

For each combination of rotation angle ``theta`` and noise level ``sigma``:
- Generate a source dataset on the line y = 0.
- Generate a target dataset by sampling on the line y = 0 and then
  rotating the points by ``theta``.
- Estimate the regression parameters in the target domain using:
    * TO  - OLS directly on the target set (target-only).
    * ALR - Adapted Linear Regression: source OLS pushed forward
            via the OT-estimated rotation (this work).
    * LFT - Linear Fine-Tuning of source OLS on target via gradient descent.
    * TL  - Trans-Lasso baseline.
- Repeat ``N_REPS`` times and store all MSEs to a CSV.

Run from the repo root:

    python -m experiments.exp1_rotation
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from src.baselines import finetuned_regression, trans_lasso_regression
from src.data import generate_sample, rotate_points
from src.metrics import mean_squared_error
from src.regression import estimate_f_parameters


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
NS = 1000              # source sample size
NT = 50                # target sample size
P = 2                  # Lp norm order for OT cost
N_REPS = 50            # repetitions per (theta, sigma) combination
N_REP_INNER = 50       # bootstrap iterations inside estimate_f_parameters
SAMPLING_FRAC = 0.8
N_JOBS = 3             # parallel workers across (theta, sigma) combinations

THETAS = [
    (5 / 6) * np.pi,
    (3 / 4) * np.pi,
    (2 / 3) * np.pi,
    (1 / 2 + 1 / 20) * np.pi,
    (1 / 2 - 1 / 20) * np.pi,
    (1 / 3) * np.pi,
    (1 / 4) * np.pi,
    (1 / 6) * np.pi,
]
NOISES = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

OUT_PATH = Path(__file__).resolve().parent.parent / "results" / "exp1_rotation.csv"


def run_one_combination(theta_real, sigma, ns, nt, p, n_reps, n_rep_inner, sampling_frac, base_seed):
    """Run ``n_reps`` independent repetitions for a single (theta, sigma) pair."""
    a_source = 0.0
    a_ground_truth = np.tan(theta_real)

    x_line = np.linspace(-10, 10, 100)
    y_ground_truth = a_ground_truth * x_line

    # Combination-specific seed so workers are deterministic and independent.
    combo_seed = (
        base_seed + int(round(theta_real * 1e6)) * 1000 + int(round(sigma * 1e3))
    ) % (2**31 - 1)
    rng = np.random.default_rng(combo_seed)

    rows = []
    for _ in tqdm(range(n_reps),
                  desc=f"theta={theta_real/np.pi:.2f}pi, sigma={sigma}",
                  leave=False):
        xs, ys = generate_sample(a_source, sigma, ns, rng=rng)
        Xs = np.vstack((xs, ys)).T

        xt_base, yt_base = generate_sample(a_source, sigma, nt, rng=rng)
        xt_rot, yt_rot = rotate_points(xt_base, yt_base, theta_real)
        Xt = np.vstack((xt_rot, yt_rot)).T

        a_to, b_to, a_alr, b_alr = estimate_f_parameters(
            Xs, Xt, f_type="R", p=p, n_rep=n_rep_inner,
            sampling_frac=sampling_frac, rng=rng,
        )
        a_lft, b_lft = finetuned_regression(Xs, Xt)
        a_tl, b_tl = trans_lasso_regression(Xs, Xt)

        mse = {
            "TO":  mean_squared_error(y_ground_truth, a_to  * x_line + b_to),
            "ALR": mean_squared_error(y_ground_truth, a_alr * x_line + b_alr),
            "LFT": mean_squared_error(y_ground_truth, a_lft * x_line + b_lft),
            "TL":  mean_squared_error(y_ground_truth, a_tl  * x_line + b_tl),
        }
        rows.append({
            "theta": theta_real,
            "noise": sigma,
            "mse_TO":  mse["TO"],
            "mse_ALR": mse["ALR"],
            "mse_LFT": mse["LFT"],
            "mse_TL":  mse["TL"],
        })

    return rows


def main():
    np.random.seed(SEED)

    param_combinations = [(theta, sigma) for theta in THETAS for sigma in NOISES]

    all_batches = Parallel(n_jobs=N_JOBS)(
        delayed(run_one_combination)(
            theta, sigma, NS, NT, P, N_REPS, N_REP_INNER, SAMPLING_FRAC, SEED,
        )
        for theta, sigma in tqdm(param_combinations, desc="Overall progress", ncols=100)
    )

    rows = [row for batch in all_batches for row in batch]
    df = pd.DataFrame(rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
