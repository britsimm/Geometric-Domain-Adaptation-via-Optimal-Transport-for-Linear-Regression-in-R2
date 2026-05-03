"""Experiment 3: Homothety.

For each scale factor ``lambda_`` in ``LAMBDAS``, generate target data
by sampling from a line and applying a homothety of factor ``lambda_``.
Compare the four methods (TO, ALR, LFT, TL) on MSE against the ground-truth
scaled line.

Run from the repo root:

    python -m experiments.exp3_homothety
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.baselines import finetuned_regression, trans_lasso_regression
from src.data import generate_sample
from src.metrics import mean_squared_error
from src.regression import estimate_f_parameters


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
A = 0.0
SIGMA = 0.1
NS = 1000
NT = 10
N_REP = 100
N_REP_INNER = 1000
P = 2
SAMPLING_FRAC = 0.8

LAMBDAS = [-2.0, -1.0, 0.5, 4.0]

OUT_PATH = Path(__file__).resolve().parent.parent / "results" / "exp3_homothety.csv"


def main():
    rng = np.random.default_rng(SEED)
    rows = []

    for lambda_ in LAMBDAS:
        x_line = np.array([-10.0 * lambda_, 10.0 * lambda_])
        # Ground-truth target line is y = lambda * A * x.
        y_truth = lambda_ * A * x_line

        for rep in tqdm(range(N_REP), desc=f"lambda={lambda_}", leave=True, ncols=100):
            xs, ys = generate_sample(A, SIGMA, NS, rng=rng)
            Xs = np.vstack((xs, ys)).T

            xt, yt = generate_sample(A, SIGMA, NT, rng=rng)
            Xt = lambda_ * np.vstack((xt, yt)).T

            a_to, b_to, a_alr, b_alr = estimate_f_parameters(
                Xs, Xt, f_type="H", p=P, n_rep=N_REP_INNER,
                sampling_frac=SAMPLING_FRAC, rng=rng,
            )
            a_lft, b_lft = finetuned_regression(Xs, Xt)
            a_tl, b_tl = trans_lasso_regression(Xs, Xt)

            rows.append({
                "lambda": lambda_,
                "rep": rep,
                "mse_TO":  mean_squared_error(y_truth, a_to  * x_line + b_to),
                "mse_ALR": mean_squared_error(y_truth, a_alr * x_line + b_alr),
                "mse_LFT": mean_squared_error(y_truth, a_lft * x_line + b_lft),
                "mse_TL":  mean_squared_error(y_truth, a_tl  * x_line + b_tl),
            })

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
