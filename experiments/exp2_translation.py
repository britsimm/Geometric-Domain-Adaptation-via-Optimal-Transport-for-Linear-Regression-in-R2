"""Experiment 2: Translation.

For each translation magnitude ``norm`` in ``NORMS``, generate target data
by translating points sampled from a line by a vector of length ``norm``
and uniformly random direction. Compare the four methods (TO, ALR, LFT, TL)
on MSE against the ground-truth translated line.

Run from the repo root:

    python -m experiments.exp2_translation
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
A = 1.0
SIGMA = 1.0
NS = 1000
NT = 10
N_REP = 100
N_REP_INNER = 1000
P = 2
SAMPLING_FRAC = 0.8

NORMS = [1, 2, 4, 16]

OUT_PATH = Path(__file__).resolve().parent.parent / "results" / "exp2_translation.csv"


def main():
    rng = np.random.default_rng(SEED)
    rows = []

    x_line = np.array([-15.0, 15.0])

    for norm in NORMS:
        for rep in tqdm(range(N_REP), desc=f"||v||={norm}", leave=True, ncols=100):
            xs, ys = generate_sample(A, SIGMA, NS, rng=rng)
            Xs = np.vstack((xs, ys)).T

            theta = rng.uniform(0, 2 * np.pi)
            t = norm * np.array([np.cos(theta), np.sin(theta)])

            xt, yt = generate_sample(A, SIGMA, NT, rng=rng)
            Xt = np.vstack((xt, yt)).T + t

            a_to, b_to, a_alr, b_alr = estimate_f_parameters(
                Xs, Xt, f_type="T", p=P, n_rep=N_REP_INNER,
                sampling_frac=SAMPLING_FRAC, rng=rng,
            )
            a_lft, b_lft = finetuned_regression(Xs, Xt)
            a_tl, b_tl = trans_lasso_regression(Xs, Xt)

            # Ground-truth target line is y = A*x + t_y (the translation
            # of the source line by t).
            y_truth = A * x_line + t[1]

            rows.append({
                "norm": norm,
                "rep": rep,
                "theta_translation": theta,
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
