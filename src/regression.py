"""Linear regression utilities and Lemma 1 transformation rules.

The functions ``rotated_regression``, ``translated_regression``, and
``homothety_regression`` implement the closed-form rules that map the
source-domain regression parameters (a, b) to the target-domain
parameters under the corresponding transformation. They follow Lemma 1
of the paper.

The function ``estimate_f_parameters`` is the main estimation routine
combining centroid transport with one of the three transformation rules,
using a sub-sampling / bagging scheme.
"""

import numpy as np
from sklearn.linear_model import LinearRegression

from .estimators import (
    estimate_homothety_constant,
    estimate_rotation_angle_svd,
    estimate_translation,
)


def linear_regression(X):
    """Simple OLS regression of column 1 on column 0 of ``X``.

    Parameters
    ----------
    X : np.ndarray of shape (n, 2)

    Returns
    -------
    a_hat : float
        Slope.
    b_hat : float
        Intercept.
    """
    x = X[:, 0].reshape(-1, 1)
    y = X[:, 1]
    model = LinearRegression().fit(x, y)
    return float(model.coef_[0]), float(model.intercept_)


def rotated_regression(a, b, theta):
    """Map (a, b) to the parameters of the line obtained after rotation by theta.

    See Lemma 1 in the paper.
    """
    a_r = (a * np.cos(theta) + np.sin(theta)) / (np.cos(theta) - a * np.sin(theta))
    b_r = b * (a_r * np.sin(theta) + np.cos(theta))
    return a_r, b_r


def translated_regression(a, b, q):
    """Map (a, b) to the parameters of the line obtained after translation by q.

    See Lemma 1 in the paper.
    """
    a_t = a
    b_t = b + a * q[1] - q[0]
    return a_t, b_t


def homothety_regression(a, b, lambda_):
    """Map (a, b) to the parameters of the line obtained after homothety of factor lambda.

    See Lemma 1 in the paper.
    """
    a_h = a
    b_h = b / lambda_
    return a_h, b_h


def estimate_f_parameters(
    Xs,
    Xt,
    f_type,
    p=2,
    n_rep=2000,
    sampling_frac=0.8,
    rng=None,
    verbose=False,
):
    """Estimate target-domain regression parameters by transferring from Xs to Xt.

    The procedure is:

    1. Fit OLS on the source set Xs to get (a1_hat, b1_hat).
    2. Repeat ``n_rep`` times: subsample a fraction ``sampling_frac`` of Xs,
       estimate the transformation parameters (rotation/translation/homothety)
       between the subsample and Xt via centroid transport, then apply the
       Lemma 1 transformation rule to (a1_hat, b1_hat).
    3. Return the median across iterations as the target-domain estimate.

    Parameters
    ----------
    Xs : np.ndarray of shape (ns, 2)
    Xt : np.ndarray of shape (nt, 2)
    f_type : {"R", "T", "H"}
        Transformation family: Rotation, Translation, or Homothety.
    p : float, optional
        Order of the Lp norm for the OT cost matrix. Default 2.
    n_rep : int, optional
        Number of bootstrap-style iterations.
    sampling_frac : float, optional
        Fraction of Xs sampled in each iteration.
    rng : np.random.Generator, optional
    verbose : bool, optional

    Returns
    -------
    a2_hat, b2_hat : float
        OLS regression on the target set Xt directly.
    a_f, b_f : float
        Adapted regression: source regression mapped to the target
        domain via the estimated transformation.
    """
    if f_type not in {"R", "T", "H"}:
        raise ValueError(f"f_type must be one of 'R', 'T', 'H'. Got {f_type!r}.")

    if rng is None:
        rng = np.random.default_rng()

    ns = Xs.shape[0]

    a1_hat, b1_hat = linear_regression(Xs)
    a2_hat, b2_hat = linear_regression(Xt)

    A_f, B_f = [], []
    n_sample = int(np.floor(ns * sampling_frac))

    for iteration in range(n_rep):
        if verbose and iteration % 500 == 0:
            print(f"  iteration {iteration}/{n_rep}")

        sampled_indices = rng.choice(ns, n_sample, replace=False)
        Xs_sampled = Xs[sampled_indices]

        if f_type == "R":
            theta, _, _ = estimate_rotation_angle_svd(Xs_sampled, Xt, p)
            a_new, b_new = rotated_regression(a1_hat, b1_hat, theta)
        elif f_type == "T":
            t = estimate_translation(Xs_sampled, Xt, p)
            a_new, b_new = translated_regression(a1_hat, b1_hat, t)
        else:  # f_type == "H"
            lam = estimate_homothety_constant(Xs_sampled, Xt, p)
            a_new, b_new = homothety_regression(a1_hat, b1_hat, lam)

        A_f.append(a_new)
        B_f.append(b_new)

    a_f = float(np.median(A_f))
    b_f = float(np.median(B_f))

    return a2_hat, b2_hat, a_f, b_f
