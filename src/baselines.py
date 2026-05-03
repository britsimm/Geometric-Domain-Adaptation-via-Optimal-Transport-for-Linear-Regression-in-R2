"""Baseline transfer-learning methods for comparison: LFT and Trans-Lasso."""

import numpy as np
from sklearn.linear_model import Lasso

from .regression import linear_regression
from .transport import centroid_transport


def finetuned_regression(Xs, Xt, learning_rate=1e-3, epochs=1000):
    """Linear regression fine-tuned on Xt, initialized from the OLS fit on Xs.

    Implements a simple gradient descent on the MSE loss in the target
    domain, starting from the source-domain OLS parameters.

    Parameters
    ----------
    Xs : np.ndarray of shape (ns, 2)
    Xt : np.ndarray of shape (nt, 2)
    learning_rate : float, optional
    epochs : int, optional

    Returns
    -------
    a_gd, b_gd : float
        Fine-tuned slope and intercept.
    """
    a_gd, b_gd = linear_regression(Xs)

    x_target = Xt[:, 0]
    y_target = Xt[:, 1]

    for _ in range(epochs):
        y_pred = a_gd * x_target + b_gd
        da = -2.0 * np.mean(x_target * (y_target - y_pred))
        db = -2.0 * np.mean(y_target - y_pred)
        a_gd -= learning_rate * da
        b_gd -= learning_rate * db

    return a_gd, b_gd


def trans_lasso_regression(Xs, Xt, p=2, alpha=0.1, random_state=42):
    """Trans-Lasso baseline: Lasso fitted on transported centroids.

    Parameters
    ----------
    Xs : np.ndarray of shape (ns, 2)
    Xt : np.ndarray of shape (nt, 2)
    p : float, optional
        Order of the Lp norm for the OT cost matrix. Default 2.
    alpha : float, optional
        Lasso regularization strength.
    random_state : int, optional

    Returns
    -------
    a_lasso, b_lasso : float
    """
    _, transported_centroids = centroid_transport(Xs, Xt, p)

    x_transformed = transported_centroids[:, 0].reshape(-1, 1)
    y_transformed = transported_centroids[:, 1]

    lasso = Lasso(alpha=alpha, random_state=random_state)
    lasso.fit(x_transformed, y_transformed)

    return float(lasso.coef_[0]), float(lasso.intercept_)
