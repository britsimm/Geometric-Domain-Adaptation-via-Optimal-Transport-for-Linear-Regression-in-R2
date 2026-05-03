"""Estimators of the geometric transformation between Xs and Xt via centroid transport."""

import numpy as np

from .transport import centroid_transport


def estimate_rotation_angle_svd(Xs, Xt, p=2):
    """Estimate the rotation angle that maps Xs to Xt using OT + SVD.

    Computes K-means centroids of Xs, transports them to Xt via OT,
    then recovers the best rotation aligning centroids to their
    transported images using a Procrustes-like SVD step.

    Parameters
    ----------
    Xs : np.ndarray of shape (ns, 2)
    Xt : np.ndarray of shape (nt, 2)
    p : float, optional
        Order of the Lp norm for the OT cost matrix. Default 2.

    Returns
    -------
    theta : float
        Estimated rotation angle in radians.
    centroids : np.ndarray of shape (nt, 2)
    transported_centroids : np.ndarray of shape (nt, 2)
    """
    centroids, transported_centroids = centroid_transport(Xs, Xt, p)

    H = centroids.T @ transported_centroids
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Avoid reflections (det should be +1 for a proper rotation).
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    theta = np.arctan2(R[1, 0], R[0, 0])
    return theta, centroids, transported_centroids


def estimate_translation(Xs, Xt, p=2):
    """Estimate the translation vector mapping Xs to Xt using OT.

    Returns
    -------
    translation : np.ndarray of shape (2,)
        Mean displacement between centroids and their transported targets.
    """
    centroids, transported_centroids = centroid_transport(Xs, Xt, p)
    translation = np.mean(transported_centroids - centroids, axis=0)
    return translation


def estimate_homothety_constant(Xs, Xt, p=2):
    """Estimate the homothety scale factor lambda mapping Xs to Xt using OT.

    Returns
    -------
    lambda_hat : float
        Median of the elementwise ratio between transported centroids
        and original centroids.
    """
    centroids, transported_centroids = centroid_transport(Xs, Xt, p)
    return np.median(transported_centroids / centroids)
