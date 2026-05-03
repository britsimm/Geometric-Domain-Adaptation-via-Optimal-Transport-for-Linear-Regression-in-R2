"""Optimal transport utilities: cost matrices and centroid-based transport."""

import numpy as np
import ot
from sklearn.cluster import KMeans


def lp_distance_matrix(set1, set2, p=2):
    """Pairwise Lp distance matrix between two sets of 2D points.

    Parameters
    ----------
    set1 : np.ndarray of shape (n, 2)
    set2 : np.ndarray of shape (m, 2)
    p : float or "inf"
        Order of the norm. p=1 -> Manhattan, p=2 -> Euclidean,
        p="inf" -> Chebyshev.

    Returns
    -------
    np.ndarray of shape (n, m)
    """
    n, m = set1.shape[0], set2.shape[0]
    distance_matrix = np.zeros((n, m))

    if p == "inf":
        for i in range(n):
            distance_matrix[i] = np.max(np.abs(set1[i] - set2), axis=1)
    else:
        for i in range(n):
            diff = np.abs(set1[i] - set2)
            distance_matrix[i] = np.sum(diff ** p, axis=1) ** (1.0 / p)

    return distance_matrix


def centroid_transport(Xs, Xt, p=2, random_state=0):
    """Compute optimal transport between K-means centroids of Xs and points in Xt.

    K-means is run on the source set ``Xs`` to obtain ``nt`` centroids
    (where ``nt = len(Xt)``). Then an exact optimal transport plan
    is solved between centroids (uniform weights) and target points
    (uniform weights), using the Lp cost matrix.

    Parameters
    ----------
    Xs : np.ndarray of shape (ns, 2)
        Source samples.
    Xt : np.ndarray of shape (nt, 2)
        Target samples.
    p : float, optional
        Order of the Lp norm used in the cost matrix. Default 2.
    random_state : int, optional
        Random state for K-means.

    Returns
    -------
    centroids : np.ndarray of shape (nt, 2)
        Centroids extracted from Xs via K-means.
    transported_centroids : np.ndarray of shape (nt, 2)
        Target points each centroid is mapped to under the OT plan.
    """
    nt = Xt.shape[0]

    kmeans = KMeans(n_clusters=nt, random_state=random_state, n_init=1).fit(Xs)
    centroids = kmeans.cluster_centers_

    mu_centroids = np.ones(nt) / nt
    nu = np.ones(nt) / nt

    C = lp_distance_matrix(centroids, Xt, p)
    P_gamma = ot.emd(mu_centroids, nu, C)

    mapped_indices = np.argmax(P_gamma, axis=1)
    transported_centroids = Xt[mapped_indices]

    return centroids, transported_centroids
