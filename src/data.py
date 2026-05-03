"""Data generation and geometric transformations for the simulation experiments."""

import numpy as np


def generate_sample(a, sigma, num_points, x_range=(-10, 10), rng=None):
    """Generate noisy samples from a line y = a*x + epsilon.

    Parameters
    ----------
    a : float
        Slope of the line.
    sigma : float
        Standard deviation of the Gaussian noise on y.
    num_points : int
        Number of points to sample.
    x_range : tuple, optional
        (low, high) range for the uniformly sampled x values. Default (-10, 10).
    rng : np.random.Generator, optional
        Random generator. If None, uses np.random (legacy global state).

    Returns
    -------
    x : np.ndarray of shape (num_points,)
    y : np.ndarray of shape (num_points,)
    """
    low, high = x_range
    if rng is None:
        x = np.random.rand(num_points) * (high - low) + low
        noise = np.random.normal(loc=0.0, scale=sigma, size=num_points)
    else:
        x = rng.uniform(low, high, size=num_points)
        noise = rng.normal(loc=0.0, scale=sigma, size=num_points)
    y = a * x + noise
    return x, y


def rotate_points(x, y, theta):
    """Rotate 2D points by angle ``theta`` (radians) around the origin.

    Parameters
    ----------
    x, y : array-like
        Coordinates of the points (scalars or arrays).
    theta : float
        Rotation angle in radians.

    Returns
    -------
    x_r, y_r : np.ndarray
        Rotated coordinates.
    """
    theta = float(theta)
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)],
         [np.sin(theta),  np.cos(theta)]]
    )
    xy_rotated = rotation_matrix @ np.vstack((x, y))
    return xy_rotated[0], xy_rotated[1]


def translate_points(X, t):
    """Translate a set of points by vector ``t``.

    Parameters
    ----------
    X : np.ndarray of shape (n, 2)
    t : np.ndarray of shape (2,)

    Returns
    -------
    np.ndarray of shape (n, 2)
    """
    return X + t
