"""Shared utilities for the quartic Bézier interpolation methods.

This module contains derivative estimation, quartic Bézier evaluation,
diagnostics, and input-validation helpers used by both the exact and bounded
quartic constructions.
"""

import numpy as np


# ============================================================
# Derivative estimation
# ============================================================

def estimate_node_derivatives(x_nodes, y_nodes, method="pchip"):
    """
    Parameters
    ----------
    x_nodes : array-like, shape (n,)
        Strictly increasing support points.

    y_nodes : array-like, shape (n,)
        Function values at support points.

    method : str
        "pchip" or "finite_difference".

    Returns
    -------
    derivatives : ndarray, shape (n,)
    """

    x_nodes, y_nodes = _validate_xy_nodes(x_nodes, y_nodes)

    if method == "pchip":
        return estimate_pchip_derivatives_numpy(x_nodes, y_nodes)

    if method == "finite_difference":
        return np.gradient(y_nodes, x_nodes)

    raise ValueError(
        "Unknown derivative_method. Use 'pchip' or 'finite_difference'."
    )


def estimate_pchip_derivatives_numpy(x_nodes, y_nodes):
    """
    This follows the classical Fritsch-Carlson / Fritsch-Butland idea:

    1) Compute secant slopes on each interval:

           delta_i = (y_{i+1} - y_i) / (x_{i+1} - x_i).

    2) For an interior node x_i:

       If delta_{i-1} and delta_i have opposite signs, or one of them
       is zero, set derivative d_i = 0. This prevents artificial turning
       points and is the main shape-preserving idea.

       Otherwise use a weighted harmonic mean of the neighboring secants.
       This keeps the derivative between the two slopes and strongly
       reduces overshoot compared with ordinary centered differences.

    3) Endpoints use a one-sided PCHIP endpoint formula, followed by
       limiting to avoid overshoot.

    Returns
    -------
    d : ndarray, shape (n,)
        Shape-preserving derivative estimates.

    Notes
    -----
    This function intentionally does not import scipy. It is meant to
    replace scipy.interpolate.PchipInterpolator(...).derivative() for the
    purpose of obtaining stable slopes for the quartic Bézier method.
    """

    x, y = _validate_xy_nodes(x_nodes, y_nodes)
    n = len(x)

    if n == 2:
        slope = (y[1] - y[0]) / (x[1] - x[0])
        return np.array([slope, slope], dtype=float)

    h = np.diff(x)
    delta = np.diff(y) / h
    d = np.zeros(n, dtype=float)

    # Interior nodes
    for i in range(1, n - 1):
        left = delta[i - 1]
        right = delta[i]

        if left == 0.0 or right == 0.0 or np.sign(left) != np.sign(right):
            d[i] = 0.0
        else:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            d[i] = (w1 + w2) / (w1 / left + w2 / right)

    # Endpoint nodes
    d[0] = _pchip_endpoint_slope(h[0], h[1], delta[0], delta[1])
    d[-1] = _pchip_endpoint_slope(h[-1], h[-2], delta[-1], delta[-2])

    return d


def _pchip_endpoint_slope(h0, h1, delta0, delta1):
    """
    One-sided endpoint slope used by PCHIP.

    The raw one-sided estimate is

        d = ((2*h0 + h1)*delta0 - h0*delta1) / (h0 + h1).

    Then it is limited:

        - if d has the wrong sign, set d = 0;
        - if the first two secants have opposite signs and d is too large,
          limit it to 3*delta0.
    """

    d = ((2.0 * h0 + h1) * delta0 - h0 * delta1) / (h0 + h1)

    if d == 0.0:
        return 0.0

    if np.sign(d) != np.sign(delta0):
        return 0.0

    if np.sign(delta0) != np.sign(delta1) and abs(d) > abs(3.0 * delta0):
        return 3.0 * delta0

    return d


# ============================================================
# Quartic Bézier evaluation
# ============================================================

def evaluate_quartic_bezier_segment(P, t):
    """
    Parameters
    ----------
    P : array-like, shape (5,)
        Control points [P0, P1, P2, P3, P4].

    t : float or ndarray
        Local Bézier parameter in [0, 1].

    Returns
    -------
    value : float or ndarray
        Interpolated value(s).
    """

    P = np.asarray(P, dtype=float)
    if P.shape != (5,):
        raise ValueError("P must have shape (5,)")

    t = np.asarray(t, dtype=float)
    one_minus_t = 1.0 - t

    P0, P1, P2, P3, P4 = P

    return (
        P0 * one_minus_t ** 4
        + 4.0 * P1 * one_minus_t ** 3 * t
        + 6.0 * P2 * one_minus_t ** 2 * t ** 2
        + 4.0 * P3 * one_minus_t * t ** 3
        + P4 * t ** 4
    )


def evaluate_piecewise_quartic_bezier(x_nodes, controls, num_points=100):
    """
    Evaluate the full piecewise quartic Bézier interpolant on a dense grid.

    Parameters
    ----------
    x_nodes : array-like, shape (n,)
        Support points.

    controls : ndarray, shape (n-1, 5)
        Quartic Bézier control points with shape (n-1, 5).

    num_points : int
        Number of evaluation points per interval.

    Returns
    -------
    x_dense : ndarray
    y_dense : ndarray
    """

    x_nodes = np.asarray(x_nodes, dtype=float)
    controls = np.asarray(controls, dtype=float)

    if controls.shape != (len(x_nodes) - 1, 5):
        raise ValueError("controls must have shape (len(x_nodes)-1, 5)")

    if num_points < 2:
        raise ValueError("num_points must be at least 2")

    x_all = []
    y_all = []

    for i in range(len(x_nodes) - 1):
        x0 = x_nodes[i]
        x1 = x_nodes[i + 1]

        if i < len(x_nodes) - 2:
            t = np.linspace(0.0, 1.0, num_points, endpoint=False)
        else:
            t = np.linspace(0.0, 1.0, num_points)

        x = x0 + (x1 - x0) * t
        y = evaluate_quartic_bezier_segment(controls[i], t)

        x_all.append(x)
        y_all.append(y)

    return np.concatenate(x_all), np.concatenate(y_all)


# ============================================================
# Diagnostics
# ============================================================

def verify_bound_constraints(controls, lower_bound=None, upper_bound=None):
    """
    Check optional physical bounds for all control points.

    Returns
    -------
    max_lower_violation : float
    max_upper_violation : float
    """

    controls = np.asarray(controls, dtype=float)

    if lower_bound is None:
        max_lower_violation = 0.0
    else:
        max_lower_violation = float(np.max(np.maximum(lower_bound - controls, 0.0)))

    if upper_bound is None:
        max_upper_violation = 0.0
    else:
        max_upper_violation = float(np.max(np.maximum(controls - upper_bound, 0.0)))

    return max_lower_violation, max_upper_violation


def compute_control_polygon_roughness(controls):
    """
    Simple diagnostic for visual smoothness of the control polygon.

    This is not an interpolation constraint. It is only useful when
    comparing methods. A larger value often indicates stronger local
    bending or possible oscillation.

    The measure is the sum of squared second differences of the control
    points within each interval.
    """

    controls = np.asarray(controls, dtype=float)
    roughness = 0.0

    for P in controls:
        second_diff = P[:-2] - 2.0 * P[1:-1] + P[2:]
        roughness += float(np.sum(second_diff ** 2))

    return roughness


# ============================================================
# Input validation
# ============================================================

def _validate_input(x_nodes, y_nodes, means):
    """
    Validate input arrays for the quartic mean interpolation problem.
    """
    x_nodes, y_nodes = _validate_xy_nodes(x_nodes, y_nodes)
    means = np.asarray(means, dtype=float)

    if len(means) != len(x_nodes) - 1:
        raise ValueError("means must have length len(x_nodes)-1")

    if not np.isfinite(means).all():
        raise ValueError("means contains NaN or inf")

    return x_nodes, y_nodes, means


def _validate_xy_nodes(x_nodes, y_nodes):
    x_nodes = np.asarray(x_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)

    if x_nodes.ndim != 1 or y_nodes.ndim != 1:
        raise ValueError("x_nodes and y_nodes must be one-dimensional")

    if len(x_nodes) != len(y_nodes):
        raise ValueError("x_nodes and y_nodes must have the same length")

    if len(x_nodes) < 2:
        raise ValueError("At least two support points are required")

    if np.any(np.diff(x_nodes) <= 0):
        raise ValueError("x_nodes must be strictly increasing")

    if not (np.isfinite(x_nodes).all() and np.isfinite(y_nodes).all()):
        raise ValueError("x_nodes or y_nodes contains NaN or inf")

    return x_nodes, y_nodes


def _validate_derivatives(derivatives, expected_length):
    derivatives = np.asarray(derivatives, dtype=float)

    if derivatives.ndim != 1:
        raise ValueError("derivatives must be one-dimensional")

    if len(derivatives) != expected_length:
        raise ValueError("derivatives must have the same length as x_nodes")

    if not np.isfinite(derivatives).all():
        raise ValueError("derivatives contains NaN or inf")

    return derivatives


def _validate_bounds(lower_bound, upper_bound):
    lb = None if lower_bound is None else float(lower_bound)
    ub = None if upper_bound is None else float(upper_bound)

    if lb is not None and not np.isfinite(lb):
        raise ValueError("lower_bound must be finite or None")

    if ub is not None and not np.isfinite(ub):
        raise ValueError("upper_bound must be finite or None")

    if lb is not None and ub is not None and lb > ub:
        raise ValueError("lower_bound must not exceed upper_bound")

    return lb, ub


def _check_endpoint_bounds(y_nodes, lower_bound, upper_bound):
    if lower_bound is not None and np.any(y_nodes < lower_bound):
        raise ValueError(
            "Support values violate lower_bound, so exact bounded "
            "interpolation is impossible."
        )

    if upper_bound is not None and np.any(y_nodes > upper_bound):
        raise ValueError(
            "Support values violate upper_bound, so exact bounded "
            "interpolation is impossible."
        )


