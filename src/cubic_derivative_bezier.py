import numpy as np


# ============================================================
# Cubic Bézier interpolation based on
#   - support point values
#   - first derivatives at the support points
#
# On each interval [x_i, x_{i+1}] we build one cubic Bézier
# segment whose endpoint values and endpoint derivatives are
# matched exactly.
#
# Important:
# In this Hermite-style construction, the derivatives are part
# of the exact interpolation conditions. Therefore, if the
# resulting control points violate the requested bounds, the
# method raises an error instead of modifying the derivatives.
# ============================================================


def build_derivative_bezier_interpolant(
    x_nodes,
    y_nodes,
    derivatives,
    lower_bound=0,
    upper_bound=None,
):
    """
    Main interface function.

    Construct a piecewise cubic Bézier interpolant from:
        - support points (x_nodes, y_nodes)
        - first derivatives at the support points

    Parameters
    ----------
    x_nodes : array-like, shape (n,)
        Strictly increasing support points.

    y_nodes : array-like, shape (n,)
        Function values at support points.

    derivatives : array-like, shape (n,)
        Derivative values at the support points.

    lower_bound : float or None, default=None
        Optional lower bound for the whole interpolant.

    upper_bound : float or None, default=None
        Optional upper bound for the whole interpolant.

    Returns
    -------
    controls : ndarray, shape (n-1, 4)
        Bézier control points for each interval.
    """

    x_nodes, y_nodes, derivatives = _validate_input(x_nodes, y_nodes, derivatives)
    lb, ub = _validate_bounds(lower_bound, upper_bound)

    controls = _compute_derivative_bezier_control_points(
        x_nodes=x_nodes,
        y_nodes=y_nodes,
        derivatives=derivatives,
        lower_bound=lb,
        upper_bound=ub,
    )

    return controls


# ============================================================
# Core construction
# ============================================================

def _compute_derivative_bezier_control_points(
    x_nodes,
    y_nodes,
    derivatives,
    lower_bound,
    upper_bound,
):
    """
    Cubic Bézier control points from endpoint values and derivatives.

    For one interval [x_i, x_{i+1}] with h_i = x_{i+1} - x_i:

        P0 = y_i
        P1 = y_i + (h_i / 3) * d_i
        P2 = y_{i+1} - (h_i / 3) * d_{i+1}
        P3 = y_{i+1}

    This guarantees:

        B_i(0) = y_i
        B_i(1) = y_{i+1}
        d/dx B_i(x_i)     = d_i
        d/dx B_i(x_{i+1}) = d_{i+1}
    """

    n_intervals = len(x_nodes) - 1
    controls = np.zeros((n_intervals, 4), dtype=float)

    for i in range(n_intervals):
        x0 = x_nodes[i]
        x1 = x_nodes[i + 1]
        h = x1 - x0

        y0 = y_nodes[i]
        y1 = y_nodes[i + 1]

        d0 = derivatives[i]
        d1 = derivatives[i + 1]

        P0 = y0
        P1 = y0 + (h / 3.0) * d0
        P2 = y1 - (h / 3.0) * d1
        P3 = y1

        _check_bounds_for_control_points(
            np.array([P0, P1, P2, P3], dtype=float),
            interval_index=i,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        controls[i] = [P0, P1, P2, P3]

    return controls


# ============================================================
# Evaluation functions
# ============================================================

def evaluate_bezier_segment(P, t):
    """
    Evaluate a cubic Bézier segment.

    Parameters
    ----------
    P : array-like, shape (4,)
        Control points [P0, P1, P2, P3]

    t : float or ndarray
        Parameter in [0, 1]

    Returns
    -------
    value : float or ndarray
    """

    P0, P1, P2, P3 = P
    t = np.asarray(t, dtype=float)

    return (
        P0 * (1 - t) ** 3
        + 3 * P1 * (1 - t) ** 2 * t
        + 3 * P2 * (1 - t) * t ** 2
        + P3 * t ** 3
    )


def evaluate_piecewise_bezier(x_nodes, controls, num_points=100):
    """
    Evaluate the full interpolant on a dense grid.

    Parameters
    ----------
    x_nodes : array-like, shape (n,)
    controls : ndarray, shape (n-1, 4)
    num_points : int
        Number of evaluation points per interval

    Returns
    -------
    x_dense, y_dense : ndarray
    """

    x_nodes = np.asarray(x_nodes, dtype=float)
    n_intervals = len(x_nodes) - 1

    x_all = []
    y_all = []

    for i in range(n_intervals):
        x0 = x_nodes[i]
        x1 = x_nodes[i + 1]

        if i < n_intervals - 1:
            t = np.linspace(0.0, 1.0, num_points, endpoint=False)
        else:
            t = np.linspace(0.0, 1.0, num_points)

        x = x0 + (x1 - x0) * t
        y = evaluate_bezier_segment(controls[i], t)

        x_all.append(x)
        y_all.append(y)

    return np.concatenate(x_all), np.concatenate(y_all)


# ============================================================
# Constraint verification
# ============================================================

def verify_derivative_interpolation_constraints(
    x_nodes,
    y_nodes,
    derivatives,
    controls,
):
    """
    Check whether the endpoint values and endpoint derivatives
    are matched exactly.

    Returns
    -------
    max_point_error : float
    max_derivative_error : float
    """

    x_nodes = np.asarray(x_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)
    derivatives = np.asarray(derivatives, dtype=float)
    controls = np.asarray(controls, dtype=float)

    max_point_error = 0.0
    max_derivative_error = 0.0

    for i in range(len(controls)):
        P0, P1, P2, P3 = controls[i]
        h = x_nodes[i + 1] - x_nodes[i]

        err_left_value = abs(P0 - y_nodes[i])
        err_right_value = abs(P3 - y_nodes[i + 1])

        left_derivative = 3.0 * (P1 - P0) / h
        right_derivative = 3.0 * (P3 - P2) / h

        err_left_derivative = abs(left_derivative - derivatives[i])
        err_right_derivative = abs(right_derivative - derivatives[i + 1])

        max_point_error = max(max_point_error, err_left_value, err_right_value)
        max_derivative_error = max(
            max_derivative_error,
            err_left_derivative,
            err_right_derivative,
        )

    return max_point_error, max_derivative_error


def verify_bound_constraints(controls, lower_bound=None, upper_bound=None):
    """
    Check whether all control points satisfy the optional bounds.

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


# ============================================================
# Input validation
# ============================================================

def _validate_input(x_nodes, y_nodes, derivatives):
    """
    Basic validation of input data.
    """

    x_nodes = np.asarray(x_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)
    derivatives = np.asarray(derivatives, dtype=float)

    if len(x_nodes) != len(y_nodes):
        raise ValueError("x_nodes and y_nodes must have the same length")

    if len(derivatives) != len(x_nodes):
        raise ValueError("derivatives must have the same length as x_nodes")

    if len(x_nodes) < 2:
        raise ValueError("At least two support points are required")

    if np.any(np.diff(x_nodes) <= 0):
        raise ValueError("x_nodes must be strictly increasing")

    if not (
        np.isfinite(x_nodes).all()
        and np.isfinite(y_nodes).all()
        and np.isfinite(derivatives).all()
    ):
        raise ValueError("Input contains NaN or inf")

    return x_nodes, y_nodes, derivatives


def _validate_bounds(lower_bound, upper_bound):
    """
    Validate optional bounds.
    """

    lb = None if lower_bound is None else float(lower_bound)
    ub = None if upper_bound is None else float(upper_bound)

    if lb is not None and ub is not None and lb > ub:
        raise ValueError("lower_bound must not exceed upper_bound")

    return lb, ub


def _check_bounds_for_control_points(control_points, interval_index, lower_bound, upper_bound):
    """
    Check whether all control points satisfy the requested bounds.
    """

    if lower_bound is not None:
        if np.any(control_points < lower_bound):
            raise ValueError(
                f"Infeasible lower bound on interval {interval_index}: "
                f"exact Hermite-Bézier interpolation would violate the lower bound."
            )

    if upper_bound is not None:
        if np.any(control_points > upper_bound):
            raise ValueError(
                f"Infeasible upper bound on interval {interval_index}: "
                f"exact Hermite-Bézier interpolation would violate the upper bound."
            )
