import numpy as np

# ============================================================
# Mean-preserving bounded cubic Bézier interpolation
# ============================================================


def build_mean_preserving_cubic_bezier(
    x_nodes,
    y_nodes,
    means,
    lower_bound=0.0,
    upper_bound=None,
):
    """
    Construct a piecewise cubic Bézier interpolant that preserves the
    prescribed mean on each interval and optionally satisfies physical bounds.

    Parameters
    ----------
    x_nodes : array-like, shape (n,)
        Strictly increasing support points.

    y_nodes : array-like, shape (n,)
        Function values at support points.

    means : array-like, shape (n-1,)
        Mean values on each interval [x_i, x_{i+1}].

    lower_bound : float or None, optional
        Lower physical bound. Default is 0.0. Set to None to disable.

    upper_bound : float or None, optional
        Upper physical bound. Set to None to disable.

    Returns
    -------
    controls : ndarray, shape (n-1, 4)
        Bézier control points for each interval.
    """

    x_nodes, y_nodes, means = _validate_input(x_nodes, y_nodes, means)
    lower_bound, upper_bound = _validate_bounds(lower_bound, upper_bound)
    _check_endpoint_bounds(y_nodes, lower_bound, upper_bound)

    controls = _compute_bezier_control_points(
        y_nodes,
        means,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )

    return controls


# ============================================================
# Core construction
# ============================================================


def _compute_bezier_control_points(y_nodes, means, lower_bound=None, upper_bound=None):
    """
    Compute cubic Bézier control points for each interval.

    For each interval [x_i, x_{i+1}]:

        P0 = y_i
        P3 = y_{i+1}

    The interval mean condition:
        (P0 + P1 + P2 + P3)/4 = mean_i

    Among all possible solutions, we choose the one minimizing:
        integral (B''(t))^2 dt

    This gives the unconstrained preferred control points:

        P1 = (2/3)*(y_i + y_{i+1}) - 2*mean_i
        P2 = 6*mean_i - (5/3)*(y_i + y_{i+1})

    If physical bounds are given, the pair (P1, P2) is projected onto the
    feasible set defined by:

        P1 + P2 = 4*mean_i - y_i - y_{i+1}
        lower_bound <= P1, P2 <= upper_bound

    This preserves the exact mean while keeping the full Bézier curve inside
    the same bounds, because a scalar Bézier curve stays in the convex hull of
    its control points.
    """

    n_intervals = len(y_nodes) - 1
    controls = np.zeros((n_intervals, 4), dtype=float)

    for i in range(n_intervals):
        y0 = y_nodes[i]
        y1 = y_nodes[i + 1]
        m = means[i]

        P0 = y0
        P3 = y1

        preferred_P1 = (2.0 / 3.0) * (y0 + y1) - 2.0 * m
        preferred_P2 = 6.0 * m - (5.0 / 3.0) * (y0 + y1)

        P1, P2 = _enforce_bounded_mean_pair(
            preferred_P1,
            preferred_P2,
            target_sum=4.0 * m - y0 - y1,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            interval_index=i,
        )

        controls[i] = [P0, P1, P2, P3]

    return controls


# ============================================================
# Helper for bounded exact-mean construction
# ============================================================


def _enforce_bounded_mean_pair(
    preferred_p1,
    preferred_p2,
    target_sum,
    lower_bound=None,
    upper_bound=None,
    interval_index=None,
):
    """
    Return the closest feasible pair (p1, p2) to the preferred pair, under
    the exact linear constraint p1 + p2 = target_sum and optional box bounds.
    """

    if lower_bound is None and upper_bound is None:
        return float(preferred_p1), float(preferred_p2)

    lower = -np.inf if lower_bound is None else float(lower_bound)
    upper = np.inf if upper_bound is None else float(upper_bound)

    feasible_low = max(lower, target_sum - upper)
    feasible_high = min(upper, target_sum - lower)

    if feasible_low > feasible_high:
        msg = (
            "No feasible bounded Bézier segment for interval "
            f"{interval_index}: exact mean preservation conflicts with the "
            f"requested bounds [{lower_bound}, {upper_bound}]."
        )
        raise ValueError(msg)

    # Projection of (preferred_p1, preferred_p2) onto the line p1+p2=target_sum:
    projected_p1 = 0.5 * (preferred_p1 - preferred_p2 + target_sum)
    p1 = min(max(projected_p1, feasible_low), feasible_high)
    p2 = target_sum - p1

    return float(p1), float(p2)


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
    t = np.asarray(t)

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
        Points per interval

    Returns
    -------
    x_dense, y_dense : ndarray
    """

    x_nodes = np.asarray(x_nodes)
    n_intervals = len(x_nodes) - 1

    x_all = []
    y_all = []

    for i in range(n_intervals):
        x0 = x_nodes[i]
        x1 = x_nodes[i + 1]

        if i < n_intervals - 1:
            t = np.linspace(0, 1, num_points, endpoint=False)
        else:
            t = np.linspace(0, 1, num_points)

        x = x0 + (x1 - x0) * t
        y = evaluate_bezier_segment(controls[i], t)

        x_all.append(x)
        y_all.append(y)

    return np.concatenate(x_all), np.concatenate(y_all)


# ============================================================
# Constraint verification
# ============================================================


def verify_interpolation_constraints(
    x_nodes,
    y_nodes,
    means,
    controls,
    lower_bound=None,
    upper_bound=None,
):
    """
    Check if constraints are satisfied:

        - endpoint interpolation
        - interval mean preservation
        - optional physical bounds
    """

    max_point_error = 0.0
    max_mean_error = 0.0
    max_lower_violation = 0.0
    max_upper_violation = 0.0

    for i in range(len(controls)):
        P = controls[i]

        err_left = abs(P[0] - y_nodes[i])
        err_right = abs(P[3] - y_nodes[i + 1])

        mean_val = np.sum(P) / 4.0
        err_mean = abs(mean_val - means[i])

        max_point_error = max(max_point_error, err_left, err_right)
        max_mean_error = max(max_mean_error, err_mean)

        if lower_bound is not None:
            max_lower_violation = max(max_lower_violation, np.max(lower_bound - P))
        if upper_bound is not None:
            max_upper_violation = max(max_upper_violation, np.max(P - upper_bound))

    return {
        "max_point_error": max_point_error,
        "max_mean_error": max_mean_error,
        "max_lower_violation": max(0.0, max_lower_violation),
        "max_upper_violation": max(0.0, max_upper_violation),
    }


# ============================================================
# Input validation
# ============================================================


def _validate_input(x_nodes, y_nodes, means):
    """
    Basic validation of input data.
    """

    x_nodes = np.asarray(x_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)
    means = np.asarray(means, dtype=float)

    if len(x_nodes) != len(y_nodes):
        raise ValueError("x_nodes and y_nodes must have same length")

    if len(means) != len(x_nodes) - 1:
        raise ValueError("means must have length n-1")

    if np.any(np.diff(x_nodes) <= 0):
        raise ValueError("x_nodes must be strictly increasing")

    if not (
        np.isfinite(x_nodes).all()
        and np.isfinite(y_nodes).all()
        and np.isfinite(means).all()
    ):
        raise ValueError("Input contains NaN or inf")

    return x_nodes, y_nodes, means



def _validate_bounds(lower_bound, upper_bound):
    if lower_bound is not None and not np.isfinite(lower_bound):
        raise ValueError("lower_bound must be finite or None")
    if upper_bound is not None and not np.isfinite(upper_bound):
        raise ValueError("upper_bound must be finite or None")
    if (
        lower_bound is not None
        and upper_bound is not None
        and lower_bound > upper_bound
    ):
        raise ValueError("lower_bound must not exceed upper_bound")
    return lower_bound, upper_bound



def _check_endpoint_bounds(y_nodes, lower_bound, upper_bound):
    if lower_bound is not None and np.any(y_nodes < lower_bound):
        raise ValueError(
            "Support values violate the requested lower_bound, so exact bounded "
            "interpolation is impossible."
        )
    if upper_bound is not None and np.any(y_nodes > upper_bound):
        raise ValueError(
            "Support values violate the requested upper_bound, so exact bounded "
            "interpolation is impossible."
        )

