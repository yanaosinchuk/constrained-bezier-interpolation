import numpy as np

# ============================================================
# Quartic Bézier interpolation with endpoint values,
# interval means and first derivatives
# ============================================================
#
# A quartic Bézier segment has five control points:
#
#     P0, P1, P2, P3, P4.
#
# Therefore we can enforce exactly:
#
#     1) B_i(x_i)       = y_i,
#     2) B_i(x_{i+1})   = y_{i+1},
#     3) mean(B_i)      = mean_i,
#     4) B_i'(x_i)      = d_i,
#     5) B_i'(x_{i+1})  = d_{i+1}
#
# Important
# ---------
# If all five constraints are required exactly, then the five control
# points are uniquely determined. Therefore optional physical bounds
# cannot always be enforced by projection without destroying either the
# derivative condition or the exact mean condition.
#
# This implementation therefore supports two modes:
#
#     strict_bounds=True
#         Raise an error if a segment would violate the bounds.
#
#     strict_bounds=False
#         Build the interpolant anyway and report bound violations later
#         using verify_quartic_bezier_constraints(...).
#
# ============================================================


# ============================================================
# Main interface
# ============================================================


def build_quartic_bezier_interpolant(
    x_nodes,
    y_nodes,
    means,
    derivatives=None,
    derivative_method="pchip",
    lower_bound=0.0,
    upper_bound=None,
    strict_bounds=True,
):
    """
    Construct a piecewise quartic Bézier interpolant.

    Each interval [x_i, x_{i+1}] is represented by one degree-4
    Bézier curve with five scalar control points:

        P0, P1, P2, P3, P4.

    The construction enforces the following conditions exactly:

        B_i(x_i)        = y_i
        B_i(x_{i+1})    = y_{i+1}
        mean(B_i)       = means[i]
        B_i'(x_i)       = derivatives[i]
        B_i'(x_{i+1})   = derivatives[i+1]

    Parameters
    ----------
    x_nodes : array-like, shape (n,)
        Strictly increasing support points.

    y_nodes : array-like, shape (n,)
        Function values at the support points.

    means : array-like, shape (n-1,)
        Prescribed mean value on each interval [x_i, x_{i+1}].

    derivatives : array-like, shape (n,), optional
        Derivatives at the support points.
        If None, derivatives are estimated from x_nodes and y_nodes.

    derivative_method : str, optional
        Method used when derivatives is None.

        Supported values:
            "pchip"
                Shape-preserving PCHIP-style.
                This is usually the safest default for measured data.

            "finite_difference"
                Standard finite differences via numpy.gradient.
                Smoother in some cases, but more likely to overshoot.

    lower_bound : float or None, optional

    upper_bound : float or None, optional

    strict_bounds : bool, optional

    Returns
    -------
    controls : ndarray, shape (n-1, 5)
        Bézier control points for every interval.
        Row i contains [P0, P1, P2, P3, P4].

    derivatives : ndarray, shape (n,)
    """

    x_nodes, y_nodes, means = _validate_input(x_nodes, y_nodes, means)
    lower_bound, upper_bound = _validate_bounds(lower_bound, upper_bound)
    _check_endpoint_bounds(y_nodes, lower_bound, upper_bound)

    if derivatives is None:
        derivatives = estimate_node_derivatives(
            x_nodes,
            y_nodes,
            method=derivative_method,
        )
    else:
        derivatives = _validate_derivatives(derivatives, expected_length=len(x_nodes))

    controls = _compute_quartic_bezier_control_points(
        x_nodes=x_nodes,
        y_nodes=y_nodes,
        means=means,
        derivatives=derivatives,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        strict_bounds=strict_bounds,
    )

    return controls, derivatives


# ============================================================
# Core construction
# ============================================================


def _compute_quartic_bezier_control_points(
    x_nodes,
    y_nodes,
    means,
    derivatives,
    lower_bound,
    upper_bound,
    strict_bounds,
):
    """
    Compute the degree-4 Bézier control points explicitly.

    On interval [x_i, x_{i+1}] let

        h  = x_{i+1} - x_i,
        y0 = y_i,
        y1 = y_{i+1},
        d0 = derivative at x_i,
        d1 = derivative at x_{i+1},
        m  = prescribed interval mean.

    A quartic Bézier curve is

        B(t) = sum_{k=0}^4 C(4,k) (1-t)^(4-k) t^k P_k,
        t in [0,1].

    Endpoint values give

        P0 = y0,
        P4 = y1.

    Endpoint derivatives with respect to x give

        B'(x_i)     = 4(P1 - P0) / h = d0,
        B'(x_{i+1}) = 4(P4 - P3) / h = d1.

    Therefore

        P1 = P0 + h*d0/4,
        P3 = P4 - h*d1/4.

    The mean of a Bézier curve over t in [0,1] equals the average of
    its control points:

        integral_0^1 B(t) dt = (P0 + P1 + P2 + P3 + P4) / 5.

    Since x = x_i + h t, the average over x is the same number.
    Hence the middle control point is

        P2 = 5*m - P0 - P1 - P3 - P4.

    This enforces endpoint values, endpoint derivatives and interval
    mean exactly.
    """

    n_intervals = len(x_nodes) - 1
    controls = np.zeros((n_intervals, 5), dtype=float)

    for i in range(n_intervals):
        x0 = x_nodes[i]
        x1 = x_nodes[i + 1]
        h = x1 - x0

        y0 = y_nodes[i]
        y1 = y_nodes[i + 1]
        m = means[i]

        d0 = derivatives[i]
        d1 = derivatives[i + 1]

        P0 = y0
        P1 = y0 + (h / 4.0) * d0
        P3 = y1 - (h / 4.0) * d1
        P4 = y1
        P2 = 5.0 * m - P0 - P1 - P3 - P4

        segment_controls = np.array([P0, P1, P2, P3, P4], dtype=float)

        if strict_bounds:
            _check_bounds_for_control_points(
                segment_controls,
                interval_index=i,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )

        controls[i] = segment_controls

    return controls


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
# Evaluation functions
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



def evaluate_quartic_bezier_derivative_segment(P, h, t):
    """
    The derivative of a degree-4 Bézier curve with respect to t is a
    degree-3 Bézier curve:

        dB/dt = 4 * sum_{k=0}^3 C(3,k)(1-t)^(3-k)t^k(P_{k+1}-P_k).

    Since x = x_i + h*t, we have

        dB/dx = (dB/dt) / h.
    """

    P = np.asarray(P, dtype=float)
    if P.shape != (5,):
        raise ValueError("P must have shape (5,)")

    if h <= 0:
        raise ValueError("h must be positive")

    t = np.asarray(t, dtype=float)
    one_minus_t = 1.0 - t

    Q0 = P[1] - P[0]
    Q1 = P[2] - P[1]
    Q2 = P[3] - P[2]
    Q3 = P[4] - P[3]

    derivative_t = 4.0 * (
        Q0 * one_minus_t ** 3
        + 3.0 * Q1 * one_minus_t ** 2 * t
        + 3.0 * Q2 * one_minus_t * t ** 2
        + Q3 * t ** 3
    )

    return derivative_t / h



def evaluate_piecewise_quartic_bezier(x_nodes, controls, num_points=100):
    """
    Evaluate the full piecewise quartic Bézier interpolant on a dense grid.

    Parameters
    ----------
    x_nodes : array-like, shape (n,)
        Support points.

    controls : ndarray, shape (n-1, 5)
        Control points returned by build_quartic_bezier_interpolant(...).

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



def evaluate_quartic_bezier_at(x_nodes, controls, x_eval):
    """
    Returns
    -------
    y_eval : float or ndarray
        Interpolated value(s). Scalar input gives scalar output.
    """

    x_nodes = np.asarray(x_nodes, dtype=float)
    controls = np.asarray(controls, dtype=float)
    x_eval_array = np.asarray(x_eval, dtype=float)
    scalar_input = x_eval_array.ndim == 0
    x_flat = np.atleast_1d(x_eval_array)

    if np.any(x_flat < x_nodes[0]) or np.any(x_flat > x_nodes[-1]):
        raise ValueError("All x_eval values must lie inside the interpolation domain")

    interval_indices = np.searchsorted(x_nodes, x_flat, side="right") - 1
    interval_indices = np.clip(interval_indices, 0, len(x_nodes) - 2)

    y_flat = np.zeros_like(x_flat, dtype=float)

    for i in range(len(x_nodes) - 1):
        mask = interval_indices == i
        if not np.any(mask):
            continue

        h = x_nodes[i + 1] - x_nodes[i]
        t = (x_flat[mask] - x_nodes[i]) / h
        y_flat[mask] = evaluate_quartic_bezier_segment(controls[i], t)

    if scalar_input:
        return float(y_flat[0])

    return y_flat.reshape(x_eval_array.shape)


# ============================================================
# Constraint verification
# ============================================================


def verify_quartic_bezier_constraints(
    x_nodes,
    y_nodes,
    means,
    derivatives,
    controls,
    lower_bound=None,
    upper_bound=None,
):
    """
    Verify all exact interpolation constraints and optional bounds.

    Checks:

        - left and right endpoint values,
        - interval means,
        - left and right endpoint derivatives,
        - lower and upper bound violations of control points.

    Because Bézier curves lie inside the convex hull of their control
    points, bounded control points imply a bounded curve.

    Returns
    -------
    report : dict
        Dictionary with maximum errors/violations.
    """

    x_nodes, y_nodes, means = _validate_input(x_nodes, y_nodes, means)
    derivatives = _validate_derivatives(derivatives, expected_length=len(x_nodes))
    controls = np.asarray(controls, dtype=float)

    if controls.shape != (len(x_nodes) - 1, 5):
        raise ValueError("controls must have shape (len(x_nodes)-1, 5)")

    max_point_error = 0.0
    max_mean_error = 0.0
    max_derivative_error = 0.0

    for i in range(len(controls)):
        P = controls[i]
        h = x_nodes[i + 1] - x_nodes[i]

        # Endpoint values
        left_value = P[0]
        right_value = P[4]

        max_point_error = max(
            max_point_error,
            abs(left_value - y_nodes[i]),
            abs(right_value - y_nodes[i + 1]),
        )

        # Mean value of a degree-4 Bézier curve
        mean_value = np.mean(P)
        max_mean_error = max(max_mean_error, abs(mean_value - means[i]))

        # Endpoint derivatives with respect to x
        left_derivative = 4.0 * (P[1] - P[0]) / h
        right_derivative = 4.0 * (P[4] - P[3]) / h

        max_derivative_error = max(
            max_derivative_error,
            abs(left_derivative - derivatives[i]),
            abs(right_derivative - derivatives[i + 1]),
        )

    max_lower_violation, max_upper_violation = verify_bound_constraints(
        controls,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )

    return {
        "max_point_error": float(max_point_error),
        "max_mean_error": float(max_mean_error),
        "max_derivative_error": float(max_derivative_error),
        "max_lower_violation": float(max_lower_violation),
        "max_upper_violation": float(max_upper_violation),
    }



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


# ============================================================
# Optional diagnostics
# ============================================================


def compute_interval_mean_from_controls(P):
    """
    Compute the exact interval mean of one quartic Bézier segment.

    For a Bézier curve of any degree, the average over t in [0,1]
    equals the arithmetic mean of the control points.
    """

    P = np.asarray(P, dtype=float)
    return float(np.mean(P))



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



def _check_bounds_for_control_points(control_points, interval_index, lower_bound, upper_bound):

    if lower_bound is not None and np.any(control_points < lower_bound):
        min_value = float(np.min(control_points))
        raise ValueError(
            f"Infeasible lower bound on interval {interval_index}: "
            f"a control point is {min_value:.12g}, below lower_bound={lower_bound}. "
            "With exact endpoint values, exact mean and exact derivatives, "
            "the quartic Bézier control points are uniquely determined. "
            "Try strict_bounds=False for diagnostics or use smaller/slower "
            "derivatives."
        )

    if upper_bound is not None and np.any(control_points > upper_bound):
        max_value = float(np.max(control_points))
        raise ValueError(
            f"Infeasible upper bound on interval {interval_index}: "
            f"a control point is {max_value:.12g}, above upper_bound={upper_bound}. "
            "With exact endpoint values, exact mean and exact derivatives, "
            "the quartic Bézier control points are uniquely determined. "
            "Try strict_bounds=False for diagnostics or use smaller/slower "
            "derivatives."
        )
