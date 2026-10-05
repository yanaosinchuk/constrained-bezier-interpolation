import numpy as np


# ============================================================
# Bounded quartic Bézier interpolation with endpoint values,
# interval means and soft first derivatives
# ============================================================
#
# A quartic Bézier segment has five control points:
#
#     P0, P1, P2, P3, P4.
#
# We enforce exactly:
#
#     1) B_i(x_i)       = y_i,
#     2) B_i(x_{i+1})   = y_{i+1},
#     3) mean(B_i)      = mean_i,
#     4) optional lower and upper bounds on all control points.
#
# The endpoint derivatives are used as preferred values, not as
# hard constraints. This is necessary because endpoint values,
# exact mean and exact endpoint derivatives uniquely determine all
# five control points; therefore bounds may become impossible.
#
# Here, P1 and P3 are chosen as close as possible to the derivative-
# based control points while keeping P2 and all other controls inside
# the requested bounds.
# ============================================================


def build_bounded_quartic_bezier_interpolant(
    x_nodes,
    y_nodes,
    means,
    derivatives=None,
    derivative_method="pchip",
    lower_bound=0.0,
    upper_bound=None,
):
    """
    Main interface function.

    Construct a bounded piecewise quartic Bézier interpolant.

    Parameters
    ----------
    x_nodes : array-like, shape (n,)
        Strictly increasing support points.

    y_nodes : array-like, shape (n,)
        Function values at the support points.

    means : array-like, shape (n-1,)
        Mean values on each interval [x_i, x_{i+1}].

    derivatives : array-like, shape (n,), optional
        Preferred derivative values at the support points.
        If None, derivatives are estimated from x_nodes and y_nodes.

    derivative_method : str, default="pchip"
        Used only if derivatives is None.
        Supported:
            - "pchip"
            - "finite_difference"

    lower_bound : float or None, default=0.0
        Optional lower bound. Use None to disable.

    upper_bound : float or None, default=None
        Optional upper bound. Use None to disable.

    Returns
    -------
    controls : ndarray, shape (n-1, 5)
        Bézier control points [P0, P1, P2, P3, P4] for each interval.

    derivatives : ndarray, shape (n,)
        Preferred derivatives used to define the target control points.
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

    controls = _compute_bounded_quartic_control_points(
        x_nodes=x_nodes,
        y_nodes=y_nodes,
        means=means,
        derivatives=derivatives,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )

    return controls, derivatives


# ============================================================
# Core construction
# ============================================================

def _compute_bounded_quartic_control_points(
    x_nodes,
    y_nodes,
    means,
    derivatives,
    lower_bound,
    upper_bound,
):
    """
    Compute bounded quartic Bézier control points.

    On interval [x_i, x_{i+1}]:

        P0 = y_i
        P4 = y_{i+1}

    Exact mean condition:

        (P0 + P1 + P2 + P3 + P4) / 5 = mean_i

    Therefore:

        P1 + P2 + P3 = 5*mean_i - P0 - P4.

    Preferred derivative-based values are:

        P1_ref = P0 + h*d_i/4
        P3_ref = P4 - h*d_{i+1}/4.

    We choose P1 and P3 closest to (P1_ref, P3_ref), subject to:

        lower_bound <= P1, P2, P3 <= upper_bound=None,
        P2 = S - P1 - P3.

    This preserves endpoint values and means exactly and enforces bounds
    whenever the exact bounded problem is feasible.
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
        P4 = y1

        # Sum needed for the three internal control points.
        internal_sum = 5.0 * m - P0 - P4

        # Preferred derivative-based positions.
        P1_ref = P0 + (h / 4.0) * d0
        P3_ref = P4 - (h / 4.0) * d1

        P1, P3 = _project_p1_p3_to_bounded_mean_feasible_set(
            target_p1=P1_ref,
            target_p3=P3_ref,
            internal_sum=internal_sum,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            interval_index=i,
        )

        P2 = internal_sum - P1 - P3

        segment_controls = np.array([P0, P1, P2, P3, P4], dtype=float)

        # This should be zero if the projection was correct.
        _check_bounds_for_control_points(
            segment_controls,
            interval_index=i,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        controls[i] = segment_controls

    return controls


def _project_p1_p3_to_bounded_mean_feasible_set(
    target_p1,
    target_p3,
    internal_sum,
    lower_bound,
    upper_bound,
    interval_index,
):
    """
    Project (target_p1, target_p3) onto the convex feasible set:

        lower_bound <= P1 <= upper_bound
        lower_bound <= P3 <= upper_bound
        lower_bound <= internal_sum - P1 - P3 <= upper_bound

    This is a small two-dimensional convex projection problem.
    The implementation is explicit: it checks the target itself and
    projections onto all finite boundary lines.
    """

    lb = -np.inf if lower_bound is None else float(lower_bound)
    ub = np.inf if upper_bound is None else float(upper_bound)

    # Feasibility of P2 = internal_sum - P1 - P3 gives a strip constraint
    # on P1 + P3.
    sum_low = -np.inf if upper_bound is None else internal_sum - ub
    sum_high = np.inf if lower_bound is None else internal_sum - lb

    # Basic feasibility check:
    # There must exist P1, P3 in [lb, ub] with sum_low <= P1+P3 <= sum_high.
    min_possible_sum = 2.0 * lb
    max_possible_sum = 2.0 * ub

    feasible_sum_low = max(sum_low, min_possible_sum)
    feasible_sum_high = min(sum_high, max_possible_sum)

    if feasible_sum_low > feasible_sum_high:
        raise ValueError(
            f"Infeasible bounded quartic Bézier segment on interval {interval_index}: "
            f"endpoint values, exact mean and bounds cannot be satisfied simultaneously."
        )

    a = float(target_p1)
    b = float(target_p3)

    candidates = []

    def is_feasible(p1, p3):
        p2 = internal_sum - p1 - p3

        if p1 < lb - 1e-12 or p1 > ub + 1e-12:
            return False
        if p3 < lb - 1e-12 or p3 > ub + 1e-12:
            return False
        if p2 < lb - 1e-12 or p2 > ub + 1e-12:
            return False

        return True

    def add_candidate(p1, p3):
        p1 = float(p1)
        p3 = float(p3)

        if is_feasible(p1, p3):
            dist = (p1 - a) ** 2 + (p3 - b) ** 2
            candidates.append((dist, p1, p3))

    # Target itself.
    add_candidate(a, b)

    # Boundaries P1 = lb / ub.
    for fixed_p1 in [lb, ub]:
        if np.isfinite(fixed_p1):
            p3_low = lb
            p3_high = ub

            if np.isfinite(sum_low):
                p3_low = max(p3_low, sum_low - fixed_p1)
            if np.isfinite(sum_high):
                p3_high = min(p3_high, sum_high - fixed_p1)

            if p3_low <= p3_high:
                p3 = _clip(b, p3_low, p3_high)
                add_candidate(fixed_p1, p3)

    # Boundaries P3 = lb / ub.
    for fixed_p3 in [lb, ub]:
        if np.isfinite(fixed_p3):
            p1_low = lb
            p1_high = ub

            if np.isfinite(sum_low):
                p1_low = max(p1_low, sum_low - fixed_p3)
            if np.isfinite(sum_high):
                p1_high = min(p1_high, sum_high - fixed_p3)

            if p1_low <= p1_high:
                p1 = _clip(a, p1_low, p1_high)
                add_candidate(p1, fixed_p3)

    # Boundaries P1 + P3 = sum_low / sum_high.
    for fixed_sum in [sum_low, sum_high]:
        if np.isfinite(fixed_sum):
            # Projection of (a,b) onto P1+P3=fixed_sum.
            p1_star = 0.5 * (a - b + fixed_sum)

            p1_low = lb
            p1_high = ub

            # P3 = fixed_sum - P1 must also be in [lb, ub].
            p1_low = max(p1_low, fixed_sum - ub)
            p1_high = min(p1_high, fixed_sum - lb)

            if p1_low <= p1_high:
                p1 = _clip(p1_star, p1_low, p1_high)
                p3 = fixed_sum - p1
                add_candidate(p1, p3)

    if not candidates:
        raise ValueError(
            f"Projection failed on interval {interval_index}: "
            f"no feasible candidate found although the feasibility check passed."
        )

    candidates.sort(key=lambda item: item[0])
    _, best_p1, best_p3 = candidates[0]

    return best_p1, best_p3


def _clip(value, lower, upper):
    return min(max(float(value), float(lower)), float(upper))


# ============================================================
# Derivative estimation
# ============================================================

def estimate_node_derivatives(x_nodes, y_nodes, method="pchip"):
    """
    Estimate first derivatives at support points.
    """

    x_nodes, y_nodes = _validate_xy_nodes(x_nodes, y_nodes)

    if method == "pchip":
        return estimate_pchip_derivatives_numpy(x_nodes, y_nodes)

    if method == "finite_difference":
        return np.gradient(y_nodes, x_nodes)

    raise ValueError("Unknown derivative_method. Use 'pchip' or 'finite_difference'.")


def estimate_pchip_derivatives_numpy(x_nodes, y_nodes):
    """
    Shape-preserving PCHIP-style derivative estimates.
    """

    x, y = _validate_xy_nodes(x_nodes, y_nodes)
    n = len(x)

    if n == 2:
        slope = (y[1] - y[0]) / (x[1] - x[0])
        return np.array([slope, slope], dtype=float)

    h = np.diff(x)
    delta = np.diff(y) / h
    d = np.zeros(n, dtype=float)

    # Interior nodes.
    for i in range(1, n - 1):
        left = delta[i - 1]
        right = delta[i]

        if left == 0.0 or right == 0.0 or np.sign(left) != np.sign(right):
            d[i] = 0.0
        else:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            d[i] = (w1 + w2) / (w1 / left + w2 / right)

    # Endpoints.
    d[0] = _pchip_endpoint_slope(h[0], h[1], delta[0], delta[1])
    d[-1] = _pchip_endpoint_slope(h[-1], h[-2], delta[-1], delta[-2])

    return d


def _pchip_endpoint_slope(h0, h1, delta0, delta1):
    """
    One-sided endpoint slope used by PCHIP.
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
    Evaluate one quartic Bézier segment.
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
    Evaluate the full piecewise quartic Bézier interpolant.
    """

    x_nodes = np.asarray(x_nodes, dtype=float)
    controls = np.asarray(controls, dtype=float)

    if controls.shape != (len(x_nodes) - 1, 5):
        raise ValueError("controls must have shape (len(x_nodes)-1, 5)")

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
# Constraint verification
# ============================================================

def verify_bounded_quartic_bezier_constraints(
    x_nodes,
    y_nodes,
    means,
    preferred_derivatives,
    controls,
    lower_bound=None,
    upper_bound=None,
):
    """
    Verify exact constraints and diagnostics for derivative changes.
    """

    x_nodes, y_nodes, means = _validate_input(x_nodes, y_nodes, means)
    preferred_derivatives = _validate_derivatives(
        preferred_derivatives,
        expected_length=len(x_nodes),
    )
    controls = np.asarray(controls, dtype=float)

    if controls.shape != (len(x_nodes) - 1, 5):
        raise ValueError("controls must have shape (len(x_nodes)-1, 5)")

    max_point_error = 0.0
    max_mean_error = 0.0
    max_derivative_change = 0.0

    for i in range(len(controls)):
        P = controls[i]
        h = x_nodes[i + 1] - x_nodes[i]

        left_value = P[0]
        right_value = P[4]

        max_point_error = max(
            max_point_error,
            abs(left_value - y_nodes[i]),
            abs(right_value - y_nodes[i + 1]),
        )

        mean_value = float(np.mean(P))
        max_mean_error = max(max_mean_error, abs(mean_value - means[i]))

        actual_left_derivative = 4.0 * (P[1] - P[0]) / h
        actual_right_derivative = 4.0 * (P[4] - P[3]) / h

        max_derivative_change = max(
            max_derivative_change,
            abs(actual_left_derivative - preferred_derivatives[i]),
            abs(actual_right_derivative - preferred_derivatives[i + 1]),
        )

    max_lower_violation, max_upper_violation = verify_bound_constraints(
        controls,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )

    return {
        "max_point_error": float(max_point_error),
        "max_mean_error": float(max_mean_error),
        "max_derivative_change": float(max_derivative_change),
        "max_lower_violation": float(max_lower_violation),
        "max_upper_violation": float(max_upper_violation),
    }


def verify_bound_constraints(controls, lower_bound=None, upper_bound=None):
    """
    Check optional physical bounds for all control points.
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
    Simple roughness diagnostic based on squared second differences.
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
            "Support values violate lower_bound, so exact bounded interpolation is impossible."
        )

    if upper_bound is not None and np.any(y_nodes > upper_bound):
        raise ValueError(
            "Support values violate upper_bound, so exact bounded interpolation is impossible."
        )


def _check_bounds_for_control_points(control_points, interval_index, lower_bound, upper_bound):
    if lower_bound is not None and np.any(control_points < lower_bound - 1e-10):
        min_value = float(np.min(control_points))
        raise ValueError(
            f"Projection error on interval {interval_index}: "
            f"a control point is {min_value:.12g}, below lower_bound={lower_bound}."
        )

    if upper_bound is not None and np.any(control_points > upper_bound + 1e-10):
        max_value = float(np.max(control_points))
        raise ValueError(
            f"Projection error on interval {interval_index}: "
            f"a control point is {max_value:.12g}, above upper_bound={upper_bound}."
        )
