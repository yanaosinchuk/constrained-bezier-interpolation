import numpy as np

from common import (
    _check_bounds_for_control_points,
    _check_endpoint_bounds,
    _validate_bounds,
    _validate_derivatives,
    _validate_input,
    compute_control_polygon_roughness,
    estimate_node_derivatives,
    evaluate_piecewise_quartic_bezier,
    evaluate_quartic_bezier_segment,
    verify_bound_constraints,
)

# ============================================================
# Exact quartic Bézier interpolation with endpoint values,
# interval means, and endpoint derivatives
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
#         using verify_exact_quartic_bezier_constraints(...).
#
# ============================================================


# ============================================================
# Main interface
# ============================================================

def build_exact_quartic_bezier(
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
    Construct a piecewise quartic Bézier interpolant that exactly matches
    the endpoint values, prescribed interval means, and endpoint derivatives.

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
# Exact-method evaluation helpers
# ============================================================

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

def verify_exact_quartic_bezier_constraints(
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


# ============================================================
# Exact-method diagnostics
# ============================================================

def compute_interval_mean_from_controls(P):
    """
    Compute the exact interval mean of one quartic Bézier segment.

    For a Bézier curve of any degree, the average over t in [0,1]
    equals the arithmetic mean of the control points.
    """

    P = np.asarray(P, dtype=float)
    return float(np.mean(P))
