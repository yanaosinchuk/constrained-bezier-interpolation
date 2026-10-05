import numpy as np

from mean_preserving_cubic import (
    _check_endpoint_bounds,
    _enforce_bounded_mean_pair,
    _validate_bounds,
    _validate_input,
)


# ============================================================
# Balanced cubic Bézier interpolation
# ============================================================


def build_balanced_bezier_interpolant(
    x_nodes,
    y_nodes,
    means,
    lower_bound=0.0,
    upper_bound=None,
):
    x_nodes, y_nodes, means = _validate_input(x_nodes, y_nodes, means)
    lower_bound, upper_bound = _validate_bounds(lower_bound, upper_bound)
    _check_endpoint_bounds(y_nodes, lower_bound, upper_bound)

    n_intervals = len(y_nodes) - 1
    controls = np.zeros((n_intervals, 4), dtype=float)

    for i in range(n_intervals):
        y0 = y_nodes[i]
        y1 = y_nodes[i + 1]
        m = means[i]

        P0 = y0
        P3 = y1

        middle = 0.5 * (4.0 * m - y0 - y1)
        P1, P2 = _enforce_bounded_mean_pair(
            middle,
            middle,
            target_sum=4.0 * m - y0 - y1,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            interval_index=i,
        )

        controls[i] = [P0, P1, P2, P3]

    return controls



def evaluate_bezier_segment(P, t):
    P0, P1, P2, P3 = P
    t = np.asarray(t)

    return (
        P0 * (1 - t) ** 3
        + 3 * P1 * (1 - t) ** 2 * t
        + 3 * P2 * (1 - t) * t ** 2
        + P3 * t ** 3
    )



def evaluate_piecewise_bezier(x_nodes, controls, num_points=100):
    x_nodes = np.asarray(x_nodes, dtype=float)

    x_all = []
    y_all = []

    for i in range(len(x_nodes) - 1):
        x0 = x_nodes[i]
        x1 = x_nodes[i + 1]

        if i < len(x_nodes) - 2:
            t = np.linspace(0, 1, num_points, endpoint=False)
        else:
            t = np.linspace(0, 1, num_points)

        x = x0 + (x1 - x0) * t
        y = evaluate_bezier_segment(controls[i], t)

        x_all.append(x)
        y_all.append(y)

    return np.concatenate(x_all), np.concatenate(y_all)
