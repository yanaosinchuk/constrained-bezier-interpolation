import numpy as np
import pytest

from mean_preserving_cubic import (
    build_mean_preserving_cubic_bezier,
    verify_interpolation_constraints,
)


def test_cubic_preserves_endpoints_means_and_bounds():
    x_nodes = np.array([0.0, 1.0, 2.0])
    y_nodes = np.array([1.0, 2.0, 1.0])
    means = np.array([1.5, 1.5])

    controls = build_mean_preserving_cubic_bezier(
        x_nodes=x_nodes,
        y_nodes=y_nodes,
        means=means,
        lower_bound=0.0,
        upper_bound=3.0,
    )

    report = verify_interpolation_constraints(
        x_nodes=x_nodes,
        y_nodes=y_nodes,
        means=means,
        controls=controls,
        lower_bound=0.0,
        upper_bound=3.0,
    )

    assert report["max_point_error"] < 1e-12
    assert report["max_mean_error"] < 1e-12
    assert report["max_lower_violation"] == 0.0
    assert report["max_upper_violation"] == 0.0


def test_cubic_rejects_infeasible_mean_and_bounds():
    x_nodes = np.array([0.0, 1.0])
    y_nodes = np.array([0.0, 0.0])
    means = np.array([2.0])

    with pytest.raises(ValueError, match="No feasible bounded"):
        build_mean_preserving_cubic_bezier(
            x_nodes=x_nodes,
            y_nodes=y_nodes,
            means=means,
            lower_bound=0.0,
            upper_bound=1.0,
        )
