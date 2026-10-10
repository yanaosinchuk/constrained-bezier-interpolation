import numpy as np
import pytest

from exact_quartic_bezier import (
    build_exact_quartic_bezier,
    verify_exact_quartic_bezier_constraints,
)
from bounded_quartic_bezier import (
    build_bounded_quartic_bezier,
    verify_bounded_quartic_bezier_constraints,
)


def test_exact_quartic_preserves_endpoints_means_and_derivatives():
    x_nodes = np.array([0.0, 1.0, 2.0])
    y_nodes = np.array([1.0, 2.0, 1.0])
    means = np.array([1.5, 1.5])

    controls, derivatives = build_exact_quartic_bezier(
        x_nodes=x_nodes,
        y_nodes=y_nodes,
        means=means,
        derivative_method="pchip",
        lower_bound=0.0,
        upper_bound=3.0,
        strict_bounds=True,
    )

    report = verify_exact_quartic_bezier_constraints(
        x_nodes=x_nodes,
        y_nodes=y_nodes,
        means=means,
        derivatives=derivatives,
        controls=controls,
        lower_bound=0.0,
        upper_bound=3.0,
    )

    assert report["max_point_error"] < 1e-12
    assert report["max_mean_error"] < 1e-12
    assert report["max_derivative_error"] < 1e-12
    assert report["max_lower_violation"] == 0.0
    assert report["max_upper_violation"] == 0.0


def test_bounded_quartic_preserves_hard_constraints_and_softens_derivatives():
    x_nodes = np.array([0.0, 1.0, 2.0])
    y_nodes = np.array([1.0, 2.0, 1.0])
    means = np.array([1.5, 1.5])
    preferred_derivatives = np.array([12.0, -12.0, 12.0])

    controls, derivatives = build_bounded_quartic_bezier(
        x_nodes=x_nodes,
        y_nodes=y_nodes,
        means=means,
        derivatives=preferred_derivatives,
        lower_bound=0.0,
        upper_bound=2.5,
    )

    report = verify_bounded_quartic_bezier_constraints(
        x_nodes=x_nodes,
        y_nodes=y_nodes,
        means=means,
        preferred_derivatives=derivatives,
        controls=controls,
        lower_bound=0.0,
        upper_bound=2.5,
    )

    assert report["max_point_error"] < 1e-12
    assert report["max_mean_error"] < 1e-12
    assert report["max_lower_violation"] == 0.0
    assert report["max_upper_violation"] == 0.0
    assert report["max_derivative_change"] > 0.0


def test_exact_quartic_strict_bounds_reject_infeasible_controls():
    x_nodes = np.array([0.0, 1.0])
    y_nodes = np.array([1.0, 1.0])
    means = np.array([1.0])
    derivatives = np.array([20.0, -20.0])

    with pytest.raises(ValueError, match="Infeasible"):
        build_exact_quartic_bezier(
            x_nodes=x_nodes,
            y_nodes=y_nodes,
            means=means,
            derivatives=derivatives,
            lower_bound=0.0,
            upper_bound=2.0,
            strict_bounds=True,
        )


def test_bounded_quartic_rejects_infeasible_mean_and_bounds():
    x_nodes = np.array([0.0, 1.0])
    y_nodes = np.array([0.0, 0.0])
    means = np.array([10.0])

    with pytest.raises(ValueError, match="Infeasible bounded quartic"):
        build_bounded_quartic_bezier(
            x_nodes=x_nodes,
            y_nodes=y_nodes,
            means=means,
            derivatives=np.array([0.0, 0.0]),
            lower_bound=0.0,
            upper_bound=1.0,
        )
