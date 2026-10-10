import numpy as np
import pytest

import common
import exact_quartic_bezier
import bounded_quartic_bezier


def test_shared_quartic_utilities_are_reused():
    assert (
        exact_quartic_bezier.evaluate_piecewise_quartic_bezier
        is common.evaluate_piecewise_quartic_bezier
    )
    assert (
        bounded_quartic_bezier.evaluate_piecewise_quartic_bezier
        is common.evaluate_piecewise_quartic_bezier
    )
    assert (
        exact_quartic_bezier.estimate_node_derivatives
        is common.estimate_node_derivatives
    )
    assert (
        bounded_quartic_bezier.estimate_node_derivatives
        is common.estimate_node_derivatives
    )


def test_pchip_derivatives_are_exact_for_linear_data():
    x = np.array([0.0, 1.0, 2.0, 4.0])
    y = 3.0 * x - 2.0

    derivatives = common.estimate_node_derivatives(x, y, method="pchip")

    np.testing.assert_allclose(derivatives, 3.0, atol=1e-12, rtol=0.0)


def test_piecewise_quartic_evaluation_hits_global_endpoints():
    x_nodes = np.array([0.0, 1.0, 2.0])
    controls = np.array(
        [
            [1.0, 1.1, 1.2, 1.4, 1.5],
            [1.5, 1.6, 1.7, 1.8, 2.0],
        ]
    )

    x_dense, y_dense = common.evaluate_piecewise_quartic_bezier(
        x_nodes,
        controls,
        num_points=11,
    )

    assert x_dense[0] == pytest.approx(x_nodes[0])
    assert x_dense[-1] == pytest.approx(x_nodes[-1])
    assert y_dense[0] == pytest.approx(controls[0, 0])
    assert y_dense[-1] == pytest.approx(controls[-1, -1])


def test_piecewise_quartic_evaluation_rejects_too_few_points():
    x_nodes = np.array([0.0, 1.0])
    controls = np.array([[1.0, 1.0, 1.0, 1.0, 1.0]])

    with pytest.raises(ValueError, match="num_points must be at least 2"):
        common.evaluate_piecewise_quartic_bezier(
            x_nodes,
            controls,
            num_points=1,
        )
