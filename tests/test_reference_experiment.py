import numpy as np

from compare_interpolation_methods import main


def test_reference_experiment_matches_expected_metrics():
    results = main(show=False)

    expected = {
        "Cubic Bézier (mean, bounded)": {
            "RMSE": 12.007279,
            "MAE": 5.330826,
            "MAX": 113.560781,
            "R2": 0.921061,
        },
        "Exact quartic Bézier (diagnostic)": {
            "RMSE": 11.242047,
            "MAE": 5.560609,
            "MAX": 94.711293,
            "R2": 0.930802,
        },
        "Bounded quartic Bézier (soft derivatives)": {
            "RMSE": 10.947902,
            "MAE": 5.470358,
            "MAX": 94.711293,
            "R2": 0.934376,
        },
    }

    for method, metrics in expected.items():
        assert method in results
        assert "error" not in results[method]

        actual = results[method]["metrics"]
        for key, expected_value in metrics.items():
            assert np.isclose(
                actual[key],
                expected_value,
                atol=1e-6,
                rtol=0.0,
            ), f"{method} {key}: expected {expected_value}, got {actual[key]}"
