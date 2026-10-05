import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Project paths
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from mean_preserving_cubic import (
    build_mean_preserving_cubic_bezier,
    evaluate_piecewise_bezier as eval_local_bezier,
    verify_interpolation_constraints,
)

from balanced_mean_cubic import (
    build_balanced_bezier_interpolant,
    evaluate_piecewise_bezier as eval_balanced_bezier,
)

from hermite_cubic import (
    build_derivative_bezier_interpolant,
    evaluate_piecewise_bezier as eval_derivative_bezier,
    verify_derivative_interpolation_constraints,
    verify_bound_constraints,
)

from legacy_spline_methods import (
    hermite_vier_spline,
    poly,
)


# ============================================================
# Data utilities
# ============================================================


def load_data(path):
    df = pd.read_excel(path)
    df = df[["timestamp", "value"]].dropna().copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    t0 = df["timestamp"].iloc[0]
    df["x"] = (df["timestamp"] - t0).dt.total_seconds() / 3600.0

    return df["x"].to_numpy(dtype=float), df["value"].to_numpy(dtype=float)



def choose_points(x, y, step):
    idx = np.arange(0, len(x), step)
    if idx[-1] != len(x) - 1:
        idx = np.append(idx, len(x) - 1)
    return idx, x[idx], y[idx]



def compute_interval_means(y, idx):
    means = []
    for i in range(len(idx) - 1):
        a, b = idx[i], idx[i + 1]
        means.append(np.mean(y[a:b + 1]))
    return np.asarray(means, dtype=float)



def estimate_node_derivatives(x_nodes, y_nodes):
    """
    Finite-difference derivative estimate at support points.
    Uses np.gradient, which is central inside and one-sided at the ends.
    """
    return np.gradient(y_nodes, x_nodes)


# ============================================================
# Evaluation helpers
# ============================================================


def metrics(y_true, y_pred):
    res = y_true - y_pred
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = np.nan if sst == 0 else 1.0 - np.sum(res ** 2) / sst

    return {
        "RMSE": float(np.sqrt(np.mean(res ** 2))),
        "MAE": float(np.mean(np.abs(res))),
        "MAX": float(np.max(np.abs(res))),
        "R2": float(r2) if np.isfinite(r2) else np.nan,
        "res": res,
    }



def evaluate_quartic_piecewise(x_nodes, coeffs, num_points=400):
    """
    Evaluate the quartic piecewise spline from hermite_vier_spline.

    coeffs = (a, b, c, d, e)
    Each interval polynomial is
        a_i (x-x_i)^4 + b_i (x-x_i)^3 + c_i (x-x_i)^2 + d_i (x-x_i) + e_i
    """
    a, b, c, d, e = coeffs

    x_all = []
    y_all = []

    for i in range(len(x_nodes) - 1):
        x0 = x_nodes[i]
        x1 = x_nodes[i + 1]

        if i < len(x_nodes) - 2:
            x = np.linspace(x0, x1, num_points, endpoint=False)
        else:
            x = np.linspace(x0, x1, num_points)

        y = poly(x, x0, a[i], b[i], c[i], d[i], e[i])

        x_all.append(x)
        y_all.append(y)

    return np.concatenate(x_all), np.concatenate(y_all)


# ============================================================
# Pretty printing
# ============================================================


def print_metrics_table(results):
    print("\n" + "=" * 92)
    print("METHOD COMPARISON")
    print("=" * 92)
    header = f"{'Method':34s} {'RMSE':>12s} {'MAE':>12s} {'MAX':>12s} {'R2':>12s}"
    print(header)
    print("-" * len(header))

    for name, info in results.items():
        if "error" in info:
            print(f"{name:34s} ERROR: {info['error']}")
            continue

        m = info["metrics"]
        print(
            f"{name:34s} "
            f"{m['RMSE']:12.6f} "
            f"{m['MAE']:12.6f} "
            f"{m['MAX']:12.6f} "
            f"{m['R2']:12.6f}"
        )



def print_constraint_info(results):
    print("\n" + "=" * 92)
    print("CONSTRAINT CHECKS")
    print("=" * 92)

    for name, info in results.items():
        print(f"\n{name}:")
        if "error" in info:
            print(f"  ERROR: {info['error']}")
            continue

        constraints = info.get("constraints")
        if constraints is None:
            print("  No dedicated constraint report.")
            continue

        for key, value in constraints.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6e}")
            else:
                print(f"  {key}: {value}")


# ============================================================
# Main experiment
# ============================================================


def main():
    data_path = ROOT / "data" / "pegelonline_leunneu_2024.xlsx"

    # -------- settings --------
    step = 672               # support-point spacing in samples
    num_points = 500         # dense evaluation points per interval
    lower_bound = 0.0        # physical lower bound
    upper_bound = None       # set a number if needed
    # -------------------------

    x_ref, y_ref = load_data(data_path)
    idx, x_nodes, y_nodes = choose_points(x_ref, y_ref, step)
    means = compute_interval_means(y_ref, idx)
    derivatives = estimate_node_derivatives(x_nodes, y_nodes)

    print(f"Loaded {len(x_ref)} reference points")
    print(f"Using {len(x_nodes)} support points and {len(means)} intervals")
    print(f"Bounds: lower_bound={lower_bound}, upper_bound={upper_bound}")

    results = {}

    # ========================================================
    # 1) Original local cubic Bézier with exact mean condition
    # ========================================================
    try:
        controls = build_mean_preserving_cubic_bezier(
            x_nodes,
            y_nodes,
            means,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        x_plot, y_plot = eval_local_bezier(x_nodes, controls, num_points=num_points)
        y_true = np.interp(x_plot, x_ref, y_ref)

        results["Original cubic Bézier (mean)"] = {
            "x": x_plot,
            "y": y_plot,
            "metrics": metrics(y_true, y_plot),
            "constraints": verify_interpolation_constraints(
                x_nodes,
                y_nodes,
                means,
                controls,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            ),
        }
    except Exception as e:
        results["Original cubic Bézier (mean)"] = {"error": str(e)}

    # ========================================================
    # 2) Balanced cubic Bézier with exact mean condition
    # ========================================================
    try:
        controls = build_balanced_bezier_interpolant(
            x_nodes,
            y_nodes,
            means,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        x_plot, y_plot = eval_balanced_bezier(x_nodes, controls, num_points=num_points)
        y_true = np.interp(x_plot, x_ref, y_ref)

        results["Balanced cubic Bézier (mean)"] = {
            "x": x_plot,
            "y": y_plot,
            "metrics": metrics(y_true, y_plot),
            "constraints": verify_interpolation_constraints(
                x_nodes,
                y_nodes,
                means,
                controls,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            ),
        }
    except Exception as e:
        results["Balanced cubic Bézier (mean)"] = {"error": str(e)}

    # ========================================================
    # 3) Cubic Bézier with derivative interpolation
    # ========================================================
    try:
        controls = build_derivative_bezier_interpolant(
            x_nodes,
            y_nodes,
            derivatives,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        x_plot, y_plot = eval_derivative_bezier(x_nodes, controls, num_points=num_points)
        y_true = np.interp(x_plot, x_ref, y_ref)

        point_err, deriv_err = verify_derivative_interpolation_constraints(
            x_nodes,
            y_nodes,
            derivatives,
            controls,
        )
        low_vio, up_vio = verify_bound_constraints(
            controls,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        results["Cubic Bézier (derivatives)"] = {
            "x": x_plot,
            "y": y_plot,
            "metrics": metrics(y_true, y_plot),
            "constraints": {
                "max_point_error": point_err,
                "max_derivative_error": deriv_err,
                "max_lower_violation": low_vio,
                "max_upper_violation": up_vio,
            },
        }
    except Exception as e:
        results["Cubic Bézier (derivatives)"] = {"error": str(e)}

    # ========================================================
    # 4) Previous attempt: quartic Hermite mean spline
    # ========================================================
    try:
        coeffs = hermite_vier_spline(x_nodes, y_nodes, derivatives, means)
        x_plot, y_plot = evaluate_quartic_piecewise(x_nodes, coeffs[:5], num_points=num_points)
        y_true = np.interp(x_plot, x_ref, y_ref)

        results["Previous attempt: quartic Hermite-mean"] = {
            "x": x_plot,
            "y": y_plot,
            "metrics": metrics(y_true, y_plot),
            "constraints": None,
        }
    except Exception as e:
        results["Previous attempt: quartic Hermite-mean"] = {"error": str(e)}

    print_metrics_table(results)
    print_constraint_info(results)

    # ========================================================
    # Plot 1: full comparison
    # ========================================================
    plt.figure(figsize=(14, 7))
    plt.plot(x_ref, y_ref, label="Reference data", linewidth=1.6, alpha=0.75)

    for name, info in results.items():
        if "error" in info:
            continue
        plt.plot(info["x"], info["y"], label=name, linewidth=1.8)

    plt.scatter(x_nodes, y_nodes, s=18, marker="o", label="Support points")
    plt.xlabel("Time since start [hours]")
    plt.ylabel("Water level")
    plt.title("Visible comparison of all interpolation methods")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # ========================================================
    # Plot 2: residuals against reference
    # ========================================================
    plt.figure(figsize=(14, 6))
    for name, info in results.items():
        if "error" in info:
            continue
        plt.plot(info["x"], info["metrics"]["res"], label=name, linewidth=1.4)

    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    plt.xlabel("Time since start [hours]")
    plt.ylabel("Reference - interpolant")
    plt.title("Residual comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # ========================================================
    # Plot 3: zoom into the first 3 intervals
    # ========================================================
    if len(x_nodes) >= 4:
        zoom_end = x_nodes[3]
    else:
        zoom_end = x_nodes[-1]

    plt.figure(figsize=(14, 6))
    mask_ref = x_ref <= zoom_end
    plt.plot(x_ref[mask_ref], y_ref[mask_ref], label="Reference data", linewidth=1.8, alpha=0.8)

    for name, info in results.items():
        if "error" in info:
            continue
        mask = info["x"] <= zoom_end
        plt.plot(info["x"][mask], info["y"][mask], label=name, linewidth=1.8)

    mask_nodes = x_nodes <= zoom_end
    plt.scatter(x_nodes[mask_nodes], y_nodes[mask_nodes], s=22, marker="o", label="Support points")
    plt.xlabel("Time since start [hours]")
    plt.ylabel("Water level")
    plt.title("Zoom: first intervals")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
