import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from mean_preserving_cubic import (
    build_mean_preserving_cubic_bezier,
    evaluate_piecewise_bezier as evaluate_cubic_bezier,
    verify_interpolation_constraints as verify_cubic_constraints,
)

from exact_quartic_bezier import (
    build_quartic_bezier_interpolant,
    evaluate_piecewise_quartic_bezier as evaluate_exact_quartic_bezier,
    verify_quartic_bezier_constraints,
)

from bounded_quartic_bezier import (
    build_bounded_quartic_bezier_interpolant,
    evaluate_piecewise_quartic_bezier as evaluate_bounded_quartic_bezier,
    verify_bounded_quartic_bezier_constraints,
    compute_control_polygon_roughness,
)


# ============================================================
# Data utilities
# ============================================================

def load_reference_data(excel_path):
    df = pd.read_excel(excel_path)

    df = df[["timestamp", "value"]].dropna().copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp")

    t0 = df["timestamp"].iloc[0]
    df["x_hours"] = (df["timestamp"] - t0).dt.total_seconds() / 3600.0

    return df["x_hours"].to_numpy(dtype=float), df["value"].to_numpy(dtype=float)


def choose_support_points(x_ref, y_ref, step):
    idx = np.arange(0, len(x_ref), step)

    if idx[-1] != len(x_ref) - 1:
        idx = np.append(idx, len(x_ref) - 1)

    return idx, x_ref[idx], y_ref[idx]


def compute_interval_means_from_reference(y_ref, idx):
    means = []

    for i in range(len(idx) - 1):
        a = idx[i]
        b = idx[i + 1]
        means.append(np.mean(y_ref[a:b + 1]))

    return np.asarray(means, dtype=float)


# ============================================================
# Error analysis
# ============================================================

def compute_error_metrics(y_true, y_pred):
    residuals = y_true - y_pred

    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    max_abs_error = float(np.max(np.abs(residuals)))

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = np.nan if ss_tot == 0 else float(1.0 - ss_res / ss_tot)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAX": max_abs_error,
        "R2": r2,
        "residuals": residuals,
    }


def print_metrics_table(results):
    print()
    print("=" * 96)
    print("APPROXIMATION QUALITY")
    print("=" * 96)

    header = f"{'Method':42s} {'RMSE':>12s} {'MAE':>12s} {'MAX':>12s} {'R2':>12s}"
    print(header)
    print("-" * len(header))

    for name, info in results.items():
        if "error" in info:
            print(f"{name:42s} ERROR: {info['error']}")
            continue

        m = info["metrics"]
        print(
            f"{name:42s} "
            f"{m['RMSE']:12.6f} "
            f"{m['MAE']:12.6f} "
            f"{m['MAX']:12.6f} "
            f"{m['R2']:12.6f}"
        )


def print_constraint_report(results):
    print()
    print("=" * 96)
    print("CONSTRAINT CHECKS")
    print("=" * 96)

    for name, info in results.items():
        print()
        print(name)
        print("-" * len(name))

        if "error" in info:
            print(f"ERROR: {info['error']}")
            continue

        for key, value in info["constraints"].items():
            print(f"{key}: {value:.6e}")


# ============================================================
# Plotting
# ============================================================

def plot_comparison(x_ref, y_ref, x_nodes, y_nodes, results):
    plt.figure(figsize=(14, 7))
    plt.plot(x_ref, y_ref, label="Reference data", linewidth=1.3, alpha=0.75)

    for name, info in results.items():
        if "error" in info:
            continue

        plt.plot(info["x"], info["y"], label=name, linewidth=2.0)

    plt.scatter(x_nodes, y_nodes, label="Support points", s=25, zorder=3)

    plt.xlabel("Time since start [hours]")
    plt.ylabel("Value")
    plt.title("Cubic mean Bézier vs quartic Bézier variants")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()


def plot_residuals(results):
    plt.figure(figsize=(14, 6))

    for name, info in results.items():
        if "error" in info:
            continue

        plt.plot(
            info["x"],
            info["metrics"]["residuals"],
            label=f"Residual: {name}",
            linewidth=1.5,
        )

    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.0)

    plt.xlabel("Time since start [hours]")
    plt.ylabel("Residual = reference - approximation")
    plt.title("Residual comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()


# ============================================================
# Main experiment
# ============================================================

def main():
    data_path = ROOT / "data" / "pegelonline_leunneu_2024.xlsx"

    step = 672
    num_points = 500

    lower_bound = 0.0
    upper_bound = None
    derivative_method = "pchip" # Alternative: "finite_difference"

    x_ref, y_ref = load_reference_data(data_path)
    idx, x_nodes, y_nodes = choose_support_points(x_ref, y_ref, step)
    means = compute_interval_means_from_reference(y_ref, idx)

    print(f"Loaded reference points: {len(x_ref)}")
    print(f"Support points:          {len(x_nodes)}")
    print(f"Intervals:               {len(means)}")
    print(f"step:                    {step}")
    print(f"lower_bound:             {lower_bound}")
    print(f"upper_bound:             {upper_bound}")
    print(f"quartic derivative:      {derivative_method}")

    results = {}

    # ========================================================
    # 1) Cubic Bézier with exact means and bounds
    # ========================================================
    name = "Cubic Bézier (mean, bounded)"

    try:
        controls = build_local_bezier_interpolant(
            x_nodes=x_nodes,
            y_nodes=y_nodes,
            means=means,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        x_plot, y_plot = evaluate_cubic_bezier(
            x_nodes=x_nodes,
            controls=controls,
            num_points=num_points,
        )

        y_true = np.interp(x_plot, x_ref, y_ref)

        results[name] = {
            "x": x_plot,
            "y": y_plot,
            "metrics": compute_error_metrics(y_true, y_plot),
            "constraints": verify_cubic_constraints(
                x_nodes=x_nodes,
                y_nodes=y_nodes,
                means=means,
                controls=controls,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            ),
            "controls": controls,
        }

    except Exception as e:
        results[name] = {"error": str(e)}

    # ========================================================
    # 2) Exact quartic Bézier with derivatives, only diagnostic
    # ========================================================
    name = "Exact quartic Bézier (diagnostic)"

    try:
        controls, derivatives = build_quartic_bezier_interpolant(
            x_nodes=x_nodes,
            y_nodes=y_nodes,
            means=means,
            derivatives=None,
            derivative_method=derivative_method,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            strict_bounds=False,
        )

        x_plot, y_plot = evaluate_exact_quartic_bezier(
            x_nodes=x_nodes,
            controls=controls,
            num_points=num_points,
        )

        y_true = np.interp(x_plot, x_ref, y_ref)

        results[name] = {
            "x": x_plot,
            "y": y_plot,
            "metrics": compute_error_metrics(y_true, y_plot),
            "constraints": verify_quartic_bezier_constraints(
                x_nodes=x_nodes,
                y_nodes=y_nodes,
                means=means,
                derivatives=derivatives,
                controls=controls,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            ),
            "controls": controls,
        }

    except Exception as e:
        results[name] = {"error": str(e)}

    # ========================================================
    # 3) Bounded quartic Bézier with soft derivatives
    # ========================================================
    name = "Bounded quartic Bézier (soft derivatives)"

    try:
        controls, derivatives = build_bounded_quartic_bezier_interpolant(
            x_nodes=x_nodes,
            y_nodes=y_nodes,
            means=means,
            derivatives=None,
            derivative_method=derivative_method,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        x_plot, y_plot = evaluate_bounded_quartic_bezier(
            x_nodes=x_nodes,
            controls=controls,
            num_points=num_points,
        )

        y_true = np.interp(x_plot, x_ref, y_ref)

        constraints = verify_bounded_quartic_bezier_constraints(
            x_nodes=x_nodes,
            y_nodes=y_nodes,
            means=means,
            preferred_derivatives=derivatives,
            controls=controls,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        constraints["control_polygon_roughness"] = compute_control_polygon_roughness(controls)

        results[name] = {
            "x": x_plot,
            "y": y_plot,
            "metrics": compute_error_metrics(y_true, y_plot),
            "constraints": constraints,
            "controls": controls,
        }

    except Exception as e:
        results[name] = {"error": str(e)}

    print_metrics_table(results)
    print_constraint_report(results)

    plot_comparison(x_ref, y_ref, x_nodes, y_nodes, results)
    plot_residuals(results)

    plt.show()


if __name__ == "__main__":
    main()
