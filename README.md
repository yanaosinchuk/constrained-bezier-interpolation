# Constrained Bézier Interpolation

Mean-preserving and bounded cubic and quartic Bézier interpolation methods for sparse time-series data.

This project implements and compares constrained Bézier interpolation methods for situations in which sparse support values are available together with additional interval information, such as prescribed means and physical bounds.

The main objective is to preserve mathematically meaningful constraints while obtaining a continuous, piecewise-smooth reconstruction of the underlying time series.

## Motivation

In scientific and engineering applications, measurements are often available only at relatively sparse time points, while additional information may be known over the intervals between them.

Typical requirements include:

- exact interpolation of measured support values,
- preservation of prescribed interval means,
- lower or upper physical bounds,
- controlled local shape between support points,
- optional use of derivative information.

Standard interpolation methods do not necessarily satisfy these requirements simultaneously.

This repository investigates cubic and quartic Bézier constructions that enforce different combinations of these constraints.

## Implemented Methods

### 1. Mean-Preserving Cubic Bézier Interpolation

Implemented in:

```text
src/mean_preserving_cubic.py
```

Main function:

```python
build_mean_preserving_cubic_bezier(...)
```

For each interval, the method constructs a cubic Bézier curve with control points `P0, P1, P2, P3`.

The endpoint values are fixed as:

```text
P0 = y_i
P3 = y_{i+1}
```

For a cubic Bézier curve, the interval mean equals the arithmetic mean of the control points:

```text
mean(B_i) = (P0 + P1 + P2 + P3) / 4
```

Therefore, the internal control points must satisfy:

```text
P1 + P2 = 4*m_i - y_i - y_{i+1}
```

where `m_i` is the prescribed mean on interval `[x_i, x_{i+1}]`.

If lower or upper bounds are specified, the pair `(P1, P2)` is projected onto the feasible set while preserving the exact mean condition.

Because a scalar Bézier curve lies inside the convex hull of its control points, bounded control points guarantee a bounded interpolated curve.

---

### 2. Exact Quartic Bézier Interpolation

Implemented in:

```text
src/exact_quartic_bezier.py
```

Main function:

```python
build_exact_quartic_bezier(...)
```

A quartic Bézier segment has five control points:

```text
P0, P1, P2, P3, P4
```

This allows the method to satisfy five conditions exactly:

- endpoint interpolation at `x_i`,
- endpoint interpolation at `x_{i+1}`,
- preservation of the prescribed interval mean,
- derivative matching at `x_i`,
- derivative matching at `x_{i+1}`.

Let:

```text
h_i = x_{i+1} - x_i
```

Then:

```text
P0 = y_i
P4 = y_{i+1}
P1 = P0 + (h_i / 4) * d_i
P3 = P4 - (h_i / 4) * d_{i+1}
```

For a quartic Bézier curve, the interval mean equals:

```text
mean(B_i) = (P0 + P1 + P2 + P3 + P4) / 5
```

So the middle control point is determined by:

```text
P2 = 5*m_i - P0 - P1 - P3 - P4
```

This means that endpoint values, interval means, and endpoint derivatives are all satisfied exactly.

However, these five exact conditions uniquely determine the control points. As a result, additional physical bounds may become infeasible.

The implementation therefore supports two modes:

- `strict_bounds=True`: reject a segment if the requested bounds are violated,
- `strict_bounds=False`: construct the exact interpolant and report bound violations diagnostically.

---

### 3. Bounded Quartic Bézier Interpolation with Soft Derivatives

Implemented in:

```text
src/bounded_quartic_bezier.py
```

Main function:

```python
build_bounded_quartic_bezier(...)
```

This method enforces the following conditions exactly whenever the bounded interpolation problem is feasible:

- endpoint interpolation,
- prescribed interval means,
- optional lower and upper bounds.

Endpoint derivatives are treated as preferred values rather than hard constraints.

For interval `[x_i, x_{i+1}]`, the derivative-based target control points are:

```text
P1_ref = P0 + (h_i / 4) * d_i
P3_ref = P4 - (h_i / 4) * d_{i+1}
```

The exact mean condition requires:

```text
P1 + P2 + P3 = 5*m_i - P0 - P4
```

The method chooses `P1` and `P3` as close as possible to their derivative-based target values while satisfying the requested bounds and leaving a feasible value for `P2`.

The middle control point is then determined by:

```text
P2 = 5*m_i - P0 - P4 - P1 - P3
```

This preserves endpoint values and interval means exactly while allowing endpoint derivatives to deviate from their preferred values when necessary to maintain physical admissibility.

## Derivative Estimation

The quartic methods support two derivative-estimation approaches:

```text
pchip
finite_difference
```

The default is a NumPy implementation of a shape-preserving PCHIP-style derivative estimate.

The alternative method uses:

```python
numpy.gradient
```

The derivative method can be selected through:

```python
derivative_method="pchip"
```

or

```python
derivative_method="finite_difference"
```

## Repository Structure

```text
constrained-bezier-interpolation/
│
├── src/
│   ├── mean_preserving_cubic.py
│   ├── exact_quartic_bezier.py
│   └── bounded_quartic_bezier.py
│
├── examples/
│   └── compare_interpolation_methods.py
│
├── data/
│   └── pegelonline_leunneu_2024.xlsx
│
├── vba/
│   ├── modBezier.bas
│   └── bounded_quartic_bezier_demo.xlsm
│
├── paper/
│   ├── main.tex
│   ├── references.bib
│   ├── constrained_bezier_interpolation.pdf
│   └── figures/
│       ├── Logo-Hochschule-Darmstadt.png
│       ├── Example_Excel.png
│       ├── VBA_Table_Result_Example.png
│       ├── Chart.png
│       ├── Compare.png
│       └── Residual_comparison.png
│
├── archive/
│   ├── balanced_mean_cubic.py
│   ├── hermite_cubic.py
│   ├── legacy_spline_methods.py
│   └── compare_cubic_previous_methods.py
│
├── README.md
└── requirements.txt

The `src/` directory contains the current Python implementations of the three interpolation methods.

The `examples/` directory contains the main numerical comparison script.

The `data/` directory contains the example time-series data used in the numerical experiments.

The `vba/` directory contains the Excel/VBA implementation of the final bounded quartic Bézier method together with a demonstration workbook.

The `paper/` directory contains the complete LaTeX source, bibliography, figures, and compiled PDF of the accompanying technical report.

The `archive/` directory contains earlier interpolation approaches retained to document the development of the final methods.
```

It compares:

1. mean-preserving bounded cubic Bézier interpolation,
2. exact quartic Bézier interpolation with endpoint derivatives,
3. bounded quartic Bézier interpolation with soft endpoint derivatives.

The comparison reports the following approximation metrics:

- root mean squared error (RMSE),
- mean absolute error (MAE),
- maximum absolute error,
- coefficient of determination (`R²`).

It also evaluates method-specific constraint diagnostics, including:

- maximum endpoint interpolation error,
- maximum interval-mean error,
- derivative error or derivative modification,
- lower-bound violation,
- upper-bound violation.

For the bounded quartic method, the control-polygon roughness is also reported as an additional diagnostic.

The script produces plots of the reconstructed time series and the corresponding residuals.

## Input Data

The example dataset is located at:

```text
data/pegelonline_leunneu_2024.xlsx
```

The comparison scripts expect the dataset to contain at least the following columns:

```text
timestamp
value
```

The timestamps are converted internally to elapsed hours relative to the first observation.

A subset of the reference observations is then selected as support points using a configurable spacing.

For each interval between consecutive support points, an interval mean is computed from the reference data and supplied to the interpolation methods as an additional constraint.

## Installation

Clone the repository:

```bash
git clone https://github.com/yanaosinchuk/constrained-bezier-interpolation.git
cd constrained-bezier-interpolation
```

Optionally create a virtual environment:

```bash
python -m venv .venv
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The project requires:

```text
numpy
pandas
matplotlib
openpyxl
```

## Running the Main Experiment

Run the comparison from the repository root:

```bash
python examples/compare_interpolation_methods.py
```

The main experimental settings can be modified directly in the script:

```python
step = 672
num_points = 500

lower_bound = 0.0
upper_bound = None

derivative_method = "pchip"
```

Here:

- `step` determines the spacing between selected support points,
- `num_points` controls the number of evaluation points per interval,
- `lower_bound` and `upper_bound` define optional physical constraints,
- `derivative_method` selects the derivative-estimation procedure used by the quartic methods.

## Previous Development Methods

Earlier interpolation approaches are retained in:

```text
archive/
```

They include:

- a balanced mean-preserving cubic Bézier construction,
- a cubic Hermite-style Bézier interpolant based on endpoint derivatives,
- earlier cubic and quartic spline formulations.

These methods can be compared using:

```bash
python archive/compare_cubic_previous_methods.py
```

They are not part of the main implementation but are retained to document the progression from earlier interpolation approaches to the final constrained Bézier methods.

## Methodological Distinction

The two quartic methods deliberately solve different interpolation problems.

### Exact quartic method

The exact quartic construction preserves:

- endpoint values,
- interval means,
- endpoint derivatives.

These conditions are exact, but physical bounds may be violated.

### Bounded quartic method

The bounded quartic construction preserves:

- endpoint values,
- interval means,
- requested physical bounds whenever the problem is feasible.

Endpoint derivatives are treated as soft targets and may be modified when necessary.

The comparison between these methods illustrates the trade-off between derivative fidelity and physical admissibility.

## Applications

The methods are intended for constrained reconstruction of sparse numerical time series in situations where additional interval-level information is available.

Potential applications include:

- environmental time series,
- hydrological measurements,
- sensor-signal reconstruction,
- scientific data processing,
- constrained numerical interpolation,
- reconstruction of sparsely sampled physical measurements.

## Author

Yana Osinchuk  
Applied Mathematics
