# Constrained Bézier Interpolation

Mean-preserving and bounded cubic and quartic Bézier interpolation methods for sparse time-series data.

This project implements and compares constrained Bézier interpolation methods for situations in which sparse support values are available together with additional interval information, such as prescribed means and physical bounds.

The main objective is to preserve mathematically meaningful constraints while obtaining a continuous, piecewise-smooth reconstruction of the underlying time series.

## Motivation

In scientific and engineering applications, measurements are often available only at relatively sparse time points, while additional information may be known over the intervals between them.

Relevant requirements may include:

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

Each interval is represented by a cubic Bézier curve with control points

$$
P_0,\;P_1,\;P_2,\;P_3.
$$

The endpoint values are fixed by

$$
P_0 = y_i,
\qquad
P_3 = y_{i+1}.
$$

For a cubic Bézier curve, the interval mean is equal to the arithmetic mean of its control points:

$$
\operatorname{mean}(B_i)
=
\frac{P_0+P_1+P_2+P_3}{4}.
$$

The internal control points are therefore chosen such that

$$
P_1+P_2
=
4m_i-y_i-y_{i+1},
$$

where $m_i$ is the prescribed mean on interval $[x_i,x_{i+1}]$.

The unconstrained preferred control points are

$$
P_1
=
\frac{2}{3}(y_i+y_{i+1})-2m_i,
$$

$$
P_2
=
6m_i-\frac{5}{3}(y_i+y_{i+1}).
$$

If lower or upper bounds are specified, the pair $(P_1,P_2)$ is projected onto the feasible set while preserving the exact mean condition.

Because a scalar Bézier curve lies inside the convex hull of its control points, bounded control points also guarantee a bounded interpolated curve.

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

A quartic Bézier segment contains five control points:

$$
P_0,\;P_1,\;P_2,\;P_3,\;P_4.
$$

This makes it possible to enforce five conditions exactly:

$$
B_i(x_i)=y_i,
$$

$$
B_i(x_{i+1})=y_{i+1},
$$

$$
\operatorname{mean}(B_i)=m_i,
$$

$$
B_i'(x_i)=d_i,
$$

$$
B_i'(x_{i+1})=d_{i+1}.
$$

Let

$$
h_i=x_{i+1}-x_i.
$$

The endpoint values give

$$
P_0=y_i,
\qquad
P_4=y_{i+1}.
$$

The endpoint derivative conditions determine

$$
P_1=P_0+\frac{h_i}{4}d_i,
$$

$$
P_3=P_4-\frac{h_i}{4}d_{i+1}.
$$

For a quartic Bézier curve,

$$
\operatorname{mean}(B_i)
=
\frac{P_0+P_1+P_2+P_3+P_4}{5}.
$$

Therefore,

$$
P_2
=
5m_i-P_0-P_1-P_3-P_4.
$$

The endpoint values, interval mean, and endpoint derivatives are thus satisfied exactly.

However, these five conditions uniquely determine all five control points. Additional physical bounds therefore cannot always be satisfied simultaneously.

The implementation supports two modes:

- `strict_bounds=True`: reject a segment if its control points violate the requested bounds;
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

For interval $[x_i,x_{i+1}]$, the derivative-based target control points are

$$
P_1^{\mathrm{ref}}
=
P_0+\frac{h_i}{4}d_i,
$$

$$
P_3^{\mathrm{ref}}
=
P_4-\frac{h_i}{4}d_{i+1}.
$$

The exact mean condition requires

$$
P_1+P_2+P_3
=
5m_i-P_0-P_4.
$$

The method chooses $P_1$ and $P_3$ as close as possible to the derivative-based target values while satisfying the requested bounds and leaving a feasible value for $P_2$.

The middle control point is then determined by

$$
P_2
=
5m_i-P_0-P_4-P_1-P_3.
$$

This preserves the endpoint values and interval means exactly while allowing the endpoint derivatives to deviate from their preferred values when necessary to maintain physical admissibility.

## Derivative Estimation

The quartic methods support two derivative-estimation approaches:

```text
pchip
finite_difference
```

The default is a NumPy implementation of a shape-preserving PCHIP-style derivative estimate.

It uses neighboring secant slopes and sets the derivative to zero at interior points where the adjacent slopes change sign, reducing artificial oscillations and overshoot.

The alternative method uses standard finite differences:

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
├── archive/
│   ├── balanced_mean_cubic.py
│   ├── hermite_cubic.py
│   ├── legacy_spline_methods.py
│   └── compare_cubic_previous_methods.py
│
├── README.md
└── requirements.txt
```

The `src/` directory contains the current interpolation methods.

The `examples/` directory contains the main numerical comparison.

The `archive/` directory contains earlier approaches retained to document the development of the final methods.

## Main Numerical Comparison

The main experiment is implemented in:

```text
examples/compare_interpolation_methods.py
```

It compares:

1. mean-preserving bounded cubic Bézier interpolation,
2. exact quartic Bézier interpolation with endpoint derivatives,
3. bounded quartic Bézier interpolation with soft endpoint derivatives.

The comparison reports the following approximation metrics:

- root mean squared error (RMSE),
- mean absolute error (MAE),
- maximum absolute error,
- coefficient of determination $R^2$.

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

The dataset is included as an example for reproducing the numerical comparison. Exact source attribution and applicable data-usage terms should be documented alongside the dataset before broader redistribution.

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

They are not part of the main implementation, but are retained to document the progression from earlier interpolation approaches to the final constrained Bézier methods.

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
