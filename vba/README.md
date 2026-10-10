# Excel/VBA implementation

This directory contains the spreadsheet implementation of the final bounded
quartic Bézier method with soft derivative constraints.

## Files

- `modBezier.bas` — reviewable VBA source module
- `bounded_quartic_bezier_demo.xlsm` — macro-enabled demonstration workbook

The VBA implementation follows the same mathematical construction as
`src/bounded_quartic_bezier.py`:

- endpoint values are preserved,
- prescribed interval means are preserved,
- lower and upper bounds are enforced whenever the constrained problem is feasible,
- endpoint derivatives are treated as soft targets and adjusted when required by the bounds.

The `.bas` file is included so that the implementation can be inspected
directly on GitHub without opening the macro-enabled workbook.

For the mathematical derivation and numerical comparison, see
`paper/constrained_bezier_interpolation.pdf`.
