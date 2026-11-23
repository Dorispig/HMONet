# Two-dimensional manufactured Poisson benchmark

This benchmark consumes meshes already produced by the formal evaluator. It
does not train a network and does not duplicate model inference. Every input
archive must have the standard location

```text
outputs/results/2D/<dataset>/<nx>x<ny>/<parameter>/generated_meshes.npz
```

## Governing equation

The implementation solves the scalar steady diffusion problem

```text
-div(k grad(u)) = q  in Omega,
u = g                 on the selected logical edges.
```

It uses bilinear Q1 quadrilateral finite elements and a 2x2 Gauss rule in the
physical mesh. A non-positive element-map determinant at any Gauss point is a
solve failure; the raw protocol never silently repairs an invalid cell.

The default benchmark uses two manufactured physical-space solutions:

```text
u_exact(x,y) = sin(pi*x) sin(pi*y),
q(x,y)       = 2*k*pi^2*u_exact(x,y),
g            = u_exact on the complete physical boundary.
```

and the directional high-gradient field

```text
u_exact(x,y) = exp(x/2)*sin(2*pi*y),
q(x,y)       = k*(4*pi^2 - 1/4)*u_exact(x,y),
g            = u_exact on the complete physical boundary.
```

Consequently, every mesh is compared directly with the same analytical field;
no numerically generated reference solution is required.

## Protocols

- `raw`: use every generated mesh exactly as evaluated.
- `exact_boundary_control`: replace only the learned model's boundary nodes by
  exact parameter-curve samples. Interior nodes are unchanged. This control
  separates boundary-geometry error from interior-grid error.

The code reports nodal mean/maximum absolute error, relative physical-domain
`L2` and `H1`-seminorm errors, solve success, and observed grid-refinement
orders. The integral errors are evaluated at physical Gauss points against the
manufactured value and gradient, so they do not depend on logical interpolation
from a second mesh.

## Run

```bash
python scripts/downstream/run_poisson_2d.py --config configs/downstream/poisson_2d.yaml
```

One case can be selected without editing YAML:

```bash
python scripts/downstream/run_poisson_2d.py --dataset run_ext_1_500_41_41 --values 1.52 1.55 --grids 101 201 401 --methods pde meshonet pimeshonet hmonet --protocols raw exact_boundary_control
```

Select one manufactured field when a focused check is needed:

```bash
python scripts/downstream/run_poisson_2d.py --dataset run_ext_1_500_41_41 --values 1.55 --grids 101 201 401 --solutions exponential_sine --methods pde meshonet pimeshonet hmonet --protocols raw
```

If an input archive is missing, first run the standard evaluation at the same
parameter and grid size with compact output disabled or generation-only output
enabled. The downstream benchmark intentionally does not guess checkpoints.

## Retained results

Each case/protocol stores `metrics.csv`, `summary.json`, compressed nodal fields,
and optional comparison figures. `aggregate/` contains detailed results,
method-level mean/standard-deviation tables, solve success rates, and a run
manifest. Checkpoints and duplicate mesh archives are not copied.
