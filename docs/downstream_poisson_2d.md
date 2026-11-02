# Two-dimensional downstream heat-conduction benchmark

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

The default airfoil and rounded-notch cases set `u=1` on the logical top edge
(the airfoil/notch boundary) and `u=0` on the other three edges.

## Protocols

- `raw`: use every generated mesh exactly as evaluated.
- `exact_boundary_control`: replace only the learned model's boundary nodes by
  exact parameter-curve samples. Interior nodes are unchanged. This control
  separates boundary-geometry error from interior-grid error.

The high-resolution `PDE_Winslow` mesh supplies the reference field. Candidate
and reference fields are compared through their common logical coordinates,
then integrated on the candidate physical mesh.

## Run

```bash
python scripts/downstream/run_poisson_2d.py --config configs/downstream/poisson_2d.yaml
```

One case can be selected without editing YAML:

```bash
python scripts/downstream/run_poisson_2d.py --dataset run_ext_1_500_41_41 --values 1.22 1.25 --grids 101 201 401 --methods pde meshonet pimeshonet hmonet --protocols raw exact_boundary_control
```

If an input archive is missing, first run the standard evaluation at the same
parameter and grid size with compact output disabled or generation-only output
enabled. The downstream benchmark intentionally does not guess checkpoints.

## Retained results

Each case/protocol stores `metrics.csv`, `summary.json`, compressed nodal fields,
and optional comparison figures. `aggregate/` contains detailed results,
method-level mean/standard-deviation tables, solve success rates, and a run
manifest. Checkpoints and duplicate mesh archives are not copied.
