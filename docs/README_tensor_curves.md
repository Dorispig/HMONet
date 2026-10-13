# Curve values and analytic derivatives

`curve(t)` preserves the NumPy/Torch input backend. `curve.derivative(t)`
returns the global [0,1] parameter derivative. Lines, circles, ellipses,
NACA4, Bezier, B-splines, mapped curves and piecewise curves use formula
derivatives. FittedCurve uses the slopes of its piecewise-linear interpolant.
Generic functions use their supplied derivative or bounded finite differences.
At endpoints, the finite-difference denominator is the actual valid parameter
interval, not twice the nominal step.

## Repeated and batched tensor queries

```python
import torch
from src.data.geometry.parametric_curve_2d import ParametricCurve, BoundaryCurveSet2D

curves = ParametricCurve.batch(curve_objects, device="cuda", dtype=torch.float32)
points = curves(t)                # t: [B,N], points: [B,N,D]
tangents = curves.derivative(t)   # same shape as points

boundary = BoundaryCurveSet2D.batch(boundary_objects, "cuda", torch.float32)
left = boundary.left(eta)
right = boundary.right(eta)
bottom = boundary.bottom(xi)
top = boundary.top(xi)
projections = boundary.project_torch(xi, eta)  # [B,N,4,2]
```

Construct these resident objects once per batch, then reuse them for all
forwards and losses. Object/type traversal and coefficient upload occur during
construction only. Native query formulas have no Python batch/point/group loops.
Mixed primitive types use a dispatch tree constructed up front; their query
arithmetic is still batched. Native B-splines are compiled to polynomial pieces
once using SciPy, then queried entirely with Torch gather/product operations.
There is no memoization of query positions or projection coordinates.

The 2D trainer and fixed-grid training MSE use this resident path. Evaluation
passes original boundary objects directly into the model and does not prepare
resident batches, Cartesian selections, or CUDA Graphs. Training coefficients
remain useful because each batch reuses them across several forward/loss calls.
Single-boundary 2D evaluation now calls the four concrete curve objects directly.
The generic dispatcher remains for heterogeneous geometry batches and the
six-surface 3D API.

## Derivatives and limitations

Mapped derivatives include the interval scaling. Piecewise derivatives at an
exact internal junction average the left/right global-parameter derivatives.
The tensor value path retains that same junction gradient convention. First and
second endpoint derivatives for polynomial curves use stable product bases.

NACA sqrt(x) parameterization has a singular tangent at the leading edge.
At exactly x=0, the existing finite tensor endpoint convention is retained
(the sqrt term contributes zero derivative); it is not a finite mathematical
analytic derivative of that singular parameterization. Interior NACA derivatives
are analytic. Geometric coefficients are fixed dataset data; rebuilding the
resident object is required after editing them or changing device/dtype.

Custom NumPy-only callables cannot automatically become differentiable GPU
functions: provide an evaluate_torch implementation for that use. They retain
ordinary value compatibility, but requesting coordinate autograd fails explicitly.

No model weights, checkpoint format, data file layout or loss weights change.
The normal-loss target now uses resident analytic tangents in 2D, preserving its
squared-cosine objective. Existing trained checkpoints remain loadable.


## Direct evaluation without geometry setup

`BoundaryCurveSet2D.project_torch(xi, eta)` calls left/right at eta and
bottom/top at xi. For supported CUDA analytic curves, live scalar coefficients
are passed directly as kernel arguments; no resident geometry adapter or
projected-point cache is created. Lines, circles, ellipses, NACA4 and compatible
two-piece NACA boundaries use this path. Other curves retain the general formula
fallback. Editing geometry values is reflected in the next forward.

HMONet2D current mapping uses three fused elementwise kernels for rational
weights, coordinate combination and exact corner replacement. Explicit rounding
preserves the reference arithmetic when large raw network outputs are cancelled
by boundary constraints. The runtime's eight-tensor-input limit prevents using
a single generic kernel without additional packing. CUDA compilation is lazy;
only executable kernels are cached, never geometry or network activations.
First-use compilation is not a steady-state prediction time and is normally
covered by the configured warmup forwards.

For at least 1024 query points per forward, coordinate feature contraction uses
a matrix product to avoid two full elementwise product tensors. Smaller queries
retain the existing reductions because the measured improvement was negligible.
These model-level optimizations apply only with autograd disabled; training and
coordinate Jacobians retain the reference differentiable calculation. CPU and
unavailable-compiler cases fall back automatically. The 3D mapping is unchanged.

`src.data.backend.inference_options.inference_routes` supplies internal ablation switches;
ordinary evaluation enables the routes automatically and needs no new CLI flags.
Existing checkpoints and evaluation output paths remain compatible.
