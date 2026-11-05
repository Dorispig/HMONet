# Ex11: HMONet architecture ablation

Ex11 isolates four architectural changes and evaluates their cumulative form.  Its baseline is the Ex9 `e9_1` training recipe: mixed sampling (200 global, 100 adaptive, and 100 boundary-layer points), scale-invariant elliptic-grid PDE loss, normalized Jacobian and minimum-singular-value barriers, and the `corner_corrected` exact-boundary mapping.  Every Ex11 run uses an initial learning rate of `4e-4`.

## Coordinate lifting

The default lifting mode is `power_fourier` with maximum power `P=2`.  For a query coordinate

`q = (q_1, ..., q_d)`, define

`s_P(q) = (q_1, q_1^2, ..., q_1^P, ..., q_d, q_d^2, ..., q_d^P)`

and

`Phi_P(q) = [s_P(q), sin(2 pi s_P(q)), cos(2 pi s_P(q))]`.

The lifted dimension is `3 d P`: 12 in 2-D and 18 in 3-D for `P=2`.

## Variants

| Name | Change relative to the baseline |
|---|---|
| `e1_modified_mlp` | Replace the two branch MLPs and shared trunk MLP with gated ModifiedMLP blocks. |
| `e2_cross_attention` | Add query-to-boundary cross-attention. Query/trunk features are queries; encoded boundary points are keys and values. |
| `e3_boundary_decoder` | Add a boundary-layer decoder whose residual is gated by `exp(-d_boundary / width)`. |
| `e4_decoupled_weights` | Give the mapping-weight pathway separate geometry and location adapters instead of reusing exactly the raw-coordinate decoder features. |
| `e5_qg_hmonet` | Train QG-HMONet, described below. |
| `e6_modified_cross` | `e1 + e2`. |
| `e7_modified_cross_boundary` | `e1 + e2 + e3`. |
| `e8_full_factorized` | `e1 + e2 + e3 + e4`. |
| `e9_meshonet_200k` | MeshONet comparison model uses a 40-dimensional branch bottleneck and uniform width 80, giving 208,242 parameters on the `41 x 41` training boundary. |
| `e10_pimeshonet_200k` | PI-MeshONet uses the same 40/80 regular dimensions, giving 191,122 parameters on the `41 x 41` training boundary. |
| `e11_raw_pde` | Keep the baseline model, sampling, barriers, and optimizer unchanged, but replace the scale-invariant PDE loss by the raw mean squared elliptic-grid residual. |

## QG-HMONet

QG-HMONet is the **Query--Geometry Gated Hard-Boundary Mesh Operator Network**.  It preserves HMONet's input/output contract and exact `corner_corrected` boundary mapping.  It does not calculate curvature, normals, an SDF, or other handcrafted differential geometry.

1. The existing `x`- and `y`-boundary branches form the global geometry descriptor.
2. Directed boundary entities are represented by physical coordinates, a normalized entity parameter, and a learned entity identifier.  Shared Conv1D paths with kernel sizes 3, 5, and 9 form local multiscale features.  A fusion projection and boundary self-attention produce local geometry tokens.
3. The lifted query passes through the shared trunk.  Multi-head cross-attention uses the trunk feature as the query and the local boundary tokens as keys and values.  A learned sigmoid gate and residual normalization yield a locally enhanced query feature.
4. Raw-coordinate heads and mapping-coefficient heads use separately adapted features.  A boundary-layer decoder adds a localized residual before the exact hard-boundary fusion.
5. The existing exact parametric projection and hard-boundary fusion produce the physical mesh coordinates.

The design is inspired by query/condition cross-attention and geometric gating in GNOT, the importance of explicit varying-geometry representations demonstrated by GINO, and Fourier-feature mitigation of coordinate-MLP spectral bias.  Unlike GINO, this implementation deliberately avoids an explicit SDF; unlike PI-MeshONet, QG-HMONet retains exact parametric hard-boundary fusion.

References:

- GNOT: <https://proceedings.mlr.press/v202/hao23c.html>
- GINO: <https://proceedings.neurips.cc/paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html>
- Fourier Features: <https://proceedings.neurips.cc/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html>

## Commands

Run the baseline and the first five variants on `run_ext_1_500_41_41`:

```text
python scripts/experiments/run.py ex11 baseline e1_modified_mlp e2_cross_attention e3_boundary_decoder e4_decoupled_weights e5_qg_hmonet --dataset run_ext_1_500_41_41 --seed 20261030 --grids 101 201 401 --tag physics_v1
```

Run the complete suite, including the approximately 200k-parameter MeshONet
and PI-MeshONet comparisons:

```text
python scripts/experiments/run.py ex11 baseline all --dataset run_ext_1_500_41_41 --seed 20261030 --grids 101 201 401 --tag physics_v1
```

For `air_t_ext_1_500_21_21`, replace only the dataset name.  Add `--gpus 0 1 2 3` on a four-GPU server if dynamic multi-GPU scheduling is desired.

Measure parameter counts and forward latency independently of training:

```text
python tools/benchmark_ex11.py --device cuda --warmup 20 --repeats 100
```

