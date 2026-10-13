"""Shared training utilities for the parameterized 2-D and 3-D datasets."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from src.utils.paths import resolve_project_path

# Apply the same requested scale to reported metrics and convergence checks.
FIXED_GRID_MSE_SCALE = 2.0

# Refined logical support points depend only on grid shape, device and dtype.
# Keeping them resident avoids rebuilding identical parameter grids every batch.
_REFINED_LOGICAL_CACHE = {}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dataset_paths(config: dict, mode: str, project_root: Path) -> list[Path]:
    datasets = config.get("datasets", {})
    keys = ["interpolation", "extrapolation"] if mode == "both" else [mode]
    paths = []
    for key in keys:
        value = datasets.get(key)
        if value:
            paths.append(resolve_project_path(value, base=project_root))
    return paths


def sample_global_random_batch(
    dataset, indices, dimension, count, device, generator=None
):
    """Draw fresh uniform logical points and interpolate PDE targets."""
    indices = list(indices)
    boundary = torch.as_tensor(
        np.asarray(dataset.boundary_points[indices]),
        dtype=torch.float32,
        device=device,
    )
    mesh = torch.as_tensor(
        np.asarray(dataset.mesh_points[indices]), dtype=torch.float32, device=device
    )
    shape = tuple(dataset.grid_shape)
    logical = sample_shared_random_points(
        len(indices), count, dimension, device, generator
    )
    return boundary, logical, interpolate_pde_mesh(mesh, logical, shape)


def sample_shared_random_points(
    batch_size, count, dimension, device, generator=None, margin=0.0
):
    """Draw one shared uniform point set from an open or inset unit domain.

    ``margin`` may be a scalar or one value per logical axis. A zero margin
    samples the complete open unit domain. Centered finite differences pass
    their step widths so every neighbor remains inside the domain.
    """
    if int(count) < 1:
        raise ValueError("The random sample count must be positive")
    margins = torch.as_tensor(margin, dtype=torch.float32, device=device).flatten()
    if margins.numel() == 1:
        margins = margins.expand(int(dimension))
    if margins.numel() != int(dimension):
        raise ValueError("margin must be scalar or contain one value per dimension")
    if not bool(torch.isfinite(margins).all()) or bool(
        ((margins < 0) | (margins >= 0.5)).any()
    ):
        raise ValueError("sampling margins must be finite and lie in [0, 0.5)")
    eps = torch.finfo(torch.float32).eps
    lower = torch.where(margins > 0, margins, torch.full_like(margins, eps))
    upper = torch.where(margins > 0, 1.0 - margins, 1.0 - eps)
    points = torch.rand((1, int(count), dimension), device=device, generator=generator)
    points = lower + (upper - lower) * points
    return points.expand(batch_size, -1, -1)


def third_refined_surface_parameters(nu, nv, device=None, dtype=None):
    """Return a three-way-refined face grid without original surface nodes.

    Every original parameter interval is split into thirds. All newly inserted
    face points are retained, including the two new points on each original
    cell edge, while locations that were already present in the input surface
    grid are removed.
    """
    nu, nv = int(nu), int(nv)
    if nu < 2 or nv < 2:
        raise ValueError("A boundary face needs at least two nodes per axis")
    device = torch.device("cpu" if device is None else device)
    dtype = torch.get_default_dtype() if dtype is None else dtype
    key = ("surface_thirds", nu, nv, str(device), dtype)
    cached = _REFINED_LOGICAL_CACHE.get(key)
    if cached is not None:
        return cached
    refinement = 3
    ui = torch.arange(refinement * (nu - 1) + 1, device=device)
    vi = torch.arange(refinement * (nv - 1) + 1, device=device)
    vv, uu = torch.meshgrid(vi, ui, indexing="ij")
    inserted = ~(
        (uu.remainder(refinement) == 0)
        & (vv.remainder(refinement) == 0)
    )
    u = uu[inserted].to(dtype=dtype) / (refinement * (nu - 1))
    v = vv[inserted].to(dtype=dtype) / (refinement * (nv - 1))
    _REFINED_LOGICAL_CACHE[key] = (u, v)
    return u, v


def _cached_curve_support(shape, device, dtype):
    """Return cached quarter-refined 2-D boundary parameters and selectors."""
    nx, ny = map(int, shape)
    key = ("curve_quarters", nx, ny, str(device), dtype)
    cached = _REFINED_LOGICAL_CACHE.get(key)
    if cached is not None:
        return cached
    fractions = torch.tensor((0.25, 0.5, 0.75), device=device, dtype=dtype)
    horizontal = (
        torch.arange(nx - 1, device=device, dtype=dtype)[:, None]
        + fractions[None, :]
    ).reshape(-1) / (nx - 1)
    vertical = (
        torch.arange(ny - 1, device=device, dtype=dtype)[:, None]
        + fractions[None, :]
    ).reshape(-1) / (ny - 1)
    zero_h, one_h = torch.zeros_like(horizontal), torch.ones_like(horizontal)
    zero_v, one_v = torch.zeros_like(vertical), torch.ones_like(vertical)
    support = torch.cat(
        (
            torch.stack((horizontal, zero_h), -1),
            torch.stack((one_v, vertical), -1),
            torch.stack((horizontal, one_h), -1),
            torch.stack((zero_v, vertical), -1),
        ),
        0,
    )
    selectors = torch.cat(
        (
            torch.full_like(horizontal, 2, dtype=torch.long),
            torch.full_like(vertical, 1, dtype=torch.long),
            torch.full_like(horizontal, 3, dtype=torch.long),
            torch.full_like(vertical, 0, dtype=torch.long),
        )
    )
    _REFINED_LOGICAL_CACHE[key] = (support, selectors)
    return support, selectors


def _cached_surface_support(shape, device, dtype):
    """Return one cached, padded parameter batch for all six 3-D faces."""
    nx, ny, nz = map(int, shape)
    key = ("surface_support", nx, ny, nz, str(device), dtype)
    cached = _REFINED_LOGICAL_CACHE.get(key)
    if cached is not None:
        return cached
    specs = (
        ("x_min", ny, nz, 0, 0.0, 1, 2),
        ("x_max", ny, nz, 0, 1.0, 1, 2),
        ("y_min", nx, nz, 1, 0.0, 0, 2),
        ("y_max", nx, nz, 1, 1.0, 0, 2),
        ("z_min", nx, ny, 2, 0.0, 0, 1),
        ("z_max", nx, ny, 2, 1.0, 0, 1),
    )
    parameters = [
        third_refined_surface_parameters(nu, nv, device=device, dtype=dtype)
        for _, nu, nv, *_ in specs
    ]
    maximum = max(u.numel() for u, _ in parameters)
    face_u = torch.zeros((6, maximum), device=device, dtype=dtype)
    face_v = torch.zeros_like(face_u)
    valid = torch.zeros((6, maximum), device=device, dtype=torch.bool)
    logical_parts = []
    for face, ((_, _, _, fixed_axis, fixed_value, u_axis, v_axis), (u, v)) in enumerate(
        zip(specs, parameters)
    ):
        count = u.numel()
        face_u[face, :count] = u
        face_v[face, :count] = v
        valid[face, :count] = True
        support = torch.zeros((count, 3), device=device, dtype=dtype)
        support[:, fixed_axis] = fixed_value
        support[:, u_axis] = u
        support[:, v_axis] = v
        logical_parts.append(support)
    result = (
        torch.cat(logical_parts, dim=0),
        face_u,
        face_v,
        valid,
        tuple(spec[0] for spec in specs),
    )
    _REFINED_LOGICAL_CACHE[key] = result
    return result


def _squared_distances(left, right):
    """Compute batched pairwise squared distances without a [B,Q,S,D] tensor."""
    left_norm = left.square().sum(-1, keepdim=True)
    right_norm = right.square().sum(-1).unsqueeze(-2)
    distances = torch.baddbmm(
        left_norm + right_norm,
        left,
        right.transpose(1, 2),
        beta=1.0,
        alpha=-2.0,
    )
    return distances.clamp_min_(0.0)


def _boundary_idw(
    target,
    straight_support,
    displacement,
    neighbors=-1,
    query_chunk_size=512,
    support_chunk_size=2048,
):
    """Spread boundary displacement by global IDW or K-nearest local IDW.

    ``neighbors=-1`` uses every support point. A positive value retains only
    the nearest K supports after a chunked top-k search. Larger chunks reduce
    Python launches while the norm/matmul distance formula bounds memory.
    """
    neighbors = int(neighbors)
    query_chunk_size = int(query_chunk_size)
    support_chunk_size = int(support_chunk_size)
    if neighbors == 0 or neighbors < -1:
        raise ValueError("IDW neighbors must be -1 or a positive integer")
    if query_chunk_size < 1 or support_chunk_size < 1:
        raise ValueError("IDW chunk sizes must be positive")
    epsilon = torch.finfo(target.dtype).eps
    adjusted = []
    support_count = straight_support.shape[1]
    local_count = min(neighbors, support_count) if neighbors > 0 else -1
    for first in range(0, target.shape[1], query_chunk_size):
        base = target[:, first : first + query_chunk_size]
        if local_count > 0:
            best_distances = None
            best_indices = None
            for support_first in range(0, support_count, support_chunk_size):
                straight = straight_support[
                    :, support_first : support_first + support_chunk_size
                ]
                distances = _squared_distances(base, straight)
                if best_distances is None:
                    keep = min(local_count, distances.shape[-1])
                    best_distances, positions = torch.topk(
                        distances, keep, dim=-1, largest=False, sorted=False
                    )
                    best_indices = positions + support_first
                else:
                    previous = best_distances.shape[-1]
                    candidates = torch.cat((best_distances, distances), dim=-1)
                    keep = min(local_count, candidates.shape[-1])
                    best_distances, positions = torch.topk(
                        candidates,
                        keep,
                        dim=-1,
                        largest=False,
                        sorted=False,
                    )
                    previous_indices = best_indices.gather(
                        -1, positions.clamp_max(previous - 1)
                    )
                    new_indices = support_first + (positions - previous)
                    best_indices = torch.where(
                        positions < previous, previous_indices, new_indices
                    )
            batch_ids = torch.arange(target.shape[0], device=target.device)[
                :, None, None
            ]
            local_displacement = displacement[batch_ids, best_indices]
            weights = best_distances.clamp_min(epsilon).reciprocal()
            correction = (weights[..., None] * local_displacement).sum(-2)
            correction = correction / weights.sum(-1, keepdim=True)
        else:
            numerator = torch.zeros_like(base)
            denominator = base.new_zeros((*base.shape[:-1], 1))
            for support_first in range(0, support_count, support_chunk_size):
                straight = straight_support[
                    :, support_first : support_first + support_chunk_size
                ]
                delta = displacement[
                    :, support_first : support_first + support_chunk_size
                ]
                weights = _squared_distances(base, straight).clamp_min_(epsilon).reciprocal_()
                numerator.add_(torch.bmm(weights, delta))
                denominator.add_(weights.sum(-1, keepdim=True))
            correction = numerator / denominator
        adjusted.append(base + correction)
    return torch.cat(adjusted, dim=1)
    adjusted = []
    epsilon = torch.finfo(target.dtype).eps
    for first in range(0, target.shape[1], 128):
        base = target[:, first : first + 128]
        numerator = torch.zeros_like(base)
        denominator = base.new_zeros((*base.shape[:-1], 1))
        for support_first in range(0, straight_support.shape[1], 512):
            straight = straight_support[:, support_first : support_first + 512]
            delta = displacement[:, support_first : support_first + 512]
            distance_squared = (
                base.unsqueeze(-2) - straight.unsqueeze(-3)
            ).square().sum(-1)
            weights = distance_squared.clamp_min(epsilon).reciprocal()
            numerator = numerator + torch.einsum("bqs,bsd->bqd", weights, delta)
            denominator = denominator + weights.sum(-1, keepdim=True)
        adjusted.append(base + numerator / denominator)
    return torch.cat(adjusted, dim=1)


def interpolate_pde_mesh(
    mesh,
    logical,
    shape,
    target_interpolation="bilinear",
    boundary_geometry=None,
    idw_neighbors=-1,
    idw_query_chunk_size=512,
    idw_support_chunk_size=2048,
):
    """Interpolate saved PDE nodes with bilinear or global boundary-IDW targets.

    ``curved_boundary_cell`` keeps its public configuration name for backward
    compatibility. In 2-D each boundary-node interval is divided into quarters
    and only its three strict interior points are retained. In 3-D each boundary
    face grid is split into thirds in both parameter directions and all original
    face nodes are removed. Exact-minus-grid-interpolation displacements at the
    new supports are spread globally using inverse squared physical distance.
    """
    dimension = len(shape)
    batch_size, count = logical.shape[:2]
    logical = logical.detach()
    mode = str(target_interpolation).lower()
    if mode not in {"bilinear", "curved_boundary_cell"}:
        raise ValueError(
            "target_interpolation must be bilinear or curved_boundary_cell"
        )
    if dimension not in (2, 3):
        raise ValueError("PDE mesh interpolation supports only 2-D and 3-D grids")

    if dimension == 2:
        nx, ny = shape
        values = mesh.reshape(batch_size, ny, nx, 2).permute(0, 3, 1, 2)

        def sample(points):
            grid = (points * 2 - 1).reshape(batch_size, -1, 1, 2)
            return (
                torch.nn.functional.grid_sample(
                    values, grid, align_corners=True, mode="bilinear"
                )
                .squeeze(-1)
                .permute(0, 2, 1)
            )

        target = sample(logical)
        if mode == "curved_boundary_cell":
            if boundary_geometry is None or not hasattr(
                boundary_geometry, "project_torch"
            ):
                raise ValueError(
                    "curved_boundary_cell requires batched parametric boundaries"
                )

            support, selectors = _cached_curve_support(
                shape, logical.device, logical.dtype
            )
            horizontal_count = 3 * (nx - 1)
            vertical_count = 3 * (ny - 1)
            support_batch = support.unsqueeze(0).expand(batch_size, -1, -1)
            straight_support = sample(support_batch)
            xi_support = support_batch[..., 0]
            eta_support = support_batch[..., 1]
            with torch.no_grad():
                projected = boundary_geometry.project_torch(xi_support, eta_support)
                exact_support = projected.gather(
                    -2,
                    selectors[None, :, None, None].expand(
                        batch_size, -1, 1, projected.shape[-1]
                    ),
                ).squeeze(-2)
                displacement = exact_support - straight_support

            target = _boundary_idw(
                target,
                straight_support,
                displacement,
                idw_neighbors,
                idw_query_chunk_size,
                idw_support_chunk_size,
            )
    else:
        nx, ny, nz = shape
        values = mesh.reshape(batch_size, nz, ny, nx, 3).permute(0, 4, 1, 2, 3)

        def sample(points):
            grid = (points * 2 - 1).reshape(batch_size, -1, 1, 1, 3)
            return (
                torch.nn.functional.grid_sample(
                    values, grid, align_corners=True, mode="bilinear"
                )
                .squeeze(-1)
                .squeeze(-1)
                .permute(0, 2, 1)
            )

        target = sample(logical)
        if mode == "curved_boundary_cell":
            if boundary_geometry is None:
                raise ValueError(
                    "3-D curved_boundary_cell requires parametric boundary surfaces"
                )
            geometries = (
                list(boundary_geometry)
                if isinstance(boundary_geometry, (list, tuple))
                else [boundary_geometry] * batch_size
            )
            if len(geometries) != batch_size:
                raise ValueError("Boundary surfaces must match the numerical batch")

            from src.data.backend.parametric_torch import evaluate_surfaces_torch

            support, face_u, face_v, valid, names = _cached_surface_support(
                shape, logical.device, logical.dtype
            )
            support_batch = support.unsqueeze(0).expand(batch_size, -1, -1)
            straight_support = sample(support_batch)
            with torch.no_grad():
                maximum = face_u.shape[1]
                surfaces = [
                    getattr(geometry, name)
                    for name in names
                    for geometry in geometries
                ]
                u = face_u[:, None, :].expand(-1, batch_size, -1).reshape(
                    6 * batch_size, maximum
                )
                v = face_v[:, None, :].expand(-1, batch_size, -1).reshape(
                    6 * batch_size, maximum
                )
                exact_faces = evaluate_surfaces_torch(surfaces, u, v)
                exact_faces = exact_faces.reshape(6, batch_size, maximum, 3).permute(
                    1, 0, 2, 3
                )
                exact_support = exact_faces[:, valid]
            target = _boundary_idw(
                target,
                straight_support,
                exact_support - straight_support,
                idw_neighbors,
                idw_query_chunk_size,
                idw_support_chunk_size,
            )
    return target


def collate_parametric_samples(samples):
    """Stack numerical fields while preserving the matching boundary objects."""
    batch = {
        key: torch.as_tensor(
            np.stack([sample[key] for sample in samples]), dtype=torch.float32
        )
        for key in ("boundary_points", "mesh_points")
    }
    if "boundary" in samples[0]:
        batch["boundary"] = [sample["boundary"] for sample in samples]
    return batch


def make_data_loader(datasets, batch_size, seed=0, shuffle=False):
    """Create a reproducible loader without applying tensor collation to curves."""
    if not datasets or any(len(dataset) == 0 for dataset in datasets):
        raise ValueError("Training datasets must not be empty")
    if len({tuple(dataset.grid_shape) for dataset in datasets}) != 1:
        raise ValueError("Datasets in one training run must share their grid shape")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    data = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    return DataLoader(
        data,
        batch_size=int(batch_size),
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(int(seed)),
        collate_fn=collate_parametric_samples,
        num_workers=0,
        drop_last=False,
    )


def logical_grid_and_interior_indices(shape, device):
    """Return x-fastest logical coordinates and strictly interior node indices."""
    if any(int(size) < 3 for size in shape):
        raise ValueError("Each grid axis needs at least three nodes for interior MSE")
    axes = [torch.linspace(0, 1, int(size), device=device) for size in reversed(shape)]
    components = torch.meshgrid(*axes, indexing="ij")
    grid = torch.stack(tuple(reversed(components)), dim=-1).reshape(-1, len(shape))
    mask = torch.zeros(tuple(reversed(shape)), dtype=torch.bool, device=device)
    mask[(slice(1, -1),) * len(shape)] = True
    return grid, torch.nonzero(mask.reshape(-1), as_tuple=False).flatten()


def sample_fixed_grid_batch(mesh, shape, count, margin=0.0):
    """Sample common fixed nodes, optionally excluding a boundary strip."""
    grid, interior = logical_grid_and_interior_indices(shape, mesh.device)
    margins = torch.as_tensor(margin, dtype=grid.dtype, device=grid.device).flatten()
    if margins.numel() == 1:
        margins = margins.expand(len(shape))
    if margins.numel() != len(shape) or bool(
        ((margins < 0) | (margins >= 0.5)).any()
    ):
        raise ValueError("margin must contain valid values for every grid axis")
    eligible = ((grid[interior] >= margins) & (grid[interior] <= 1 - margins)).all(-1)
    interior = interior[eligible]
    if not 0 < int(count) <= len(interior):
        raise ValueError(
            "fixed_grid_count exceeds the nodes left after the sampling margin"
        )
    selected = interior[torch.randperm(len(interior), device=mesh.device)[: int(count)]]
    return grid[selected].unsqueeze(0).expand(len(mesh), -1, -1), mesh[:, selected]


def hmonet_main_loss(
    prediction,
    target,
    logical,
    fixed=False,
    point_weighting="position",
):
    """Return coordinate MSE with optional position-dependent point weights.

    Position mode preserves the original boundary-enhanced weighting. Uniform
    mode assigns every sampled point weight one. In 2-D the factor of two
    converts component-wise MSE to mean squared Euclidean distance per point;
    it is not a sampling-point weight. The 3-D loss scale remains unchanged.
    """
    point_weighting = str(point_weighting).lower()
    if point_weighting not in {"position", "uniform"}:
        raise ValueError("point_weighting must be position or uniform")
    if point_weighting == "position":
        coefficient = 32.0 if fixed else 16.0
        weight = 1.0 + coefficient * (
            (logical - 0.5) ** 4
        ).sum(dim=-1, keepdim=True)
    else:
        weight = 1.0
    factor = 2.0 if prediction.shape[-1] == 2 else 1.0
    return factor * ((prediction - target) * weight).square().mean()


def predict_coordinates(model, boundary, logical, geometry=None, **kwargs):
    """Evaluate all numerical samples in one network forward pass."""
    dimension = logical.shape[-1]
    from src.models.model import AttentionHMONet2D, AttentionHMONet3D

    if isinstance(model, (AttentionHMONet2D, AttentionHMONet3D)):
        if geometry is None:
            raise ValueError(
                "Attention-PI-HMONet requires exact parameter boundary geometry"
            )
        from scripts.train.attention_pihmonet_common import build_boundary_inputs

        boundary_inputs = build_boundary_inputs(
            geometry,
            tuple(model.declared_grid_shape),
            logical.device,
            logical.dtype,
            getattr(model, "boundary_sample_counts", None),
        )
        kwargs["boundary_geometry"] = geometry
        outputs = model(
            *boundary_inputs, *logical.unbind(dim=-1), **kwargs
        )
        prediction = torch.cat(outputs[:dimension], dim=-1)
        if kwargs.get("return_displacement", False):
            return prediction, outputs[-1]
        return prediction
    if geometry is not None:
        kwargs["boundary_geometry"] = geometry
    outputs = model(*boundary.unbind(dim=-1), *logical.unbind(dim=-1), **kwargs)
    prediction = torch.cat(outputs[:dimension], dim=-1)
    if kwargs.get("return_displacement", False):
        return prediction, outputs[-1]
    return prediction


def safe_optimizer_step(
    loss, model, optimizer, gradient_clip_norm=None, skip_nonfinite_loss=True
):
    """Apply the legacy loss/gradient checks before committing an Adam step."""
    optimizer.zero_grad(set_to_none=True)
    if not torch.isfinite(loss):
        if skip_nonfinite_loss:
            print("Warning: skipped a batch with non-finite loss", flush=True)
            return False
        raise FloatingPointError("training loss is NaN or Inf")
    loss.backward()
    if gradient_clip_norm is not None:
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(gradient_clip_norm)
        )
        finite = bool(torch.isfinite(gradient_norm))
    else:
        finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
    if not finite:
        optimizer.zero_grad(set_to_none=True)
        if skip_nonfinite_loss:
            print("Warning: skipped a batch with non-finite gradient norm", flush=True)
            return False
        raise FloatingPointError("gradient norm is NaN or Inf")
    optimizer.step()
    return True


class BatchLossTotals:
    """Accumulate successful batch means weighted by actual sample counts."""

    def __init__(self):
        self.sample_count = 0
        self.totals = {}

    def add(self, sample_count, **losses):
        self.sample_count += int(sample_count)
        for name, value in losses.items():
            scalar = float(value.detach()) if torch.is_tensor(value) else float(value)
            self.totals[name] = self.totals.get(name, 0.0) + scalar * sample_count

    def means(self):
        if self.sample_count == 0:
            raise FloatingPointError(
                "all training batches had non-finite loss or gradients"
            )
        return {name: total / self.sample_count for name, total in self.totals.items()}


def jacobian_positive_loss_autograd(prediction, logical, minimum):
    """Penalize mappings with a Jacobian determinant below ``minimum``."""
    grads = []
    for axis in range(prediction.shape[-1]):
        grads.append(
            torch.autograd.grad(
                prediction[..., axis].sum(),
                logical,
                create_graph=True,
                retain_graph=True,
            )[0]
        )
    jac = torch.stack(grads, dim=-2)
    determinant = torch.linalg.det(jac)
    return torch.relu(float(minimum) - determinant).square().mean()


def jacobian_positive_loss(prediction, logical, minimum):
    """Backward-compatible entry point for the original autograd penalty."""
    return jacobian_positive_loss_autograd(prediction, logical, minimum)


def _jacobian_fd_scheme(logical, step):
    """Use central differences inside and second-order one-sided differences near faces."""
    step = float(step)
    if not np.isfinite(step) or not 0 < step <= 1 / 3:
        raise ValueError("jacobian_fd_step must be finite and in (0, 1/3]")
    if logical.shape[-1] not in (2, 3):
        raise ValueError("Jacobian finite differences require 2-D or 3-D points")
    lower = logical < step
    upper = logical > 1 - step
    # At lower faces evaluate +h,+2h; at upper faces evaluate -h,-2h.
    # No derivative of parameter geometry with respect to coordinates is needed.
    direction = torch.where(upper, -torch.ones_like(logical), torch.ones_like(logical))
    one_sided = lower | upper
    first_offset = direction * step
    second_offset = torch.where(
        one_sided, 2 * direction * step, -torch.ones_like(logical) * step
    )
    w0 = torch.where(one_sided, -1.5 * direction / step, torch.zeros_like(logical))
    w1 = torch.where(
        one_sided, 2 * direction / step, torch.ones_like(logical) / (2 * step)
    )
    w2 = torch.where(
        one_sided, -0.5 * direction / step, -torch.ones_like(logical) / (2 * step)
    )
    return first_offset, second_offset, w0, w1, w2


def jacobian_finite_difference_queries(logical, step=1e-3):
    """Return [..., point, 1+2*d, d] batched queries, all within the unit domain.

    The center is followed by two points per coordinate axis. One-sided
    second-order stencils keep boundary/near-boundary queries in the domain.
    """
    first, second, _, _, _ = _jacobian_fd_scheme(logical, step)
    dimension = logical.shape[-1]
    eye = torch.eye(dimension, dtype=logical.dtype, device=logical.device)
    offsets = torch.stack((first, second), dim=-1)[..., None] * eye[:, None, :]
    neighbors = logical.unsqueeze(-2) + offsets.flatten(-3, -2)
    return torch.cat((logical.unsqueeze(-2), neighbors), dim=-2)


def jacobian_positive_loss_finite_difference(prediction, logical, minimum, step=1e-3):
    """Penalize the determinant computed from final mapped stencil coordinates.

    ``prediction`` corresponds to jacobian_finite_difference_queries(logical).
    Spatial derivatives use finite differences; gradients to model parameters
    remain attached. Coordinate components form matrix rows, axes form columns.
    """
    dimension = logical.shape[-1]
    if prediction.shape != (*logical.shape[:-1], 1 + 2 * dimension, dimension):
        raise ValueError(
            "Prediction shape must match the Jacobian finite-difference stencil"
        )
    _, _, w0, w1, w2 = _jacobian_fd_scheme(logical, step)
    center = prediction[..., 0, :].unsqueeze(-2)
    first = prediction[..., 1::2, :]
    second = prediction[..., 2::2, :]
    derivatives = w1[..., None] * (first - center) + w2[..., None] * (second - center)
    jacobian = derivatives.transpose(-1, -2)
    determinant = torch.linalg.det(jacobian)
    return torch.relu(float(minimum) - determinant).square().mean()


def normal_alignment_loss(direction, normal):
    """Mean squared cosine mismatch; ignore undefined reference normals.

    A zero predicted direction has loss one and remains in the objective.
    Reference normals are fixed geometry targets, not trainable quantities.
    """
    normal = normal.detach()
    length = torch.linalg.vector_norm(normal, dim=-1)
    valid = torch.isfinite(normal).all(dim=-1) & (length > 1e-12)
    if not bool(valid.any()):
        raise ValueError("No valid boundary normals are available for normal loss")
    target = torch.nn.functional.normalize(normal[valid], dim=-1, eps=1e-12)
    predicted = torch.nn.functional.normalize(direction[valid], dim=-1, eps=1e-12)
    cosine = (predicted * target).sum(dim=-1).clamp(-1.0, 1.0)
    return (1.0 - cosine).square().mean()


def boundary_normal_loss(
    model, boundary, mesh, shape, dimension, device, sample_count, offset, geometry=None
):
    """Align inward grid lines with exact inward unit boundary normals.

    The target is a rotated curve tangent in 2D or a surface cross product
    in 3D. PDE inward displacements determine only its sign, never its angle
    or magnitude. Without parametric geometry, mesh boundary tangents provide
    a piecewise-linear approximation. Existing offset controls the inward
    computational displacement, not a derivative magnitude target.
    """
    if dimension not in (2, 3) or int(sample_count) < 1:
        raise ValueError("Normal loss requires dimension 2/3 and positive sample_count")
    if not 0.0 < float(offset) < 0.5:
        raise ValueError(
            "boundary_normal_offset must lie strictly between zero and 0.5"
        )
    q = torch.rand(
        (boundary.shape[0], int(sample_count), dimension),
        device=device,
        dtype=boundary.dtype,
    )
    side = torch.randint(
        0, 2 * dimension, (boundary.shape[0], int(sample_count)), device=device
    )
    inner = q.clone()
    for axis in range(dimension):
        lower, upper = side == 2 * axis, side == 2 * axis + 1
        q[..., axis] = torch.where(lower, torch.zeros_like(q[..., axis]), q[..., axis])
        q[..., axis] = torch.where(upper, torch.ones_like(q[..., axis]), q[..., axis])
        inner[..., axis] = torch.where(
            lower, torch.full_like(inner[..., axis], float(offset)), inner[..., axis]
        )
        inner[..., axis] = torch.where(
            upper,
            torch.full_like(inner[..., axis], 1.0 - float(offset)),
            inner[..., axis],
        )
    kwargs = {"boundary_geometry": geometry} if geometry is not None else {}
    pred_q = torch.cat(
        model(
            *[boundary[..., i] for i in range(dimension)],
            *[q[..., i] for i in range(dimension)],
            **kwargs,
        ),
        dim=-1,
    )
    pred_i = torch.cat(
        model(
            *[boundary[..., i] for i in range(dimension)],
            *[inner[..., i] for i in range(dimension)],
            **kwargs,
        ),
        dim=-1,
    )
    if dimension == 2:
        nx, ny = shape
        values = mesh.reshape(len(mesh), ny, nx, 2).permute(0, 3, 1, 2)

        def interp(points):
            grid = (points * 2 - 1).reshape(len(mesh), -1, 1, 2)
            return (
                torch.nn.functional.grid_sample(
                    values, grid, align_corners=True, mode="bilinear"
                )
                .squeeze(-1)
                .permute(0, 2, 1)
            )

    else:
        nx, ny, nz = shape
        values = mesh.reshape(len(mesh), nz, ny, nx, 3).permute(0, 4, 1, 2, 3)

        def interp(points):
            grid = (points * 2 - 1).reshape(len(mesh), -1, 1, 1, 3)
            return (
                torch.nn.functional.grid_sample(
                    values, grid, align_corners=True, mode="bilinear"
                )
                .squeeze(-1)
                .squeeze(-1)
                .permute(0, 2, 1)
            )

    # Differentiate geometry only once. Its normals are fixed targets, so no
    # higher-order geometry graph is retained during network backpropagation.
    if dimension == 2 and hasattr(geometry, "tangents_torch"):
        with torch.no_grad():
            tangents = geometry.tangents_torch(*q.unbind(-1))
            tangent = tangents.gather(
                2, side[..., None, None].expand(-1, -1, 1, 2)
            ).squeeze(2)
            derivatives = tangent.new_zeros((*q.shape[:2], 2, 2))
            derivatives.scatter_(
                -1,
                (1 - side // 2)[..., None, None].expand(-1, -1, 2, 1),
                tangent[..., None],
            )
    else:
        with torch.enable_grad():
            query = q.detach().requires_grad_(True)
            if geometry is not None:
                from src.data.backend.parametric_torch import (
                    evaluate_boundary_curves_torch,
                    evaluate_boundary_surfaces_torch,
                )

                evaluator = (
                    evaluate_boundary_curves_torch
                    if dimension == 2
                    else evaluate_boundary_surfaces_torch
                )
                projections = evaluator(geometry, *query.unbind(-1))
                selected = projections.gather(
                    2, side[..., None, None].expand(-1, -1, 1, dimension)
                ).squeeze(2)
            else:
                selected = interp(query)
            derivatives = torch.stack(
                [
                    torch.autograd.grad(
                        selected[..., coordinate].sum(), query, retain_graph=True
                    )[0]
                    for coordinate in range(dimension)
                ],
                dim=-2,
            ).detach()
    axis = side // 2
    if dimension == 2:
        tangent = derivatives.gather(
            -1, (1 - axis)[..., None, None].expand(-1, -1, 2, 1)
        ).squeeze(-1)
        normal = torch.stack((-tangent[..., 1], tangent[..., 0]), dim=-1)
    else:
        # Face-local parameter axes: (eta,zeta), (xi,zeta), (xi,eta).
        first = torch.where(axis == 0, 1, 0)
        second = torch.where(axis == 2, 1, 2)
        tangents = [
            derivatives.gather(-1, index[..., None, None].expand(-1, -1, 3, 1)).squeeze(
                -1
            )
            for index in (first, second)
        ]
        normal = torch.linalg.cross(*tangents, dim=-1)
    with torch.no_grad():
        reference_inward = interp(inner) - interp(q)
        orientation = (normal * reference_inward).sum(dim=-1)
        normal = torch.where((orientation < 0)[..., None], -normal, normal)
        # Degenerate reference cells cannot determine an inward orientation.
        normal = torch.where((orientation.abs() > 1e-12)[..., None], normal, 0.0)
    return normal_alignment_loss(pred_i - pred_q, normal)


def write_history(path: Path, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def calculate_fixed_grid_metrics(
    model,
    datasets,
    shape,
    device,
    geometry_getter=None,
    batch_size=64,
    calculate_point_errors=False,
):
    """Measure twice unweighted coordinate MSE on all fixed interior nodes.

    The multiplier is 2 in both dimensions. Point-distance statistics retain
    their physical units and do not receive this MSE multiplier.
    """
    grid, indices = logical_grid_and_interior_indices(shape, device)
    points = grid[indices]
    squared_sum = 0.0
    coordinate_count = 0
    point_errors = []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for dataset in datasets:
                for start in range(0, len(dataset), batch_size):
                    stop = min(start + batch_size, len(dataset))
                    boundary = torch.as_tensor(
                        np.asarray(dataset.boundary_points[start:stop]),
                        dtype=torch.float32,
                        device=device,
                    )
                    target = torch.as_tensor(
                        np.asarray(dataset.mesh_points[start:stop]),
                        dtype=torch.float32,
                        device=device,
                    ).reshape(stop - start, -1, len(shape))[:, indices]
                    geometry = None
                    if geometry_getter is not None:
                        geometry = [
                            geometry_getter(dataset, i) for i in range(start, stop)
                        ]
                    from src.models.model import AttentionHMONet2D

                    if (
                        geometry is not None
                        and len(shape) == 2
                        and not isinstance(model, AttentionHMONet2D)
                    ):
                        from src.data.geometry.parametric_curve_2d import BoundaryCurveSet2D

                        geometry = BoundaryCurveSet2D.batch(
                            geometry, device, boundary.dtype
                        )
                    query = points.unsqueeze(0).expand(stop - start, -1, -1)
                    prediction = predict_coordinates(model, boundary, query, geometry)
                    difference = prediction - target
                    if not torch.isfinite(difference).all():
                        raise FloatingPointError(
                            "Non-finite prediction during fixed-grid MSE"
                        )
                    squared_sum += difference.square().sum().item()
                    coordinate_count += target.numel()
                    if calculate_point_errors:
                        point_errors.append(
                            torch.linalg.vector_norm(difference, dim=-1).cpu()
                        )
    finally:
        model.train(was_training)
    if coordinate_count == 0:
        raise ValueError("Fixed-grid MSE requires a nonempty dataset")
    metrics = {"mse": FIXED_GRID_MSE_SCALE * squared_sum / coordinate_count}
    if point_errors:
        values = torch.cat(point_errors).reshape(-1)
        metrics["max_point_error"] = values.max().item()
        metrics["p99_point_error"] = torch.quantile(values, 0.99).item()
    return metrics


def calculate_fixed_grid_mse_2d(
    model, datasets, nx, ny, device, geometry_getter=None, batch_size=64
):
    """Return twice 2-D coordinate MSE over fixed interior nodes only."""
    return calculate_fixed_grid_metrics(
        model, datasets, (nx, ny), device, geometry_getter, batch_size
    )["mse"]


def calculate_fixed_grid_mse_3d(
    model, datasets, nx, ny, nz, device, geometry_getter=None, batch_size=64
):
    """Return twice 3-D coordinate MSE over fixed interior nodes only."""
    return calculate_fixed_grid_metrics(
        model, datasets, (nx, ny, nz), device, geometry_getter, batch_size
    )["mse"]


def write_mse_outputs(output_dir, epochs, values, title):
    """Save fixed-grid MSE history as CSV and a line plot."""
    import csv

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "mse_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "dataset_mse"])
        writer.writerows(zip(epochs, values))
    if not values:
        return
    try:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(epochs, values, linewidth=1.2)
        axis.set_yscale("log")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Full-dataset MSE")
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(output_dir / "dataset_mse.png", dpi=200)
        plt.close(figure)
    except ImportError:
        pass


def write_loss_outputs(output_dir, history, title):
    """Save the legacy loss figure; training_history.csv is written by state."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if history:
        try:
            import matplotlib.pyplot as plt

            figure, axis = plt.subplots(figsize=(8, 5))
            axis.plot(
                [item["epoch"] for item in history],
                [item["train_loss"] for item in history],
                linewidth=1.2,
            )
            axis.set_yscale("log")
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Training loss")
            axis.set_title(title)
            axis.grid(True, alpha=0.3)
            figure.tight_layout()
            figure.savefig(output_dir / "training_loss.png", dpi=200)
            plt.close(figure)
        except ImportError:
            pass


class DualSummaryWriter:
    """Write matching TensorBoard events to two independent run directories."""

    def __init__(self, primary, mirror):
        self.writers = (primary, mirror)
        self.log_dir = primary.log_dir
        self.mirror_log_dir = mirror.log_dir
        self.closed = False

    def add_scalar(self, tag, value, global_step=None, walltime=None, **kwargs):
        import time

        if self.closed:
            raise RuntimeError("TensorBoard writer is closed")
        walltime = time.time() if walltime is None else walltime
        for writer in self.writers:
            writer.add_scalar(tag, value, global_step, walltime=walltime, **kwargs)

    def add_text(self, tag, text_string, global_step=None, walltime=None):
        import time

        if self.closed:
            raise RuntimeError("TensorBoard writer is closed")
        walltime = time.time() if walltime is None else walltime
        for writer in self.writers:
            writer.add_text(tag, text_string, global_step, walltime=walltime)

    def flush(self):
        if not self.closed:
            for writer in self.writers:
                writer.flush()

    def close(self):
        # Training completion and the trainer's finally block may both close.
        if not self.closed:
            self.closed = True
            try:
                self.writers[0].close()
            finally:
                self.writers[1].close()


def create_summary_writer(output_dir, enabled, mirror_dir=None, primary_enabled=True):
    """Create an optional TensorBoard writer without making it mandatory."""
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return None
    if not primary_enabled:
        if mirror_dir is None:
            raise ValueError("mirror_dir is required when primary TensorBoard output is disabled")
        return SummaryWriter(log_dir=str(mirror_dir))
    primary = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    if mirror_dir is None:
        return primary
    try:
        mirror = SummaryWriter(log_dir=str(mirror_dir))
    except Exception:
        primary.close()
        raise
    return DualSummaryWriter(primary, mirror)


def update_learning_rate(
    optimizer,
    completed_epochs,
    warmup_epochs,
    update_interval,
    gamma=0.9,
    stop_below=1e-5,
):
    """Apply the configured post-warmup multiplicative learning-rate decay."""
    if completed_epochs <= warmup_epochs or (completed_epochs - warmup_epochs) % max(
        1, update_interval
    ):
        return
    for group in optimizer.param_groups:
        if group["lr"] >= stop_below:
            group["lr"] = group["lr"] * gamma


def metric_window_convergence_status(
    history,
    window,
    check_stability,
    fluctuation_tolerance,
    check_best_improvement,
    improvement_tolerance,
    reference_best=None,
    stale=0,
    patience=1,
    quantiles=(0.1, 0.9),
    check_absolute_stability=False,
    absolute_tolerance=5e-6,
    check_near_best=False,
    near_best_relative_tolerance=0.2,
    near_best_absolute_tolerance=1e-7,
    check_consecutive_stability=False,
    stable_count=0,
    stable_required=50,
):
    """Evaluate plateau stability and significant-improvement conditions."""
    if len(history) < max(2, window):
        return None
    recent = np.asarray(history[-window:], dtype=float)
    lower, upper = np.quantile(recent, quantiles)
    median = float(np.median(recent))
    scale = max(abs(median), 1e-12)
    fluctuation = float((upper - lower) / scale)
    window_min = float(np.min(recent))
    if reference_best is None:
        reference_best = float(np.min(history))
    relative_improvement = float(
        (reference_best - window_min) / max(abs(reference_best), 1e-12)
    )
    significant = (
        check_best_improvement and relative_improvement > improvement_tolerance
    )
    if significant:
        reference_best, stale = window_min, 0
    elif check_best_improvement:
        stale += 1
    absolute_fluctuation = float(upper - lower)
    relative_stable = (not check_stability) or fluctuation <= fluctuation_tolerance
    absolute_stable = (
        not check_absolute_stability
    ) or absolute_fluctuation <= absolute_tolerance
    stable = relative_stable and absolute_stable
    if not check_best_improvement:
        stale = stale + 1 if stable else 0
    # Use the actual best observed value, not the significant-improvement anchor.
    historical_best = float(np.min(history))
    near_best_limit = historical_best + max(
        near_best_absolute_tolerance,
        near_best_relative_tolerance * abs(historical_best),
    )
    near_best = (not check_near_best) or median <= near_best_limit
    stable_count = stable_count + 1 if stable else 0
    consecutive_passes = (
        not check_consecutive_stability
    ) or stable_count >= stable_required
    enabled = (
        check_stability
        or check_absolute_stability
        or check_best_improvement
        or check_near_best
        or check_consecutive_stability
    )
    return {
        "median": median,
        "lower_quantile": float(quantiles[0]),
        "upper_quantile": float(quantiles[1]),
        "lower_quantile_value": lower,
        "upper_quantile_value": upper,
        "relative_fluctuation": fluctuation,
        "absolute_fluctuation": absolute_fluctuation,
        "relative_stability_passes": relative_stable,
        "absolute_stability_passes": absolute_stable,
        "window_min": window_min,
        "reference_best": reference_best,
        "relative_best_improvement": relative_improvement,
        "significant_improvement": significant,
        "stale": stale,
        "epochs_without_significant_improvement": stale,
        "stability_passes": stable,
        "best_improvement_passes": stale >= patience,
        "historical_best": historical_best,
        "near_best_limit": near_best_limit,
        "near_best_passes": near_best,
        "consecutive_stable_count": stable_count,
        "consecutive_stability_passes": consecutive_passes,
        "passes": enabled
        and stable
        and stale >= patience
        and near_best
        and consecutive_passes,
    }


def simplified_plateau_status(
    history,
    window,
    patience,
    relative_tolerance,
    absolute_minimum,
    absolute_maximum,
    improvement_tolerance,
    near_relative,
    near_absolute,
    previous=None,
    quantiles=(0.1, 0.9),
    stable_required=None,
):
    """Track improvement patience and short-window stability independently.

    The lower window quantile detects material improvement, while the window
    median determines whether the current state remains near the historical
    best. A significant improvement resets only the no-improvement counter.
    Short-window instability resets only the consecutive-stability counter.
    """
    if len(history) < window:
        return None
    recent = np.asarray(history[-window:], dtype=float)
    lower, upper = np.quantile(recent, quantiles)
    lower = float(lower)
    upper = float(upper)
    median = float(np.median(recent))
    previous = previous or {}
    best_median = min(median, previous.get("historical_best", median))
    anchor = previous.get("reference_best", lower)
    improvement = (anchor - lower) / max(abs(anchor), 1e-12)
    significant = improvement > improvement_tolerance
    if significant:
        anchor = lower
    width = float(upper - lower)
    relative_limit = relative_tolerance * abs(median)
    limit = float(np.clip(relative_limit, absolute_minimum, absolute_maximum))
    stable = bool(np.isfinite(recent).all() and width <= limit)
    near_limit = best_median + max(
        near_absolute, near_relative * abs(best_median)
    )
    near = median <= near_limit
    no_improvement_count = (
        previous.get("epochs_without_significant_improvement", 0) + 1
        if previous and not significant
        else 0
    )
    stable_count = (
        previous.get("consecutive_stable_count", 0) + 1
        if previous and stable
        else 0
    )
    stable_required = int(patience if stable_required is None else stable_required)
    no_improvement_passes = no_improvement_count >= int(patience)
    short_stability_passes = stable_count >= stable_required
    short_branch_passes = bool(
        near and no_improvement_passes and short_stability_passes
    )
    return {
        "median": median,
        "lower_quantile": quantiles[0],
        "upper_quantile": quantiles[1],
        "lower_quantile_value": float(lower),
        "upper_quantile_value": float(upper),
        "relative_fluctuation": width / max(abs(median), 1e-12),
        "absolute_fluctuation": width,
        "relative_fluctuation_limit": relative_limit,
        "fluctuation_absolute_minimum": absolute_minimum,
        "fluctuation_absolute_maximum": absolute_maximum,
        "fluctuation_limit": limit,
        "window_min": float(np.min(recent)),
        "reference_best": anchor,
        "historical_best": best_median,
        "historical_best_basis": "window_median",
        "improvement_basis": "window_lower_quantile",
        "relative_best_improvement": improvement,
        "significant_improvement": significant,
        "stability_passes": stable,
        "near_best_limit": near_limit,
        "near_best_passes": near,
        "stale": no_improvement_count,
        "epochs_without_significant_improvement": no_improvement_count,
        "no_improvement_passes": no_improvement_passes,
        "consecutive_stable_count": stable_count,
        "short_stability_required": stable_required,
        "short_stability_passes": short_stability_passes,
        "short_branch_passes": short_branch_passes,
        "passes": short_branch_passes,
    }
