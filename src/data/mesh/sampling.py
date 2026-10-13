"""Uniform logical/boundary sampling and structured-grid interpolation."""

import numpy as np
import torch

from . import topology as quad_mesh
from .topology import GenerateQuadMeshBoundaryIndex


def GenerateRectangleBoundaryDataRandom(
    x_min=-0.5, x_max=0.5, y_min=-0.5, y_max=0.5, Ny=100, Nx=100, sorted=False
):
    """Generate random points independently on the four rectangle edges."""
    Ny = int(Ny)
    Nx = int(Nx)
    bottom = np.random.uniform(low=x_min, high=x_max, size=Nx)
    if sorted:
        bottom = np.sort(bottom)
    right = np.random.uniform(low=y_min, high=y_max, size=Ny)
    if sorted:
        right = np.sort(right)
    top = np.random.uniform(low=x_min, high=x_max, size=Nx)
    if sorted:
        top = np.sort(top)[::-1]
    left = np.random.uniform(low=y_min, high=y_max, size=Ny)
    if sorted:
        left = np.sort(left)[::-1]

    boundary_points_random = np.vstack(
        [
            np.stack((bottom, np.full_like(bottom, y_min)), axis=1),
            np.stack((np.full_like(right, x_max), right), axis=1),
            np.stack((top, np.full_like(top, y_max)), axis=1),
            np.stack((np.full_like(left, x_min), left), axis=1),
        ]
    )
    return boundary_points_random


def GenerateRectangleInteriorPointsRandom(
    x_min=-0.5, x_max=0.5, y_min=-0.5, y_max=0.5, num=100
):
    """Generate ``num`` uniformly random points inside a rectangle."""
    x = np.random.uniform(low=x_min, high=x_max, size=num)
    # interior_xi = np.sort(interior_xi)
    y = np.random.uniform(low=y_min, high=y_max, size=num)
    # interior_eta = np.sort(interior_eta)
    points_random = np.stack((x, y), axis=1)
    return points_random


def GenerateRectangleInteriorPointsUniform(
    x_min=-0.5, x_max=0.5, y_min=-0.5, y_max=0.5, num=100
):
    """Generate a ``num`` by ``num`` Cartesian grid inside a rectangle."""
    x = np.linspace(x_min, x_max, num)
    y = np.linspace(y_min, y_max, num)
    points_uniform = np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 2)
    return points_uniform


def GenerateRectanglePointsGrid(
    x_min=-0.5, x_max=0.5, y_min=-0.5, y_max=0.5, Ny=100, Nx=100
):
    """Generate a row-major ``Ny`` by ``Nx`` rectangular point grid."""
    Ny = int(Ny)
    Nx = int(Nx)
    interior_xi = np.linspace(x_min, x_max, Nx)
    interior_eta = np.linspace(y_min, y_max, Ny)
    interior_points_grid = np.stack(
        np.meshgrid(interior_xi, interior_eta), axis=-1
    ).reshape(-1, 2)
    return interior_points_grid


def sample_rectangle_boundary_uniform(Nx, Ny, num_samples=None):
    """Uniformly sample the unit-square boundary in mesh boundary order.

    Returns ``(points, sample_indices)``. ``points`` are computational-domain
    coordinates and ``sample_indices`` index the input physical boundary, whose
    ordering is bottom, right, top, left without duplicated corner points.
    When ``num_samples`` is omitted or is no smaller than the boundary size,
    all boundary points are returned.
    """
    Nx, Ny = int(Nx), int(Ny)
    if Nx < 2 or Ny < 2:
        raise ValueError("Nx and Ny must both be at least 2")
    boundary_indices = quad_mesh.GenerateQuadMeshBoundaryIndex(Ny, Nx)
    boundary_size = len(boundary_indices)
    if num_samples is None:
        num_samples = boundary_size
    num_samples = int(num_samples)
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if num_samples >= boundary_size:
        sample_indices = np.arange(boundary_size, dtype=int)
    else:
        # Equally spaced positions around the non-duplicated closed boundary.
        sample_indices = np.floor(
            np.arange(num_samples) * boundary_size / num_samples
        ).astype(int)
    grid = GenerateRectanglePointsGrid(0.0, 1.0, 0.0, 1.0, Ny=Ny, Nx=Nx)
    return grid[boundary_indices[sample_indices]], sample_indices


def sample_fixed_grid_interior_points(grid, nx, ny, count, batch_size, device=None):
    """Randomly select fixed-grid interior nodes independently per sample.

    Returns ``(points, indices)`` with shapes ``[B, N, 2]`` and ``[B, N]``.
    Sampling is vectorized across the batch and excludes all boundary nodes.
    """
    grid = torch.as_tensor(grid, dtype=torch.float32, device=device)
    mask = torch.ones(nx * ny, dtype=torch.bool, device=grid.device)
    mask[:nx] = mask[(ny - 1) * nx :] = False
    mask[::nx] = mask[nx - 1 :: nx] = False
    interior = torch.arange(nx * ny, device=grid.device)[mask]
    count = min(int(count), len(interior))
    if count < 1:
        raise ValueError("fixed-grid sample count must be positive")
    positions = (
        torch.rand(batch_size, len(interior), device=grid.device)
        .topk(count, dim=1, largest=False)
        .indices
    )
    indices = interior[positions]
    return grid[indices], indices


def sample_uniform_global_interior_points(
    batch_size, count, device=None, eps=1e-6, margin=0.0
):
    """Sample uniform points in the open or inset unit square for a batch."""
    if int(count) < 1:
        raise ValueError("global random sample count must be positive")
    margin = float(margin)
    if not 0.0 <= margin < 0.5:
        raise ValueError("margin must lie in [0, 0.5)")
    lower = margin if margin > 0 else float(eps)
    upper = 1.0 - lower
    return torch.empty(batch_size, int(count), 2, device=device).uniform_(lower, upper)


def sample_curvature_distance_adaptive_points(
    boundary_curvature,
    computational_boundary,
    count,
    candidate_multiplier=10,
    curvature_power=1.0,
    sigma_min=0.02,
    sigma_max=0.10,
    uniform_probability=0.05,
    eps=1.0e-8,
    margin=0.0,
):
    r"""Draw interior points near high-curvature physical boundaries.

    A uniform candidate ``q`` receives the unnormalized score

    ``u + sum_i (eps + k_i**p) exp(-||q-g_i||^2/(2*sigma_i^2))``,

    where ``g_i`` is the matching computational-boundary point, normalized
    curvature ``k_i`` lies in ``[0, 1]``, and high curvature uses the narrower
    ``sigma_i = sigma_max - (sigma_max - sigma_min) * k_i``.
    Sampling is fully batched and uses no Python loop over samples or points.
    """
    curvature = torch.as_tensor(boundary_curvature)
    if curvature.ndim == 1:
        curvature = curvature.unsqueeze(0)
    computational = torch.as_tensor(
        computational_boundary, dtype=curvature.dtype, device=curvature.device
    )
    if curvature.ndim != 2 or computational.shape != (curvature.shape[1], 2):
        raise ValueError(
            "boundary_curvature [B,P] and computational_boundary [P,2] must match"
        )
    count = int(count)
    candidate_multiplier = int(candidate_multiplier)
    if count < 1 or candidate_multiplier < 1:
        raise ValueError("count and candidate_multiplier must be positive")
    if not 0 < float(sigma_min) <= float(sigma_max):
        raise ValueError("require 0 < sigma_min <= sigma_max")
    candidate_count = max(count, count * candidate_multiplier)
    candidates = sample_uniform_global_interior_points(
        curvature.shape[0], candidate_count, device=curvature.device, margin=margin
    )
    distance_squared = (
        candidates[:, :, None, :] - computational[None, None, :, :]
    ).square().sum(dim=-1)
    sigma = float(sigma_max) - (float(sigma_max) - float(sigma_min)) * curvature
    weights = float(eps) + curvature.pow(float(curvature_power))
    scores = float(uniform_probability) + (
        weights[:, None, :]
        * torch.exp(-distance_squared / (2.0 * sigma[:, None, :].square()))
    ).sum(dim=-1)
    probabilities = scores / scores.sum(dim=1, keepdim=True)
    indices = torch.multinomial(probabilities, count, replacement=False)
    return torch.gather(candidates, 1, indices[..., None].expand(-1, -1, 2))


def GenerateCustomGridPoints(x_points, y_points):
    """Return the Cartesian product of explicit x and y coordinate arrays."""
    interior_points_grid = np.stack(np.meshgrid(x_points, y_points), axis=-1).reshape(
        -1, 2
    )
    return interior_points_grid


def find_index_in_com_domain(internal_pts, Nx=20, Ny=20):
    """Documentation string."""
    d_xi = 1.0 / (Nx - 1)
    d_eta = 1.0 / (Ny - 1)
    xi0 = np.floor(internal_pts[..., 0] / d_xi).astype(int)
    eta0 = np.floor(internal_pts[..., 1] / d_eta).astype(int)
    xi1 = np.clip(xi0 + 1, 0, Nx - 1)
    eta1 = np.clip(eta0 + 1, 0, Ny - 1)
    v0 = eta0 * Nx + xi0
    v1 = eta0 * Nx + xi1
    v2 = eta1 * Nx + xi1
    v3 = eta1 * Nx + xi0
    quad_vtx_idx = np.stack([v0, v1, v2, v3], axis=-1)  # [B, N, 4]
    return quad_vtx_idx


def get_quad_vtx_coords(mesh_pts, quad_vtx_idx):
    """Documentation string."""
    mesh_pts = np.asarray(mesh_pts)
    quad_vtx_idx = np.asarray(quad_vtx_idx)

    squeeze_batch = False
    if mesh_pts.ndim == 2:
        mesh_pts = mesh_pts[None, ...]
        squeeze_batch = True
    if quad_vtx_idx.ndim == 2:
        quad_vtx_idx = quad_vtx_idx[None, ...]
        squeeze_batch = True

    if mesh_pts.shape[0] == 1 and quad_vtx_idx.shape[0] > 1:
        mesh_pts = np.repeat(mesh_pts, quad_vtx_idx.shape[0], axis=0)
    if quad_vtx_idx.shape[0] == 1 and mesh_pts.shape[0] > 1:
        quad_vtx_idx = np.repeat(quad_vtx_idx, mesh_pts.shape[0], axis=0)
    if mesh_pts.shape[0] != quad_vtx_idx.shape[0]:
        raise ValueError(
            "mesh_pts and quad_vtx_idx must share the same batch size after broadcasting"
        )

    quad_vtx_coords = np.take_along_axis(
        mesh_pts[:, None, :, :], quad_vtx_idx[..., None], axis=2
    )
    if squeeze_batch and quad_vtx_coords.shape[0] == 1:
        quad_vtx_coords = quad_vtx_coords[0]
    return quad_vtx_coords


def interpolate_physical_values(com_quads, phy_quads, internal_pts):
    """Documentation string."""
    com_quads = np.asarray(com_quads)
    phy_quads = np.asarray(phy_quads)
    internal_pts = np.asarray(internal_pts)

    squeeze_batch = False
    if com_quads.ndim == 3:
        com_quads = com_quads[None, ...]
        squeeze_batch = True
    if phy_quads.ndim == 3:
        phy_quads = phy_quads[None, ...]
        squeeze_batch = True
    if internal_pts.ndim == 2:
        internal_pts = internal_pts[None, ...]
    if internal_pts.shape[0] == 1 and com_quads.shape[0] > 1:
        internal_pts = np.repeat(internal_pts, com_quads.shape[0], axis=0)
    if com_quads.shape[0] == 1 and internal_pts.shape[0] > 1:
        com_quads = np.repeat(com_quads, internal_pts.shape[0], axis=0)
    if phy_quads.shape[0] == 1 and internal_pts.shape[0] > 1:
        phy_quads = np.repeat(phy_quads, internal_pts.shape[0], axis=0)
    if (
        com_quads.shape[0] != phy_quads.shape[0]
        or com_quads.shape[0] != internal_pts.shape[0]
    ):
        raise ValueError(
            "com_quads, phy_quads and internal_pts must share the same batch size after broadcasting"
        )

    eps = 1e-12

    def gather_v(verts, idx):
        """Gather one vertex per batch, query and local vertex index."""
        return np.take_along_axis(verts, idx[..., None, None], axis=2).squeeze(axis=2)

    def barycentric_weights(p, a, b, c):
        """Return barycentric weights and inside flags for batched triangles."""
        v0 = b - a
        v1 = c - a
        v2 = p - a
        den = v0[..., 0] * v1[..., 1] - v1[..., 0] * v0[..., 1]
        den_safe = np.where(np.abs(den) < eps, eps, den)

        w1 = (v2[..., 0] * v1[..., 1] - v1[..., 0] * v2[..., 1]) / den_safe
        w2 = (v0[..., 0] * v2[..., 1] - v2[..., 0] * v0[..., 1]) / den_safe
        w0 = 1.0 - w1 - w2
        w = np.stack([w0, w1, w2], axis=-1)

        inside = np.all(w >= -1e-9, axis=-1) & np.all(w <= 1.0 + 1e-9, axis=-1)
        return w, inside

    e = np.roll(com_quads, -1, axis=2) - com_quads
    e_next = np.roll(e, -1, axis=2)
    cross = e[..., 0] * e_next[..., 1] - e[..., 1] * e_next[..., 0]  # [B, N, 4]
    signs = np.sign(cross)
    major_sign = np.where(np.sum(signs, axis=-1) >= 0, 1.0, -1.0)
    is_concave_corner = signs * major_sign[..., None] < 0
    has_concave = np.any(is_concave_corner, axis=-1)

    concave_cross_idx = np.argmax(is_concave_corner, axis=-1)
    concave_v = (concave_cross_idx + 1) % 4

    diag_a = np.where(has_concave, concave_v, 0)
    diag_c = np.where(has_concave, (concave_v + 2) % 4, 2)

    diag_a, diag_c = np.broadcast_arrays(diag_a, diag_c)
    all_idx = np.arange(4)[None, None, :]
    rem_mask = (all_idx != diag_a[..., None]) & (all_idx != diag_c[..., None])
    rem = np.where(rem_mask, all_idx, -1)
    rem = np.sort(rem, axis=-1)[..., -2:]
    r0 = rem[..., 0]
    r1 = rem[..., 1]

    tri1_idx = np.stack([diag_a, r0, diag_c], axis=-1)
    tri2_idx = np.stack([diag_a, diag_c, r1], axis=-1)

    com_t1 = np.take_along_axis(com_quads, tri1_idx[..., None], axis=2)
    com_t2 = np.take_along_axis(com_quads, tri2_idx[..., None], axis=2)
    phy_t1 = np.take_along_axis(phy_quads, tri1_idx[..., None], axis=2)
    phy_t2 = np.take_along_axis(phy_quads, tri2_idx[..., None], axis=2)

    w1, inside1 = barycentric_weights(
        internal_pts, com_t1[..., 0, :], com_t1[..., 1, :], com_t1[..., 2, :]
    )
    w2, inside2 = barycentric_weights(
        internal_pts, com_t2[..., 0, :], com_t2[..., 1, :], com_t2[..., 2, :]
    )

    interp1 = np.sum(w1[..., :, None] * phy_t1, axis=-2)
    interp2 = np.sum(w2[..., :, None] * phy_t2, axis=-2)

    use_tri1 = inside1 | (~inside2)
    interpolated_values = np.where(use_tri1[..., None], interp1, interp2)
    if squeeze_batch and interpolated_values.shape[0] == 1:
        interpolated_values = interpolated_values[0]
    return interpolated_values


def interpolate_physical_values1(com_quads, phy_quads, internal_pts):
    """Documentation string."""
    com_quads = np.asarray(com_quads)
    phy_quads = np.asarray(phy_quads)
    internal_pts = np.asarray(internal_pts)

    squeeze_batch = False
    if com_quads.ndim == 3:
        com_quads = com_quads[None, ...]
        squeeze_batch = True
    if phy_quads.ndim == 3:
        phy_quads = phy_quads[None, ...]
        squeeze_batch = True
    if internal_pts.ndim == 2:
        internal_pts = internal_pts[None, ...]

    if internal_pts.shape[0] == 1 and com_quads.shape[0] > 1:
        internal_pts = np.repeat(internal_pts, com_quads.shape[0], axis=0)
    if com_quads.shape[0] == 1 and internal_pts.shape[0] > 1:
        com_quads = np.repeat(com_quads, internal_pts.shape[0], axis=0)
    if phy_quads.shape[0] == 1 and internal_pts.shape[0] > 1:
        phy_quads = np.repeat(phy_quads, internal_pts.shape[0], axis=0)
    if (
        com_quads.shape[0] != phy_quads.shape[0]
        or com_quads.shape[0] != internal_pts.shape[0]
    ):
        raise ValueError(
            "com_quads, phy_quads and internal_pts must share the same batch size after broadcasting"
        )

    eps = 1e-12
    x0 = com_quads[..., 0, 0]
    x1 = com_quads[..., 1, 0]
    y0 = com_quads[..., 0, 1]
    y3 = com_quads[..., 3, 1]

    denom_x = np.where(np.abs(x1 - x0) < eps, eps, x1 - x0)
    denom_y = np.where(np.abs(y3 - y0) < eps, eps, y3 - y0)
    u = (internal_pts[..., 0] - x0) / denom_x
    v = (internal_pts[..., 1] - y0) / denom_y

    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)

    w0 = (1.0 - u) * (1.0 - v)
    w1 = u * (1.0 - v)
    w2 = u * v
    w3 = (1.0 - u) * v
    weights = np.stack([w0, w1, w2, w3], axis=-1)

    interpolated_values = np.sum(weights[..., None] * phy_quads, axis=-2)
    if squeeze_batch and interpolated_values.shape[0] == 1:
        interpolated_values = interpolated_values[0]
    return interpolated_values


def logical_grid(shape):
    """Return x-fastest points: index=(k*ny+j)*nx+i in 3-D."""
    if len(shape) not in (2, 3) or min(shape) < 2:
        raise ValueError("grid_shape must contain two or three counts >= 2")
    axes = [np.linspace(0.0, 1.0, int(n)) for n in shape]
    coordinates = np.meshgrid(*axes[::-1], indexing="ij")
    return np.stack(coordinates[::-1], axis=-1).reshape(-1, len(shape))


def boundary_indices(shape):
    """Use the same boundary ordering as each dimension's training loader."""
    if len(shape) == 2:
        nx, ny = shape
        return np.asarray(GenerateQuadMeshBoundaryIndex(ny, nx), dtype=int)
    nx, ny, nz = shape
    mask = np.zeros((nz, ny, nx), dtype=bool)
    mask[[0, -1], :, :] = True
    mask[:, [0, -1], :] = True
    mask[:, :, [0, -1]] = True
    return np.flatnonzero(mask.reshape(-1))


def sample_faces(boundary, shape, tolerance=1e-8):
    """Sample six faces and reject incompatible shared edges/corners."""
    nx, ny, nz = map(int, shape)
    counts = ((ny, nz), (ny, nz), (nx, nz), (nx, nz), (nx, ny), (nx, ny))
    locations = (
        (slice(None), slice(None), 0),
        (slice(None), slice(None), -1),
        (slice(None), 0, slice(None)),
        (slice(None), -1, slice(None)),
        (0, slice(None), slice(None)),
        (-1, slice(None), slice(None)),
    )
    full = np.full((nz, ny, nx, 3), np.nan)
    faces = {}
    for name, count, location in zip(boundary.names, counts, locations):
        values = np.asarray(getattr(boundary, name).sample(*count)).transpose(1, 0, 2)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite coordinates on surface {name}")
        previous = full[location]
        occupied = np.isfinite(previous).all(axis=-1)
        if occupied.any() and not np.allclose(
            previous[occupied], values[occupied], atol=tolerance, rtol=tolerance
        ):
            error = float(np.max(np.abs(previous[occupied] - values[occupied])))
            raise ValueError(f"Surface {name} has inconsistent shared edges: {error:g}")
        full[location] = values
        faces[name] = values
    return faces, full


def sample_boundary(boundary, shape):
    """Evaluate precise boundaries at uniform logical parameters, including ends."""
    if len(shape) == 2:
        nx, ny = shape
        boundary.validate(tol=1e-8)
        return np.asarray(
            boundary.sample_closed(dict(bottom=nx, right=ny, top=nx, left=ny))[:-1],
            dtype=float,
        )
    _, full = sample_faces(boundary, shape)
    return full.reshape(-1, 3)[boundary_indices(shape)]
