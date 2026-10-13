"""TFI anchor targets and validity masks for PI-HMONet training."""

import math
import torch


def central_anchor_queries(dimension, points_per_axis, margin, *, device, dtype):
    """Return a fixed tensor-product grid strictly inside the unit domain."""
    if dimension not in (2, 3):
        raise ValueError("Anchor queries support two or three dimensions")
    n = int(points_per_axis)
    if n < 1 or not 0.0 < float(margin) < 0.5:
        raise ValueError("points_per_axis must be positive and margin must be in (0, 0.5)")
    axis = torch.linspace(float(margin), 1.0 - float(margin), n, device=device, dtype=dtype)
    return torch.stack(torch.meshgrid(*([axis] * dimension), indexing="ij"), -1).reshape(-1, dimension)


def _interp1(edge, t):
    batch, count = edge.shape[:2]
    x = t.clamp(0, 1) * (count - 1)
    i = x.floor().long().clamp(0, count - 2)
    f = x - i
    b = torch.arange(batch, device=edge.device)[:, None].expand_as(i)
    return (1 - f)[..., None] * edge[b, i] + f[..., None] * edge[b, i + 1]


def _interp2(face, a, b):
    batch, height, width = face.shape[:3]
    x = a.clamp(0, 1) * (width - 1)
    y = b.clamp(0, 1) * (height - 1)
    i, j = x.floor().long().clamp(0, width - 2), y.floor().long().clamp(0, height - 2)
    fx, fy = x - i, y - j
    batch_ids = torch.arange(batch, device=face.device)[:, None].expand_as(i)
    return (
        ((1 - fx) * (1 - fy))[..., None] * face[batch_ids, j, i]
        + (fx * (1 - fy))[..., None] * face[batch_ids, j, i + 1]
        + ((1 - fx) * fy)[..., None] * face[batch_ids, j + 1, i]
        + (fx * fy)[..., None] * face[batch_ids, j + 1, i + 1]
    )


def _boundary_grid(boundary, shape):
    dimension = len(shape)
    if dimension == 2:
        nx, ny = shape
        grid = boundary.new_zeros((boundary.shape[0], ny * nx, 2))
        ids = torch.arange(nx * ny, device=boundary.device).reshape(ny, nx)
        # The 2-D dataset stores the perimeter counter-clockwise: bottom,
        # right, reversed top, reversed left. Assign with that same ordering.
        perimeter = torch.cat(
            (
                ids[0, :-1],
                ids[:-1, -1],
                ids[-1, 1:].flip(0),
                ids[1:, 0].flip(0),
            )
        )
        grid[:, perimeter] = boundary
        return grid.reshape((boundary.shape[0], ny, nx, dimension))
    else:
        nx, ny, nz = shape
        grid = boundary.new_zeros((boundary.shape[0], nz * ny * nx, 3))
        ids = torch.arange(nx * ny * nz, device=boundary.device).reshape(nz, ny, nx)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        mask[[0, -1], :, :] = True
        mask[:, [0, -1], :] = True
        mask[:, :, [0, -1]] = True
    grid[:, ids[mask].reshape(-1)] = boundary
    return grid.reshape((boundary.shape[0], *ids.shape, dimension))


def _points_in_polygon(points, polygon):
    """Vectorized even-odd test for [B,N,2] points and [B,M,2] polygons."""
    x, y = points[..., 0:1], points[..., 1:2]
    x0, y0 = polygon[..., 0].unsqueeze(1), polygon[..., 1].unsqueeze(1)
    rolled = polygon.roll(-1, dims=1)
    x1, y1 = rolled[..., 0].unsqueeze(1), rolled[..., 1].unsqueeze(1)
    crosses = (y0 > y) != (y1 > y)
    intersection_x = (x1 - x0) * (y - y0) / (y1 - y0 + 1e-12) + x0
    return (crosses & (x < intersection_x)).sum(-1).remainder(2).bool()


def _tfi_2d(boundary, queries, shape):
    nx, ny = shape
    grid = _boundary_grid(boundary, shape)
    bottom, top = grid[:, 0], grid[:, -1]
    left, right = grid[:, :, 0], grid[:, :, -1]
    u = queries[..., 0]
    v = queries[..., 1]
    b, t = _interp1(bottom, u), _interp1(top, u)
    l, r = _interp1(left, v), _interp1(right, v)
    p00, p10 = grid[:, 0, 0, None], grid[:, 0, -1, None]
    p01, p11 = grid[:, -1, 0, None], grid[:, -1, -1, None]
    uu, vv = u[..., None], v[..., None]
    target = (
        (1 - uu) * l + uu * r + (1 - vv) * b + vv * t
        - (1 - uu) * (1 - vv) * p00
        - uu * (1 - vv) * p10
        - (1 - uu) * vv * p01
        - uu * vv * p11
    )
    return target, _points_in_polygon(target, boundary)


def _surface_triangles(grid):
    """Triangulate the six sampled boundary faces of [B,nz,ny,nx,3]."""
    faces = (
        grid[:, :, :, 0], grid[:, :, :, -1],
        grid[:, :, 0, :], grid[:, :, -1, :],
        grid[:, 0, :, :], grid[:, -1, :, :],
    )
    triangles = []
    for face in faces:
        p00, p10 = face[:, :-1, :-1], face[:, :-1, 1:]
        p01, p11 = face[:, 1:, :-1], face[:, 1:, 1:]
        triangles.extend((torch.stack((p00, p10, p11), -2), torch.stack((p00, p11, p01), -2)))
    return torch.cat([value.reshape(value.shape[0], -1, 3, 3) for value in triangles], 1)


def _points_in_surface(points, triangles):
    """Odd-even ray test with a non-axis-aligned ray to avoid regular-grid edges."""
    direction = points.new_tensor([1.0, 0.37139067, 0.21789141])
    direction = direction / torch.linalg.vector_norm(direction)
    v0, v1, v2 = triangles.unbind(-2)
    edge1, edge2 = v1 - v0, v2 - v0
    h = torch.linalg.cross(direction.expand_as(edge2), edge2, dim=-1)
    a = (edge1 * h).sum(-1)
    valid_a = a.abs() > 1e-9
    inv_a = torch.where(valid_a, a.reciprocal(), torch.zeros_like(a))
    s = points[:, :, None, :] - v0[:, None, :, :]
    u = inv_a[:, None, :] * (s * h[:, None, :, :]).sum(-1)
    q = torch.linalg.cross(s, edge1[:, None, :, :], dim=-1)
    v = inv_a[:, None, :] * (direction * q).sum(-1)
    distance = inv_a[:, None, :] * (edge2[:, None, :, :] * q).sum(-1)
    hit = (
        valid_a[:, None, :]
        & (u >= -1e-7) & (v >= -1e-7) & (u + v <= 1 + 1e-7)
        & (distance > 1e-7)
    )
    return hit.sum(-1).remainder(2).bool()


def _tfi_3d(boundary, queries, shape):
    grid = _boundary_grid(boundary, shape)
    u, v, w = queries.unbind(-1)
    uu, vv, ww = u[..., None], v[..., None], w[..., None]
    left, right = _interp2(grid[:, :, :, 0], v, w), _interp2(grid[:, :, :, -1], v, w)
    front, back = _interp2(grid[:, :, 0, :], u, w), _interp2(grid[:, :, -1, :], u, w)
    bottom, top = _interp2(grid[:, 0, :, :], u, v), _interp2(grid[:, -1, :, :], u, v)
    surface = (1-uu)*left + uu*right + (1-vv)*front + vv*back + (1-ww)*bottom + ww*top
    edge = (
        (1-uu)*(1-vv)*_interp1(grid[:, :, 0, 0], w)
        + uu*(1-vv)*_interp1(grid[:, :, 0, -1], w)
        + (1-uu)*vv*_interp1(grid[:, :, -1, 0], w)
        + uu*vv*_interp1(grid[:, :, -1, -1], w)
        + (1-uu)*(1-ww)*_interp1(grid[:, 0, :, 0], v)
        + uu*(1-ww)*_interp1(grid[:, 0, :, -1], v)
        + (1-uu)*ww*_interp1(grid[:, -1, :, 0], v)
        + uu*ww*_interp1(grid[:, -1, :, -1], v)
        + (1-vv)*(1-ww)*_interp1(grid[:, 0, 0, :], u)
        + vv*(1-ww)*_interp1(grid[:, 0, -1, :], u)
        + (1-vv)*ww*_interp1(grid[:, -1, 0, :], u)
        + vv*ww*_interp1(grid[:, -1, -1, :], u)
    )
    corners = [
        grid[:, 0, 0, 0, None], grid[:, 0, 0, -1, None],
        grid[:, 0, -1, 0, None], grid[:, 0, -1, -1, None],
        grid[:, -1, 0, 0, None], grid[:, -1, 0, -1, None],
        grid[:, -1, -1, 0, None], grid[:, -1, -1, -1, None],
    ]
    corner = (
        (1-uu)*(1-vv)*(1-ww)*corners[0] + uu*(1-vv)*(1-ww)*corners[1]
        + (1-uu)*vv*(1-ww)*corners[2] + uu*vv*(1-ww)*corners[3]
        + (1-uu)*(1-vv)*ww*corners[4] + uu*(1-vv)*ww*corners[5]
        + (1-uu)*vv*ww*corners[6] + uu*vv*ww*corners[7]
    )
    target = surface - edge + corner
    return target, _points_in_surface(target, _surface_triangles(grid))


def tfi_anchor_targets(boundary, queries, shape):
    """Return TFI coordinates and a mask excluding targets outside the boundary."""
    if queries.dim() == 2:
        queries = queries.unsqueeze(0).expand(boundary.shape[0], -1, -1)
    return _tfi_2d(boundary, queries, shape) if len(shape) == 2 else _tfi_3d(boundary, queries, shape)


def masked_anchor_mse(prediction, target, valid):
    """Coordinate-summed MSE over valid anchors only."""
    squared = (prediction - target).square().sum(-1)
    if not bool(valid.any()):
        return prediction.sum() * 0.0
    return squared[valid].mean()

