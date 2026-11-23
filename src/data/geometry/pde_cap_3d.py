"""Dense planar Winslow caps with exact edges and differentiable interpolation.

Only run_e uses this representation. The volume is still solved by Winslow3D.
The spline interpolates the displacement from the analytic Coons map, whose
four exact curves are retained. Logical parameters are not redistributed.
"""
from __future__ import annotations

import base64
from functools import lru_cache
import zlib

import numpy as np
from scipy.interpolate import RectBivariateSpline
import torch

from .parametric_geometry_3d import (
    ParametricSurface3D, BoundarySurfaceSet3D,
    _rounded_u_notch_top_numpy, _rounded_u_notch_top_torch,
)


def _coons_numpy(u, v, parameters):
    p = parameters
    top = _rounded_u_notch_top_numpy(
        u, p["width"], p["height"], p["notch_width"], p["depth"], p["radius"])
    return np.stack(((1-v)*p["width"]*u + v*top[..., 0], v*top[..., 1]), -1)


def _cubic_basis(query, knots):
    """Four nonzero B-spline basis values; retains first/second query derivatives."""
    span = (torch.searchsorted(knots, query.contiguous(), right=True)-1).clamp(3, len(knots)-5)
    basis = [torch.ones_like(query)]
    for degree in range(1, 4):
        saved = torch.zeros_like(query)
        new = []
        for index in range(degree):
            left = query - knots[span + 1 - degree + index]
            right = knots[span + index + 1] - query
            term = basis[index] / (left + right)
            new.append(saved + right*term)
            saved = left*term
        basis = new + [saved]
    return span[..., None]-3+torch.arange(4, device=query.device), torch.stack(basis, -1)


def _same_knots_residual(caps, u, v):
    """Evaluate a batch with common knots and per-geometry coefficients.

    Only tensor preparation loops over geometries. Basis evaluation and all
    sixteen coefficient gathers are batched, including query derivatives.
    No batch-combination cache is retained across training iterations.
    """
    tensors = [cap.tensors(u) for cap in caps]
    tu, tv = tensors[0][:2]
    coefficients = torch.stack([item[2] for item in tensors])
    iu, bu = _cubic_basis(u, tu)
    iv, bv = _cubic_basis(v, tv)
    batch = torch.arange(len(caps), device=u.device)[:, None, None, None]
    local = coefficients[batch, iu[..., :, None], iv[..., None, :], :]
    return (local * bu[..., :, None, None] * bv[..., None, :, None]).sum(dim=(-3, -2))


def _batch_spline_residual(caps, u, v):
    """Group heterogeneous knot vectors without changing any parameter map."""
    groups = {}
    for index, cap in enumerate(caps):
        groups.setdefault(cap._knot_key, []).append(index)
    if len(groups) == 1:
        return _same_knots_residual(caps, u, v)
    # Different cap resolutions/custom knots remain supported. Reassemble in
    # original batch order, preserving autograd through index_select/cat.
    values, order = [], []
    for indices in groups.values():
        selection = torch.tensor(indices, device=u.device)
        values.append(_same_knots_residual(
            [caps[i] for i in indices], u.index_select(0, selection),
            v.index_select(0, selection)))
        order.extend(indices)
    inverse = torch.tensor(np.argsort(order), device=u.device)
    return torch.cat(values, dim=0).index_select(0, inverse)


class RunPdeCapMap:
    """Immutable numerical cap shared by the six faces and serialized once."""

    def __init__(self, parameters, knots_u, knots_v, coefficients, solver_info):
        self.parameters = dict(parameters)
        self.knots_u = np.asarray(knots_u, dtype=np.float64)
        self.knots_v = np.asarray(knots_v, dtype=np.float64)
        # CPU-only grouping metadata; not serialized and independent of device.
        self._knot_key = (self.knots_u.tobytes(), self.knots_v.tobytes())
        self.coefficients = np.array(coefficients, dtype=np.float64, copy=True)
        expected = (len(self.knots_u)-4, len(self.knots_v)-4, 2)
        if self.coefficients.shape != expected or not np.isfinite(self.coefficients).all():
            raise ValueError("Invalid run_e cap spline coefficients")
        self.solver_info = dict(solver_info)
        self._splines = [RectBivariateSpline._from_tck(
            (self.knots_u, self.knots_v, self.coefficients[..., axis].ravel(), 3, 3))
            for axis in (0, 1)]
        self._tensors = {}

    def evaluate_numpy(self, u, v, *, residual=True):
        u, v = np.broadcast_arrays(np.clip(u, 0, 1), np.clip(v, 0, 1))
        xy = _coons_numpy(u, v, self.parameters)
        if residual:
            xy = xy + np.stack([s.ev(u, v) for s in self._splines], -1)
        return xy

    def tensors(self, query):
        key = (query.device, query.dtype)
        if key not in self._tensors:
            self._tensors[key] = tuple(torch.as_tensor(a.copy(), dtype=query.dtype, device=query.device)
                                       for a in (self.knots_u, self.knots_v, self.coefficients))
        return self._tensors[key]

    def to_dict(self):
        # Lossless float64 storage; avoid a multi-megabyte decimal JSON array.
        return dict(type="run_pde_cap_map_v1", parameters=self.parameters,
                    knots_u=self.knots_u.tolist(), knots_v=self.knots_v.tolist(),
                    coefficient_shape=list(self.coefficients.shape),
                    coefficients_zlib_f64=base64.b64encode(zlib.compress(
                        self.coefficients.astype('<f8').tobytes())).decode('ascii'),
                    solver_info=self.solver_info)

    @classmethod
    def from_dict(cls, data):
        coefficients = np.frombuffer(zlib.decompress(base64.b64decode(
            data["coefficients_zlib_f64"])), dtype='<f8').reshape(data["coefficient_shape"])
        return cls(data["parameters"], data["knots_u"], data["knots_v"], coefficients,
                   data["solver_info"])


@lru_cache(maxsize=8)
def build_run_cap(width, height, notch_width, depth, radius, nx=201, ny=201,
                  max_iter=1000, tolerance=1e-10):
    """Solve once per geometry; loading a saved cap never invokes this solver."""
    from .boundary_generator_2d import rounded_u_notch_boundary
    from src.numerics.traditional.winslow_vectorized_2d import solve_winslow_2d_vectorized
    from src.data.generation.dataset_quality_2d import audit_grid

    if int(nx) != nx or int(ny) != ny or min(nx, ny) < 11:
        raise ValueError("run_e cap_nx and cap_ny must be integers >= 11")
    nx, ny = int(nx), int(ny)
    p = dict(width=width, height=height, notch_width=notch_width, depth=depth, radius=radius)
    boundary = rounded_u_notch_boundary(**p)
    mesh, info = solve_winslow_2d_vectorized(boundary, nx, ny, max_iter=max_iter,
                                           tolerance=tolerance, return_info=True)
    if not info["converged"]:
        raise ValueError(f"run_e end-cap 2-D PDE did not converge: {info}")
    audit = audit_grid(mesh, nx, ny)
    if not audit["all_cells_legal"]:
        raise ValueError(f"run_e prescribed PDE end-cap is invalid: {audit}")
    u, v = np.linspace(0, 1, nx), np.linspace(0, 1, ny)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    displacement = mesh.reshape(ny, nx, 2) - _coons_numpy(uu, vv, p)
    displacement[[0, -1], :] = 0
    displacement[:, [0, -1]] = 0
    splines = [RectBivariateSpline(u, v, displacement[..., axis].T, kx=3, ky=3, s=0)
               for axis in (0, 1)]
    tu, tv = splines[0].get_knots()
    coefficients = np.stack([s.get_coeffs().reshape(len(tu)-4, len(tv)-4) for s in splines], -1)
    # Open cubic knots select exactly these coefficients on the four edges.
    coefficients[[0, -1], :] = 0
    coefficients[:, [0, -1]] = 0
    cap = RunPdeCapMap(p, tu, tv, coefficients,
                      dict(info, grid_shape=[nx, ny], solver="winslow_2d_vectorized",
                           tolerance=tolerance, max_iter=max_iter))
    # Check the continuous representation at a denser, nonidentical grid too.
    checks = {}
    for nu, nv in ((nx, ny), (2*nx-1, 2*ny-1)):
        a, b = np.meshgrid(np.linspace(0, 1, nu), np.linspace(0, 1, nv), indexing="xy")
        check = audit_grid(cap.evaluate_numpy(a, b), nu, nv)
        checks[f"{nu}x{nv}"] = check
        if not check["all_cells_legal"]:
            raise ValueError(f"run_e interpolated end-cap invalid at {nu}x{nv}: {check}")
    cap.solver_info["interpolation_checks"] = checks
    return cap


class RunPdeFace3D(ParametricSurface3D):
    """Face of the PDE-parametrized extrusion, with shared cap and exact edges."""

    def __init__(self, cap, fixed_axis, fixed_value, length, z_min):
        if fixed_axis not in (0, 1, 2) or fixed_value not in (0, 1):
            raise ValueError("Invalid run_e face")
        self.cap = cap
        self.fixed_axis, self.fixed_value = int(fixed_axis), int(fixed_value)
        self.length, self.z_min = float(length), float(z_min)

    def evaluate(self, u, v):
        if torch.is_tensor(u) or torch.is_tensor(v):
            ref = u if torch.is_tensor(u) else v
            return self.evaluate_torch(torch.as_tensor(u, device=ref.device, dtype=ref.dtype),
                                       torch.as_tensor(v, device=ref.device, dtype=ref.dtype))
        u, v = np.broadcast_arrays(np.clip(u, 0, 1), np.clip(v, 0, 1))
        fixed = np.full_like(u, self.fixed_value)
        xi, eta, zeta = ((fixed, u, v) if self.fixed_axis == 0 else
                         (u, fixed, v) if self.fixed_axis == 1 else (u, v, fixed))
        xy = self.cap.evaluate_numpy(xi, eta, residual=self.fixed_axis == 2)
        return np.concatenate((xy, (self.z_min+self.length*zeta)[..., None]), -1)

    def evaluate_torch(self, u, v):
        u, v = torch.broadcast_tensors(u, v)
        return self.evaluate_batch_torch([self], u.reshape(1, -1), v.reshape(1, -1)).reshape(*u.shape, 3)

    @classmethod
    def evaluate_batch_torch(cls, surfaces, u, v):
        axis, side = surfaces[0].fixed_axis, surfaces[0].fixed_value
        if any((s.fixed_axis, s.fixed_value) != (axis, side) for s in surfaces):
            raise ValueError("run_e face batch must share axis and side")
        u, v = u.clamp(0, 1), v.clamp(0, 1)
        fixed = torch.full_like(u, side)
        xi, eta, zeta = ((fixed, u, v) if axis == 0 else
                         (u, fixed, v) if axis == 1 else (u, v, fixed))
        names = ("width", "height", "notch_width", "depth", "radius")
        parameters = u.new_tensor([[s.cap.parameters[n] for n in names] for s in surfaces])
        p = {n: parameters[:, i:i+1] for i, n in enumerate(names)}
        top = _rounded_u_notch_top_torch(xi, *(p[n] for n in names))
        xy = torch.stack(((1-eta)*p["width"]*xi + eta*top[..., 0], eta*top[..., 1]), -1)
        if axis == 2:
            xy = xy + _batch_spline_residual([s.cap for s in surfaces], xi, eta)
        z0 = u.new_tensor([s.z_min for s in surfaces])[:, None]
        length = u.new_tensor([s.length for s in surfaces])[:, None]
        return torch.cat((xy, (z0+length*zeta)[..., None]), -1)

    def to_dict(self):
        return dict(type="run_pde_face_3d", fixed_axis=self.fixed_axis,
                    fixed_value=self.fixed_value, length=self.length, z_min=self.z_min,
                    cap=self.cap.to_dict())

    @classmethod
    def from_dict(cls, data, cap=None):
        return cls(cap if cap is not None else RunPdeCapMap.from_dict(data["cap"]),
                   data["fixed_axis"], data["fixed_value"], data["length"], data["z_min"])


def run_pde_boundary(parameters, cap_nx=201, cap_ny=201,
                     cap_max_iter=1000, cap_tolerance=1e-10):
    p = parameters
    cap = build_run_cap(*(p[n] for n in ("width", "height", "notch_width", "depth", "radius")),
                        nx=cap_nx, ny=cap_ny,
                        max_iter=cap_max_iter, tolerance=cap_tolerance)
    faces = [RunPdeFace3D(cap, axis, side, p["length"], p["z_min"])
             for axis in range(3) for side in (0, 1)]
    return BoundarySurfaceSet3D(*faces, metadata=dict(
        p, family="rounded_u_notch_extrusion", topology="exact_rounded_u_notch_extrusion",
        geometry_revision=2, cap_parameterization="winslow_2d_cubic_displacement",
        cap_nx=int(cap_nx), cap_ny=int(cap_ny),
        cap_max_iter=cap_max_iter, cap_tolerance=cap_tolerance))
