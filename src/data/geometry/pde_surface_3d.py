"""Serializable planar PDE caps with exact Coons edges.

Each cap is solved independently. Side faces retain the original geometry:
TFI of the four unit-square chart edges is the identity chart map.
Dense PDE nodes are construction data, not network input or saved duplicates.
"""
from __future__ import annotations

import base64
import json
import zlib
from functools import lru_cache
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np
import torch
from scipy.interpolate import RectBivariateSpline

from .parametric_geometry_3d import ParametricSurface3D, BoundarySurfaceSet3D, surface_from_dict
from .pde_cap_3d import _batch_spline_residual

CAP_SCHEME = "planar_winslow_coons_cubic_v1"
SIDE_SCHEME = "original_surface_chart_tfi"
_CHECK_CAPS = ContextVar('check_pde_caps', default=True)


@contextmanager
def cap_validation_context(enabled):
    """Dataset workers audit representative samples; standalone calls audit by default."""
    token = _CHECK_CAPS.set(bool(enabled))
    try:
        yield
    finally:
        _CHECK_CAPS.reset(token)

CURRENT_CAP_FAMILIES = (
    "cylindrical_domain_with_airfoil_hole", "sidewall_deformed_block",
    "rounded_u_notch_extrusion", "tapered_elliptic_bore_block",
    "bent_annular_duct", "lobed_annular_duct", "square_to_round_transition",
    "variable_area_s_duct", "twisted_blade_passage",
)


def coons_numpy(source, u, v):
    u, v = np.broadcast_arrays(u, v)
    z, o = np.zeros_like(u), np.ones_like(u)
    a, b = u[..., None], v[..., None]
    return ((1-b)*source.evaluate(u,z) + b*source.evaluate(u,o)
            + (1-a)*source.evaluate(z,v) + a*source.evaluate(o,v)
            - ((1-a)*(1-b)*source.evaluate(0.,0.) + a*(1-b)*source.evaluate(1.,0.)
               + (1-a)*b*source.evaluate(0.,1.) + a*b*source.evaluate(1.,1.)))


class PdeCapSurface3D(ParametricSurface3D):
    """One planar cap: exact edge Coons map plus cubic displacement spline."""

    def __init__(self, source, knots_u, knots_v, coefficients, solver_info):
        self.source = source
        self.knots_u, self.knots_v = np.asarray(knots_u, float), np.asarray(knots_v, float)
        self.coefficients = np.array(coefficients, dtype=np.float64, copy=True)
        expected = (len(self.knots_u)-4, len(self.knots_v)-4, 3)
        if self.coefficients.shape != expected or not np.isfinite(self.coefficients).all():
            raise ValueError("Invalid planar PDE cap coefficients")
        if any(np.any(x != 0) for x in (self.coefficients[0],self.coefficients[-1],
                                        self.coefficients[:,0],self.coefficients[:,-1])):
            raise ValueError("PDE cap displacement must vanish on all four edges")
        self.solver_info = dict(solver_info)
        self._knot_key = (self.knots_u.tobytes(),self.knots_v.tobytes())
        self._tensors = {}
        self._splines = [RectBivariateSpline._from_tck((self.knots_u,self.knots_v,
                            self.coefficients[...,k].ravel(),3,3)) for k in range(3)]
        self.corners = np.array([source.evaluate(u,v) for u,v in ((0.,0.),(1.,0.),(0.,1.),(1.,1.))])

    def evaluate(self,u,v):
        if torch.is_tensor(u) or torch.is_tensor(v):
            ref=u if torch.is_tensor(u) else v
            return self.evaluate_torch(torch.as_tensor(u,device=ref.device,dtype=ref.dtype),
                                       torch.as_tensor(v,device=ref.device,dtype=ref.dtype))
        u,v=np.broadcast_arrays(np.clip(u,0,1),np.clip(v,0,1))
        return coons_numpy(self.source,u,v)+np.stack([s.ev(u,v) for s in self._splines],-1)

    def tensors(self,q):
        key=(q.device,q.dtype)
        if key not in self._tensors:
            self._tensors[key]=tuple(torch.as_tensor(a.copy(),device=q.device,dtype=q.dtype)
                                    for a in (self.knots_u,self.knots_v,self.coefficients))
        return self._tensors[key]

    @classmethod
    def evaluate_batch_torch(cls,surfaces,u,v):
        from src.data.backend.parametric_torch import evaluate_surfaces_torch
        u,v=u.clamp(0,1),v.clamp(0,1)
        z,o=torch.zeros_like(u),torch.ones_like(u)
        source=[s.source for s in surfaces]
        values=evaluate_surfaces_torch(source*4,torch.cat((u,u,z,o)),torch.cat((z,o,v,v)))
        bottom,top,left,right=values.reshape(4,*u.shape,3).unbind(0)
        c=u.new_tensor(np.stack([s.corners for s in surfaces]))
        a,b=u[...,None],v[...,None]
        base=((1-b)*bottom+b*top+(1-a)*left+a*right
              -((1-a)*(1-b)*c[:,None,0]+a*(1-b)*c[:,None,1]
                +(1-a)*b*c[:,None,2]+a*b*c[:,None,3]))
        return base+_batch_spline_residual(surfaces,u,v)

    def to_dict(self):
        return dict(type="planar_pde_cap_3d",scheme=CAP_SCHEME,source=self.source.to_dict(),
                    knots_u=self.knots_u.tolist(),knots_v=self.knots_v.tolist(),degree=[3,3],
                    coefficient_shape=list(self.coefficients.shape),
                    coefficients_zlib_f64=base64.b64encode(zlib.compress(self.coefficients.astype('<f8').tobytes())).decode('ascii'),
                    solver_info=self.solver_info)

    @classmethod
    def from_dict(cls,data):
        if data.get('scheme')!=CAP_SCHEME or data.get('degree')!=[3,3]:
            raise ValueError("Unsupported PDE cap scheme or degree")
        c=np.frombuffer(zlib.decompress(base64.b64decode(data['coefficients_zlib_f64'])),dtype='<f8').reshape(data['coefficient_shape'])
        return cls(surface_from_dict(data['source']),data['knots_u'],data['knots_v'],c,data['solver_info'])


@lru_cache(maxsize=18)
def _build_cached(serialized_source,nx,ny,max_iter,tolerance,
                  solver_backend='cpu',device='cuda:0',validate=True):
    from .parametric_curve_2d import FunctionalCurve, BoundaryCurveSet2D
    from src.numerics.traditional.winslow_vectorized_2d import solve_winslow_2d_vectorized
    from src.data.generation.dataset_quality_2d import audit_grid
    source=surface_from_dict(json.loads(serialized_source))
    xyz=source.sample(31,31).reshape(-1,3);origin=xyz.mean(0)
    _,s,vh=np.linalg.svd(xyz-origin,full_matrices=False)
    if s[1]<=1e-12 or s[-1]>1e-10*s[0]:
        raise ValueError("Planar PDE cap requires a nondegenerate planar source face")
    a,b=vh[:2];du=source.evaluate(.5001,.5)-source.evaluate(.4999,.5)
    dv=source.evaluate(.5,.5001)-source.evaluate(.5,.4999)
    if np.dot(du,a)*np.dot(dv,b)-np.dot(du,b)*np.dot(dv,a)<0:b=-b
    frame=np.stack((a,b),-1)
    def planar(u,v):return (source.evaluate(u,v)-origin)@frame
    boundary=BoundaryCurveSet2D(FunctionalCurve(lambda t:planar(t,np.zeros_like(t))),
        FunctionalCurve(lambda t:planar(np.ones_like(t),t)),
        FunctionalCurve(lambda t:planar(t,np.ones_like(t))),
        FunctionalCurve(lambda t:planar(np.zeros_like(t),t)))
    u,v=np.linspace(0,1,nx),np.linspace(0,1,ny);uu,vv=np.meshgrid(u,v,indexing='xy')
    solver = solve_winslow_2d_vectorized
    options = {}
    if solver_backend == 'cuda':
        from src.numerics.traditional.cudss_solver import CudssSolver
        direct = CudssSolver(device)
        options = dict(linear_solver=direct)
    mesh,info=solver(boundary,nx,ny,initial_grid=planar(uu,vv),
                     max_iter=max_iter,tolerance=tolerance,return_info=True,**options)
    if solver_backend == 'cuda':
        info.update(direct.diagnostics())
    if not info['converged']:raise ValueError(f"End-cap PDE did not converge: {info}")
    if validate:
        audit=audit_grid(mesh,nx,ny)
        if not audit['all_cells_legal']:raise ValueError(f"Invalid prescribed PDE end-cap: {audit}")
    residual=mesh.reshape(ny,nx,2)@frame.T+origin-coons_numpy(source,uu,vv)
    residual[[0,-1],:]=0;residual[:,[0,-1]]=0
    splines=[RectBivariateSpline(u,v,residual[...,k].T,kx=3,ky=3,s=0) for k in range(3)]
    tu,tv=splines[0].get_knots()
    coeff=np.stack([sp.get_coeffs().reshape(len(tu)-4,len(tv)-4) for sp in splines],-1)
    coeff[[0,-1],:]=0;coeff[:,[0,-1]]=0
    cap=PdeCapSurface3D(source,tu,tv,coeff,dict(info,grid_shape=[nx,ny],solver=('winslow_2d_cudss' if solver_backend=='cuda' else 'winslow_2d_vectorized'),
                                            tolerance=tolerance,max_iter=max_iter))
    checks={}
    for nu,nv in (((nx,ny),) if validate else ()):
        a,b=np.meshgrid(np.linspace(0,1,nu),np.linspace(0,1,nv),indexing='xy')
        check=audit_grid((cap.evaluate(a,b)-origin)@frame,nu,nv)
        checks[f'{nu}x{nv}']=check
        if not check['all_cells_legal']:raise ValueError(f"Interpolated PDE cap invalid at {nu}x{nv}: {check}")
    cap.solver_info['interpolation_checks']=checks
    cap.solver_info['legality_checked']=bool(validate)
    return cap


def pde_cap_boundary(boundary,cap_nx=201,cap_ny=201,cap_max_iter=1000,cap_tolerance=1e-6):
    """Independent planar PDE caps; identity TFI in original side-face charts."""
    if any(isinstance(n,bool) or int(n)!=n or n<11 for n in (cap_nx,cap_ny)):
        raise ValueError('cap_nx and cap_ny must be integers >=11')
    if (isinstance(cap_max_iter, bool) or int(cap_max_iter) != cap_max_iter
            or cap_max_iter < 1 or not np.isfinite(cap_tolerance) or cap_tolerance <= 0):
        raise ValueError('cap_max_iter must be a positive integer and cap_tolerance finite and positive')
    faces=boundary.surfaces()
    from src.numerics.traditional.generation_backend import current_solver_options
    options = current_solver_options()
    for name in ('z_min','z_max'):
        serialized=json.dumps(faces[name].to_dict(),sort_keys=True,separators=(',',':'))
        faces[name]=_build_cached(serialized,int(cap_nx),int(cap_ny),int(cap_max_iter),float(cap_tolerance),validate=_CHECK_CAPS.get(),**options)
    metadata=dict(boundary.metadata,geometry_revision=3,cap_parameterization=CAP_SCHEME,
                  side_parameterization=SIDE_SCHEME,cap_nx=int(cap_nx),cap_ny=int(cap_ny),
                  cap_max_iter=int(cap_max_iter),cap_tolerance=float(cap_tolerance))
    return BoundarySurfaceSet3D(**faces,edge_curves=boundary.edge_curves,metadata=metadata)
