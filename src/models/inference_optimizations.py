"""Stateless forward optimizations; model parameters and checkpoints are unchanged."""

import torch
from src.data.backend.curve_kernels import launch_elementwise
from src.data.backend.inference_options import ROUTES

# Explicit rounding prevents NVRTC reassociation from amplifying weight errors
# when a trained model has large raw coordinates cancelled by the constraint.
WEIGHTS = """template <typename T> void hmonet_current_weights(
    T u, T v, T qu, T qv, T a0, T a1, T a2, T a3,
    T& w0, T& w1, T& w2, T& w3) {
    volatile T c0=a0*qv, c1=a1*qv;
    volatile T c2=a2*qu, c3=a3*qu;
    T d0=u*(T(1)-u)+c0, d1=u*(T(1)-u)+c1;
    T d2=v*(T(1)-v)+c2, d3=v*(T(1)-v)+c3;
    d0 = d0 == T(0) ? T(1) : d0;
    d1 = d1 == T(0) ? T(1) : d1;
    d2 = d2 == T(0) ? T(1) : d2;
    d3 = d3 == T(0) ? T(1) : d3;
    volatile T n0=c0*(T(1)-u), n1=c1*u;
    volatile T n2=c2*(T(1)-v), n3=c3*v;
        w0=sizeof(T)==4 ? T(__fdiv_rn(float(n0),float(d0))) : T(__ddiv_rn(double(n0),double(d0)));
    w1=sizeof(T)==4 ? T(__fdiv_rn(float(n1),float(d1))) : T(__ddiv_rn(double(n1),double(d1)));
    w2=sizeof(T)==4 ? T(__fdiv_rn(float(n2),float(d2))) : T(__ddiv_rn(double(n2),double(d2)));
    w3=sizeof(T)==4 ? T(__fdiv_rn(float(n3),float(d3))) : T(__ddiv_rn(double(n3),double(d3)));
    w0 = w0 < T(0) ? T(0) : (w0 > T(1) ? T(1) : w0);
    w1 = w1 < T(0) ? T(0) : (w1 > T(1) ? T(1) : w1);
    w2 = w2 < T(0) ? T(0) : (w2 > T(1) ? T(1) : w2);
    w3 = w3 < T(0) ? T(0) : (w3 > T(1) ? T(1) : w3);
    T sum = ((w0+w1)+w2)+w3;
    T center = ((T(1)-w0)*(T(1)-w1))*((T(1)-w2)*(T(1)-w3));
    if (sum > T(0)) {
        T scale = (T(1)-center)/sum;
        w0 *= scale; w1 *= scale; w2 *= scale; w3 *= scale;
    }
    if ((u == T(0) || u == T(1)) && (v == T(0) || v == T(1))) {
        w0 = u == T(0) ? T(1) : T(0);
        w1 = u == T(0) ? T(0) : T(1);
        w2 = T(-1); // Corner sentinel: final kernel bypasses the raw prediction.
        w3 = T(0);
    }
}"""
PAIR = """template <typename T> void hmonet_projection_pair(
    T raw, T wl, T wr, T wb, T wt, T left, T right, T& value) {
    if (wb < T(0)) { value = wl == T(1) ? left : right; return; }
    volatile T w = T(1)-wl;
    w=w-wr; w=w-wb; w=w-wt;
    volatile T a=w*raw, b=wl*left, c=wr*right;
    a=a+b; value=a+c;
}"""
COMBINE = """template <typename T> void hmonet_combine(
    T pair, T wb, T wt, T bottom, T top, T& value) {
    if (wb < T(0)) { value = pair; return; }
    volatile T b=wb*bottom, t=wt*top;
    volatile T a=pair+b;
    value=a+t;
}"""


def blend_current_2d(
    xi,
    eta,
    weight,
    raw,
    projected,
    transition_width=0.2,
    normal_smoothing=0.5,
    tangential_lift_beta=0.6,
):
    """Fuse rational weights and blending in three kernels (eight-input limit).

    Any active autograd path uses the model's original Torch implementation.
    No geometry, query points, activations or model weights are cached here.
    """
    if (
        not ROUTES.get()[0]
        or torch.is_grad_enabled()
    ):
        return None
    smoothing = float(normal_smoothing)
    beta = float(tangential_lift_beta)
    u = (1.0 - smoothing) * xi + smoothing * xi.square() * (3.0 - 2.0 * xi)
    v = (1.0 - smoothing) * eta + smoothing * eta.square() * (3.0 - 2.0 * eta)
    qu0 = 4.0 * xi * (1.0 - xi)
    qv0 = 4.0 * eta * (1.0 - eta)
    qu = qu0 / (beta + (1.0 - beta) * qu0)
    qv = qv0 / (beta + (1.0 - beta) * qv0)
    scaled_weight = weight * float(transition_width)
    weights = launch_elementwise(
        "hmonet_current_weights",
        WEIGHTS,
        4,
        u,
        v,
        qu,
        qv,
        *scaled_weight.unbind(-1),
    )
    if weights is None:
        return None
    wl, wr, wb, wt = (value[..., None] for value in weights)
    pair = launch_elementwise(
        "hmonet_projection_pair",
        PAIR,
        1,
        raw,
        wl,
        wr,
        wb,
        wt,
        projected[..., 0, :],
        projected[..., 1, :],
    )
    if pair is None:
        return None
    if isinstance(pair, tuple):
        pair = pair[0]
    result = launch_elementwise(
        "hmonet_combine",
        COMBINE,
        1,
        pair,
        wb,
        wt,
        projected[..., 2, :],
        projected[..., 3, :],
    )
    if isinstance(result, tuple):
        result = result[0]
    return result


def fuse_coordinates(trunk, branch_x, branch_y, bias):
    """Contract feature dimensions directly instead of materializing two products."""
    branch = torch.cat((branch_x, branch_y), dim=-2)
    return torch.matmul(trunk, branch.transpose(-1, -2)) + bias
