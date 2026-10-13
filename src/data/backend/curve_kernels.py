"""Optional fused CUDA curve formulas with differentiable PyTorch fallback.

Jiterator compiles a scalar formula once, then evaluates all query points in one
GPU kernel. Only compiled code is cached: geometry values and projected points
are never cached. Queries requiring gradients keep the ordinary Torch formulas.
"""

from contextlib import contextmanager
from contextvars import ContextVar
import ctypes
import os
from pathlib import Path
import sys
import time

import torch

_ENABLED = ContextVar("hmonet_curve_kernels_enabled", default=True)
_FUNCTIONS = {}
_FAILED = {}
_WARM = set()
_LIBRARIES = []
_RUNTIME_READY = False
_STATS = {"initialization_seconds": 0.0, "calls": 0}


@contextmanager
def curve_kernel_mode(enabled):
    """Temporarily select fused formulas or the reference Torch implementation."""
    token = _ENABLED.set(bool(enabled))
    try:
        yield
    finally:
        _ENABLED.reset(token)


def kernel_status():
    """Return diagnostics without synchronizing the GPU."""
    return {
        **_STATS,
        "compiled_variants": len(_WARM),
        "failures": {str(key): value for key, value in _FAILED.items()},
    }


def eligible(query):
    return (
        _ENABLED.get()
        and query.is_cuda
        and query.dtype in (torch.float32, torch.float64)
        and not (torch.is_grad_enabled() and query.requires_grad)
    )


def _initialize_runtime():
    """Preload Conda's NVRTC builtins when Windows cannot find its bin folder."""
    global _RUNTIME_READY
    if _RUNTIME_READY:
        return
    if os.name == "nt" and torch.version.cuda:
        suffix = torch.version.cuda.replace(".", "")
        filename = f"nvrtc-builtins64_{suffix}.dll"
        directories = (
            Path(sys.prefix) / "bin",
            Path(sys.prefix) / "Library" / "bin",
            Path(torch.__file__).parent / "lib",
        )
        for directory in directories:
            path = directory / filename
            if path.is_file():
                _LIBRARIES.append(ctypes.WinDLL(str(path)))
                break
    _RUNTIME_READY = True


def _persist_windows_kernel(kernel_name):
    """Finalize only this process's compiled curve files after NVRTC returns.

    PyTorch 2.5 attempts to rename its cache file before closing it, which can fail
    on Windows. After the call returns, that handle is closed and the exact owned
    temporary file can be renamed safely. Existing cache entries are never replaced.
    """
    if os.name != "nt" or os.environ.get("USE_PYTORCH_KERNEL_CACHE") == "0":
        return
    configured = os.environ.get("PYTORCH_KERNEL_CACHE_PATH")
    if configured:
        directory = Path(configured)
    elif os.environ.get("TEMP"):
        directory = Path(os.environ["TEMP"]) / "torch" / "kernels"
    else:
        return
    directory = directory.resolve()
    suffix = f"_tmp_{os.getpid()}"
    try:
        for temporary in directory.glob(f"{kernel_name}_arch*{suffix}"):
            target = temporary.with_name(temporary.name[: -len(suffix)])
            if (
                temporary.resolve().parent == directory
                and target.resolve().parent == directory
                and temporary.is_file()
                and temporary.stat().st_size > 0
                and not target.exists()
            ):
                temporary.rename(target)
    except OSError:
        # Persistent cache is optional; the current process already has a kernel.
        pass


NACA_BODY = """
    const bool upper = t <= T(0.5);
    const T xx = upper ? T(2)*t : T(2)*(T(1)-t);
    const T root = xx > T(0) ? sqrt(xx) : T(0);
    const T xx2 = xx*xx;
    const T yt = T(5)*thickness*chord*(T(0.2969)*root - T(0.1260)*xx
        - T(0.3516)*xx2 + T(0.2843)*(xx2*xx) - T(0.1036)*(xx2*xx2));
    const T one_p = T(1)-position;
    const bool before = xx < position;
    const T yc = before
        ? m/(position*position)*(T(2)*position*xx-xx2)
        : m/(one_p*one_p)*((T(1)-T(2)*position)+T(2)*position*xx-xx2);
    const T slope = before
        ? T(2)*m/(position*position)*(position-xx)
        : T(2)*m/(one_p*one_p)*(position-xx);
    const T angle = atan(slope);
    const T signed_yt = upper ? yt : -yt;
    x = xx*chord - signed_yt*sin(angle) + cx;
    y = yc + signed_yt*cos(angle) + cy;
"""


SOURCES = {
    "naca": """template <typename T> void exact_naca(
        T t, T m, T position, T thickness, T chord, T cx, T cy,
        T& x, T& y) {"""
    + NACA_BODY
    + "}",
    "piecewise_naca": """template <typename T> void exact_piecewise_naca(
        T q, T m, T position, T thickness, T chord, T cx, T cy,
        T k0, T k1, T k2, T start0, T end0, T start1, T end1, T& x, T& y) {
        q = q < T(0) ? T(0) : (q > T(1) ? T(1) : q);
        const bool first = q < k1;
        T local = first ? (q-k0)/(k1-k0) : (q-k1)/(k2-k1);
        local = local < T(0) ? T(0) : (local > T(1) ? T(1) : local);
        const T t = first ? start0+(end0-start0)*local
                          : start1+(end1-start1)*local;
    """
    + NACA_BODY
    + "}",
    "line2": """template <typename T> void exact_line2(
        T t, T sx, T sy, T ex, T ey, T& x, T& y) {
        t = t < T(0) ? T(0) : (t > T(1) ? T(1) : t);
        x = (T(1)-t)*sx + t*ex;
        y = (T(1)-t)*sy + t*ey;
    }""",
    "line3": """template <typename T> void exact_line3(
        T t, T sx, T sy, T sz, T ex, T ey, T ez, T& x, T& y, T& z) {
        t = t < T(0) ? T(0) : (t > T(1) ? T(1) : t);
        x = (T(1)-t)*sx + t*ex;
        y = (T(1)-t)*sy + t*ey;
        z = (T(1)-t)*sz + t*ez;
    }""",
    "circle": """template <typename T> void exact_circle(
        T t, T cx, T cy, T a, T b, T start, T sweep, T rotation,
        T& x, T& y) {
        t = t < T(0) ? T(0) : (t > T(1) ? T(1) : t);
        const T angle = start+sweep*t;
        x = cx+a*cos(angle);
        y = cy+b*sin(angle);
    }""",
    "ellipse": """template <typename T> void exact_ellipse(
        T t, T cx, T cy, T a, T b, T start, T sweep, T rotation,
        T& x, T& y) {
        t = t < T(0) ? T(0) : (t > T(1) ? T(1) : t);
        const T angle = start+sweep*t;
        const T xx = a*cos(angle);
        const T yy = b*sin(angle);
        x = cx+cos(rotation)*xx-sin(rotation)*yy;
        y = cy+sin(rotation)*xx+cos(rotation)*yy;
    }""",
}


def evaluate_fused(name, query, parameters, **constants):
    """Return exact formula values, or None when the reference path is needed."""
    if not eligible(query):
        return None
    if torch.is_tensor(parameters):
        if torch.is_grad_enabled() and parameters.requires_grad:
            return None
        parameters = [value[:, None] for value in parameters.unbind(-1)]
    key = (name, str(query.device), query.dtype)
    if key in _FAILED:
        return None
    if any(torch.is_grad_enabled() and p.requires_grad for p in parameters):
        return None
    cold = key not in _WARM
    # Compilation and module loading must happen before a graph capture.
    if cold and torch.cuda.is_current_stream_capturing():
        return None
    started = time.perf_counter() if cold else None
    try:
        if name not in _FUNCTIONS:
            _initialize_runtime()
            from torch.cuda.jiterator import _create_multi_output_jit_fn

            outputs = 3 if name in ("line3", "scalar_line3") else 2
            _FUNCTIONS[name] = _create_multi_output_jit_fn(
                SOURCES[name], outputs, **{key: 0.0 for key in constants}
            )
        values = _FUNCTIONS[name](query, *parameters, **constants)
        if cold:
            _persist_windows_kernel(_FUNCTIONS[name].kernel_name)
    except (ImportError, AttributeError, OSError, RuntimeError) as error:
        # Do not retry a missing compiler on every forward. Preserve errors
        # unrelated to compilation, such as an out-of-memory or invalid device.
        message = str(error)
        if isinstance(error, RuntimeError) and not any(
            term in message.lower()
            for term in ("nvrtc", "jiterator", "compil", "not implemented")
        ):
            raise
        _FAILED[key] = message[-1600:]
        return None
    finally:
        if cold:
            _STATS["initialization_seconds"] += time.perf_counter() - started
    _WARM.add(key)
    _STATS["calls"] += 1
    return torch.stack(values, dim=-1)


def evaluate_scalar_formula(name, query, coefficients):
    """Pass live geometry scalars as kernel arguments, without GPU parameter arrays."""
    from src.data.backend.inference_options import ROUTES

    if not ROUTES.get()[2] or not eligible(query):
        return None
    scalar_name = "scalar_" + name
    if scalar_name not in SOURCES:
        SOURCES[scalar_name] = SOURCES[name].replace(
            "exact_" + name, "scalar_exact_" + name
        )
    return evaluate_fused(scalar_name, query, [], **coefficients)


def launch_elementwise(name, source, outputs, *inputs):
    """Launch an inference-only elementwise formula with a safe Torch fallback."""
    if not inputs or not all(eligible(value) for value in inputs):
        return None
    key = (name, str(inputs[0].device), inputs[0].dtype)
    if key in _FAILED:
        return None
    cold = key not in _WARM
    started = time.perf_counter() if cold else None
    try:
        if name not in _FUNCTIONS:
            _initialize_runtime()
            from torch.cuda.jiterator import _create_multi_output_jit_fn

            _FUNCTIONS[name] = _create_multi_output_jit_fn(source, outputs)
        result = _FUNCTIONS[name](*inputs)
        if cold:
            _persist_windows_kernel(_FUNCTIONS[name].kernel_name)
    except (ImportError, AttributeError, OSError, RuntimeError) as error:
        message = str(error)
        if isinstance(error, RuntimeError) and not any(
            term in message.lower()
            for term in ("nvrtc", "jiterator", "compil", "not implemented")
        ):
            raise
        _FAILED[key] = message[-1600:]
        return None
    finally:
        if cold:
            _STATS["initialization_seconds"] += time.perf_counter() - started
    _WARM.add(key)
    _STATS["calls"] += 1
    return result
