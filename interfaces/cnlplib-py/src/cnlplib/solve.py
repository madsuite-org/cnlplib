"""Solve a model with a cnlp solver library (`cnlp_solver.h`), in this process.

A solver installed in the store (`cnlp install ipopt`, `cnlp install madnlp`) is loaded once
and kept: its start-up (a runtime, a GPU warm-up) is paid by the first solve only. The model
library's own functions are handed to it as the problem's callbacks, so the solver evaluates
the model with no Python in between: on the host, or on the GPU (through the library's device
tier) for a model created with `backend="cuda"`. They are handed over as they are: a cnlp callback takes `(void* ctx, ...)` and a model
function `(int64_t id, ...)`, which a 64-bit platform passes in the same register, so the
instance id is the context.
"""
import ctypes
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from . import registry

__all__ = ["Result", "load_solver", "solve"]

_i64, _dbl, _vp = ctypes.c_int64, ctypes.c_double, ctypes.c_void_p
_STATUS = ("solved", "acceptable", "infeasible", "unbounded", "iteration_limit", "time_limit", "error")


class _Problem(ctypes.Structure):
    _fields_ = [("ctx", _vp), ("nvar", _i64), ("ncon", _i64), ("nnzj", _i64), ("nnzh", _i64),
                ("minimize", _i64)] + [(k, _vp) for k in (
                    "x0", "lvar", "uvar", "lcon", "ucon", "y0", "zl0", "zu0",
                    "obj", "grad", "cons", "jac_structure", "jac", "hess_structure", "hess")]


class _DeviceProblem(ctypes.Structure):
    _fields_ = [("device", _i64)] + [(k, _vp) for k in ("obj_d", "grad_d", "cons_d", "jac_d", "hess_d")]


class _Result(ctypes.Structure):
    _fields_ = [("x", _vp), ("y", _vp), ("zl", _vp), ("zu", _vp), ("objective", _dbl),
                ("status", _i64), ("iterations", _i64), ("solver_status", ctypes.c_char * 128)]


@dataclass
class Result:
    """A solve: the cnlp status (`solved`, `acceptable`, `infeasible`, ...), the objective,
    the iteration count, the solution and multipliers (cnlp sign convention), and the
    solver's name, own status and device."""
    status: str
    objective: float
    iterations: int
    solution: np.ndarray
    multipliers: np.ndarray
    multipliers_L: np.ndarray
    multipliers_U: np.ndarray
    solver: str
    solver_status: str
    device: str
    seconds: float = field(default=0.0)


class Solver:
    """A cnlp solver library, loaded. `name` is a solver installed in the store (installed
    there with `cnlp install` when it is not) or a path to its library."""

    def __init__(self, name):
        if os.sep in name:
            path = name
        else:
            n = name.partition("@")[0]
            path = os.path.join(registry._home(), "solvers", n, "lib", f"libcnlp_{n}{registry._ext()}")
            if not os.path.isfile(path):
                registry.install(name)
        flags = os.RTLD_LOCAL | getattr(os, "RTLD_DEEPBIND", 0)
        self._lib = ctypes.CDLL(path, mode=flags)
        ver = self._lib.cnlp_solver_version
        ver.restype = _i64
        if int(ver()) // 1000 != 0:
            raise RuntimeError(f"{path} is not a cnlp solver of interface 0.x")
        buf = (ctypes.c_uint8 * 64)()
        fn = self._lib.cnlp_solver_name
        fn.restype, fn.argtypes = _i64, [ctypes.POINTER(ctypes.c_uint8), _i64]
        n = int(fn(buf, 63))
        self.name = bytes(buf[:max(n, 0)]).decode()
        self._solve = self._lib.cnlp_solve
        self._solve.restype = _i64
        self._solve.argtypes = [ctypes.POINTER(_Problem), _i64, _vp, _vp, ctypes.POINTER(_Result)]
        try:
            self._solve_d = self._lib.cnlp_solve_device
            self._solve_d.restype = _i64
            self._solve_d.argtypes = [ctypes.POINTER(_Problem), ctypes.POINTER(_DeviceProblem), _i64,
                                      _vp, _vp, ctypes.POINTER(_Result)]
        except AttributeError:
            self._solve_d = None

    def __repr__(self):
        return f"cnlplib.Solver({self.name!r}{', cpu and cuda' if self._solve_d else ''})"

    def solve(self, model, x0=None, **options):
        """Solve `model` (a `CModel`) where it lives: on the GPU for a model created with
        `backend="cuda"`, on the host otherwise. `x0` replaces the starting point; other
        keywords are solver options (`tol=1e-8`, `max_iter=100`)."""
        lib, pre, mid = model._lib, model._prefix, model._id

        def fp(name):
            return ctypes.cast(getattr(lib, f"{pre}_{name}"), _vp).value

        on_device = model.backend == "cuda"
        if on_device and self._solve_d is None:
            raise ValueError(f"the model is on the GPU (backend='cuda') and {self.name} does not "
                             "run there: use a GPU solver (cumadnlp), or a model on the host")
        n, m = model.nvar, model.ncon
        x0 = np.ascontiguousarray(model.x0 if x0 is None else _host(x0), dtype=np.float64)
        keep = [x0, model.lvar, model.uvar, model.lcon, model.ucon]
        ptr = lambda a: a.ctypes.data
        p = _Problem(_vp(mid), n, m, model.nnzj, model.nnzh, 1 if model.minimize else 0,
                     *(ptr(a) for a in keep), None, None, None,
                     fp("obj"), fp("grad"), fp("cons"), fp("jac_structure"), fp("jac"),
                     fp("hess_structure"), fp("hess"))
        out = [np.zeros(n), np.zeros(m), np.zeros(n), np.zeros(n)]
        r = _Result(*(ptr(a) for a in out), 0.0, 6, -1, b"")
        keys = [str(k).encode() for k in options]
        vals = [str(v).encode() for v in options.values()]
        ka, va = (ctypes.c_char_p * max(len(keys), 1))(*keys), (ctypes.c_char_p * max(len(vals), 1))(*vals)
        t = time.perf_counter()
        if on_device:
            d = _DeviceProblem(1, fp("obj_d"), fp("grad_d"), fp("cons_d"), fp("jac_d"), fp("hess_d"))
            rc = self._solve_d(ctypes.byref(p), ctypes.byref(d), len(keys), ctypes.cast(ka, _vp),
                               ctypes.cast(va, _vp), ctypes.byref(r))
        else:
            rc = self._solve(ctypes.byref(p), len(keys), ctypes.cast(ka, _vp), ctypes.cast(va, _vp), ctypes.byref(r))
        seconds = time.perf_counter() - t
        msg = r.solver_status.decode(errors="replace")
        if rc != 0:
            raise RuntimeError(f"{self.name} failed: {msg}")
        if on_device:       # the solution where the model lives
            out = [model._xp.asarray(a) for a in out]
        return Result(status=_STATUS[r.status if 0 <= r.status <= 6 else 6], objective=r.objective,
                      iterations=int(r.iterations), solution=out[0], multipliers=out[1],
                      multipliers_L=out[2], multipliers_U=out[3], solver=self.name, solver_status=msg,
                      device="cuda" if on_device else "cpu", seconds=seconds)


_SOLVERS = {}


def load_solver(name):
    """The cnlp solver `name`, loaded once per process (see `Solver`)."""
    if name not in _SOLVERS:
        _SOLVERS[name] = Solver(name)
    return _SOLVERS[name]


def solve(model, solver="ipopt", x0=None, **options):
    """Solve `model` (a `CModel`) with the cnlp solver `solver`, in this process, where the
    model lives (its `backend`); the solver is loaded on first use and kept. See `Solver.solve`.

        m = cnlplib.CModel("@emp", "acp", "pglib_opf_case118_ieee", backend="cuda")
        r = cnlplib.solve(m, solver="cumadnlp")
    """
    if sys.maxsize <= 2**32:
        raise RuntimeError("cnlplib.solve needs a 64-bit platform")
    return load_solver(solver).solve(model, x0=x0, **options)


def _host(x):
    return x.get() if hasattr(x, "get") and not isinstance(x, np.ndarray) else x
