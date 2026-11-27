"""Solution files and a model's own file formats.

The standard cnlp-solution file (formats/solution.md) is written and read by libcnlpio,
the library the `cnlp` command installs (`$CNLPLIB_HOME/lib/libcnlpio.so`) and CNLPLib.jl
uses too, so a solution file is the same whichever wrote it. A library's own formats (a
MATPOWER case, ...) are its readers and writers, called directly. Multipliers are in the
cnlp sign convention throughout, as `cnlplib.solve` returns them.
"""

from __future__ import annotations

import ctypes
import os
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from . import registry

if TYPE_CHECKING:
    from .model import CModel

_i64, _dbl, _vp, _cp = ctypes.c_int64, ctypes.c_double, ctypes.c_void_p, ctypes.c_char_p
_pd = ctypes.POINTER(ctypes.c_double)
_KINDS = ("instance", "parameters", "start")


class Start(NamedTuple):
    """A starting point: `x`, and the multipliers `y`, `zl`, `zu` (cnlp sign convention)."""

    x: np.ndarray
    y: np.ndarray
    zl: np.ndarray
    zu: np.ndarray


_IO = []


def _cnlpio():
    if not _IO:
        path = os.environ.get("CNLPIO_LIB") or os.path.join(registry._home(), "lib", "libcnlpio.so")
        if not os.path.isfile(path):
            raise RuntimeError(
                f"the standard solution file needs libcnlpio ({path}), installed with the cnlp "
                "command: curl -fsSL https://cnlplib.madsuite.org/install | sh"
            )
        lib = ctypes.CDLL(path)
        w = lib.cnlpio_write_solution
        w.restype = _i64
        w.argtypes = [_cp, _vp, _cp, _cp, _i64, _i64, _i64, _i64, _pd, _pd, _pd, _pd, _dbl, _cp, _i64]
        w.argtypes += [_cp, _cp, ctypes.c_char_p, _i64]
        r = lib.cnlpio_read_start
        r.restype = _i64
        r.argtypes = [_cp, _i64, _i64, _pd, _pd, _pd, _pd, ctypes.c_char_p, _i64]
        _IO.append(lib)
    return _IO[0]


def _host(v, n):
    if v is None:
        return np.zeros(n)
    v = v.get() if hasattr(v, "get") and not isinstance(v, np.ndarray) else v
    a = np.ascontiguousarray(v, dtype=np.float64)
    if a.shape != (n,):
        raise ValueError(f"expected an array of shape ({n},), got {a.shape}")
    return a


def _p(a):
    return a.ctypes.data_as(_pd)


def _names(m: CModel, count: str, name: str) -> list[str]:
    from .model import _copyout, _f, _pu8

    try:
        nf = _f(m._lib, m._prefix, count, [])
    except AttributeError:
        return []
    fn = _f(m._lib, m._prefix, name, [_i64, _pu8, _i64])
    return [_copyout(fn, _i64(k)) for k in range(int(nf()))]


def writers(m: CModel) -> list[str]:
    """The solution writers the library carries for this model."""
    return _names(m, "nwriters", "writer_name")


def default_writer(m: CModel) -> str | None:
    """The writer the library names as this model's default, or None."""
    from .model import _copyout, _f, _pu8

    try:
        fn = _f(m._lib, m._prefix, "default_writer", [_i64, _pu8, _i64])
    except AttributeError:
        return None
    return _copyout(fn, _i64(m._id)) or None


def readers(lib: Any, model: str) -> dict[str, str]:
    """The model's own input readers, `name -> kind` (`instance`, `parameters`, `start`)."""
    from .model import _copyout, _f, _pu8, _resolve_spec

    lib = _resolve_spec(lib) if isinstance(lib, str) else lib
    try:
        nf = _f(lib, model, "nreaders", [])
    except AttributeError:
        return {}
    fn = _f(lib, model, "reader_name", [_i64, _pu8, _i64])
    fk = _f(lib, model, "reader_kind", [_i64])
    return {_copyout(fn, _i64(k)): _KINDS[int(fk(_i64(k)))] for k in range(int(nf()))}


def default_reader(lib: Any, model: str) -> str | None:
    """The instance reader the model uses for a bare path, or None."""
    from .model import _copyout, _f, _pu8, _resolve_spec

    lib = _resolve_spec(lib) if isinstance(lib, str) else lib
    try:
        fn = _f(lib, model, "default_reader", [_pu8, _i64])
    except AttributeError:
        return None
    return _copyout(fn) or None


def _reader_of_kind(lib, model, name, kind):
    rs = readers(lib, model)
    if rs.get(name) != kind:
        raise ValueError(f"{model!r} has no {kind} reader {name!r}; its readers are {rs or 'none'}")


def instantiate_from(lib, model: str, reader: str, path: str) -> int:
    """An instance read from the model's own format by one of its instance readers."""
    from .model import _f

    _reader_of_kind(lib, model, reader, "instance")
    mid = int(_f(lib, model, f"new_from_{reader}", [_cp])(os.fspath(path).encode()))
    if mid <= 0:
        raise RuntimeError(f"{model}_new_from_{reader} could not create an instance from {path}")
    return mid


def write_solution(m: CModel, path: str, result: Any, format: str | None = None) -> str:
    """Write `result` (a `cnlplib.solve` result, or anything with `solution`,
    `multipliers`, `multipliers_lower`, `multipliers_upper`, `objective` and `status`) in
    the model's default format: the library's default writer if it names one, otherwise
    the standard cnlp-solution file. `format` forces `"cnlp"` (the standard file) or one of
    `m.writers`."""
    from .model import _f

    n, k = m.nvar, m.ncon
    x = _host(result.solution, n)
    y = _host(getattr(result, "multipliers", None), k)
    zl = _host(getattr(result, "multipliers_lower", None), n)
    zu = _host(getattr(result, "multipliers_upper", None), n)
    objective = float(getattr(result, "objective", m.obj(x) if m.backend is None else np.nan))
    status = str(getattr(result, "status", ""))
    path = os.fspath(path)
    f = format if format is not None else (default_writer(m) or "cnlp")
    if f != "cnlp":
        if f not in writers(m):
            raise ValueError(f"{m._prefix!r} has no writer {f!r}; it has {writers(m)}")
        w = _f(m._lib, m._prefix, f"write_{f}", [_i64, _cp, _pd, _pd, _pd, _pd, _dbl, _cp])
        if int(w(_i64(m._id), path.encode(), _p(x), _p(y), _p(zl), _p(zu), objective, status.encode())):
            raise RuntimeError(f"writer {f!r} of {m._prefix!r} could not write {path}")
        return path
    err = ctypes.create_string_buffer(512)
    rc = _cnlpio().cnlpio_write_solution(
        path.encode(),
        m._lib._handle,
        os.fspath(m._lib._name).encode(),
        m._prefix.encode(),
        m._id,
        n,
        k,
        1 if m.minimize else 0,
        _p(x),
        _p(y),
        _p(zl),
        _p(zu),
        objective,
        status.encode(),
        int(getattr(result, "iterations", -1)),
        str(getattr(result, "solver", "")).encode(),
        str(getattr(result, "solver_status", "")).encode(),
        err,
        len(err),
    )
    if rc:
        raise RuntimeError(err.value.decode(errors="replace"))
    return path


def read_start(m: CModel, path: str, reader: str | None = None) -> Start:
    """A starting point from a cnlp-solution file, or with `reader` from the model's own
    format by one of its start readers."""
    from .model import _f

    n, k = m.nvar, m.ncon
    x, y, zl, zu = np.zeros(n), np.zeros(k), np.zeros(n), np.zeros(n)
    path = os.fspath(path)
    if reader is not None:
        _reader_of_kind(m._lib, m._prefix, reader, "start")
        fn = _f(m._lib, m._prefix, f"start_from_{reader}", [_i64, _cp, _pd, _pd, _pd, _pd])
        if int(fn(_i64(m._id), path.encode(), _p(x), _p(y), _p(zl), _p(zu))):
            raise RuntimeError(f"reader {reader!r} of {m._prefix!r} could not read {path}")
    else:
        err = ctypes.create_string_buffer(512)
        if _cnlpio().cnlpio_read_start(path.encode(), n, k, _p(x), _p(y), _p(zl), _p(zu), err, len(err)):
            raise RuntimeError(err.value.decode(errors="replace"))
    return Start(x, y, zl, zu)


def read_parameters(m: CModel, reader: str, path: str) -> None:
    """Set the model's parameters from its own format, with one of its parameter readers."""
    from .model import _f

    _reader_of_kind(m._lib, m._prefix, reader, "parameters")
    fn = _f(m._lib, m._prefix, f"set_from_{reader}", [_i64, _cp])
    if int(fn(_i64(m._id), os.fspath(path).encode())):
        raise RuntimeError(f"reader {reader!r} of {m._prefix!r} could not read {path}")
