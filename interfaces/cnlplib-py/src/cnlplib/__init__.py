"""A cnlp library (cnlp ABI 0.2, ``cnlp.h``) as a numpy-backed model, through ctypes.

``CModel`` exposes ``nvar``, ``ncon``, ``nnzj``, ``nnzh``, ``x0``, ``lvar``, ``uvar``,
``lcon``, ``ucon``, ``minimize`` and the methods ``obj(x)``, ``grad(x)``, ``cons(x)``,
``jac_structure()``, ``jac(x)``, ``hess_structure()``, ``hess(x, y, obj_weight=1.0)``:
numpy in, numpy out, structure indices 0-based (the ABI's are 1-based).
"""

from .model import (
    BlockRef,
    CModel,
    argtype,
    available_models,
    lib,
    load,
    multipliers,
    multipliers_lower,
    multipliers_upper,
    schema,
    set_path,
    solution,
)
from .registry import data_file, install, update
from .solve import Result, Solver, load_solver, solve

__version__ = "0.1.0"

__all__ = [
    "BlockRef",
    "CModel",
    "Result",
    "Solver",
    "argtype",
    "available_models",
    "data_file",
    "install",
    "lib",
    "load",
    "load_solver",
    "multipliers",
    "multipliers_lower",
    "multipliers_upper",
    "schema",
    "set_path",
    "solution",
    "solve",
    "update",
]
