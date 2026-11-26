"""Libraries from the cnlplib registry, through the `cnlp` command.

``lib("lv")`` (so ``CModel("@lv", ...)``) falls back to the libraries installed in
``$CNLPLIB_HOME`` (default ``~/.cnlplib``), and installs one that is not there with
``cnlp install``: one store, and one installer, for the ``cnlp`` command, Julia and Python.
"""
import os
import shutil
import subprocess
import sys

__all__ = ["data_file", "install", "update"]

_KINDS = ("lib", "data", "solvers")


def _home():
    return os.environ.get("CNLPLIB_HOME", os.path.join(os.path.expanduser("~"), ".cnlplib"))


def _ext():
    return {"win32": ".dll", "darwin": ".dylib"}.get(sys.platform, ".so")


def _cnlp():
    """The `cnlp` command: the installer's, else on PATH."""
    c = os.path.join(_home(), "bin", "cnlp")
    if os.path.isfile(c):
        return c
    c = shutil.which("cnlp")
    if c is None:
        raise RuntimeError("the `cnlp` command is not installed; install it with\n"
                           "    curl -fsSL https://cnlplib.madsuite.org/install | sh")
    return c


def _installed(name):
    """What is installed under `name`: (version, sha256, request), or None."""
    for d in _KINDS:
        f = os.path.join(_home(), d, name, ".cnlplib-version")
        if os.path.isfile(f):
            with open(f) as io:
                p = io.readline().split()
            return (p[0] if p else "", p[1] if len(p) > 1 else "", p[2] if len(p) > 2 else name)
    return None


def install(name):
    """Install registry library, solver or data set `name` ("lv", "emp@nightly", "lv@1.2.0")
    with `cnlp install name`, into $CNLPLIB_HOME (default ~/.cnlplib), and return its
    directory. `lib` calls this for a name it finds nowhere else, so
    `CModel("@lv", "wood", 1000)` works on first use."""
    # cnlp says what went wrong on stderr; here, that it did
    if subprocess.run([_cnlp(), "install", name], check=False).returncode != 0:
        raise RuntimeError(f"`cnlp install {name}` failed")
    n = name.partition("@")[0]
    for d in _KINDS:
        p = os.path.join(_home(), d, n)
        if os.path.isdir(p):
            return p
    raise RuntimeError(f"`cnlp install {name}` installed nothing under {_home()}")


def update(name):
    """Install registry library `name` again if its current release differs from the
    installed one; see `install`."""
    from . import _LIBS
    _LIBS.pop(name, None)
    return install(name)


def registry_lib(name):
    """A registry library's file: installed (for `name`, under any channel; for
    `name@channel`, installed with that request), else installed now."""
    n = name.partition("@")[0]
    f = os.path.join(_home(), "lib", n, "lib", f"lib{n}{_ext()}")
    cur = _installed(n)
    if not (os.path.isfile(f) and (n == name or (cur is not None and cur[2] == name))):
        install(name)
    if not os.path.isfile(f):
        raise RuntimeError(f"the registry asset of {name!r} has no lib/lib{n}{_ext()}")
    return f


def _find_data(name):
    """The data-set file `name`, or None with the names that contain it."""
    if os.path.isfile(name) or "/" in name:
        return name, []
    near = []
    for root, _, files in os.walk(os.path.join(_home(), "data")):
        for f in files:
            if f == name or os.path.splitext(f)[0] == name:
                return os.path.join(root, f), near
            if name in f:
                near.append(os.path.splitext(f)[0])
    return None, sorted(set(near))[:10]


def data_file(name):
    """A file of an installed data set by name, as `cnlp solve` resolves one: a path as it
    is, else the file of that exact name, or that name without its extension, under
    $CNLPLIB_HOME/data. A name that matches no file is an error that lists the files whose
    names contain it."""
    f, near = _find_data(name)
    if f is not None:
        return f
    raise FileNotFoundError(
        f"no file {name!r}: not a path, and no installed data-set file of that name"
        + (f"; did you mean: {', '.join(near)}" if near else ""))


def string_arg(s):
    """A model's string argument: a data-set file when one has that name, else the string
    as given (the model's own function decides what it means)."""
    f, _ = _find_data(s)
    return f if f is not None else s
