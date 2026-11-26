"""Solution files and the model's own formats (cnlplib.io), against the shared fixture
(conformance/tinyqp.c: `tq` has a `txt` writer and `txtn`/`txtw`/`txt` readers). The
standard file needs libcnlpio (solve/cnlpio.c), built here against the system's HDF5."""

import os
import pathlib
import shutil
import subprocess
import types

import numpy as np
import pytest

import cnlplib

ROOT = pathlib.Path(__file__).resolve().parents[3]
FIX = ROOT / "conformance" / "tinyqp.c"
CC = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    so = tmp_path_factory.mktemp("lib") / "libtinyqp.so"
    subprocess.run([CC, "-shared", "-fPIC", "-O2", f"-I{ROOT}", "-o", str(so), str(FIX)], check=True)
    return cnlplib.load(so)


def _hdf5_flags():
    """Compiler flags for the system's HDF5 (pkg-config, or Debian's serial layout)."""
    for pkg in ("hdf5", "hdf5-serial"):
        try:
            out = subprocess.run(["pkg-config", "--cflags", "--libs", pkg], capture_output=True, text=True)
        except FileNotFoundError:
            break
        if out.returncode == 0:
            return out.stdout.split()
    inc, lib = "/usr/include/hdf5/serial", "/usr/lib/x86_64-linux-gnu/hdf5/serial"
    if os.path.isfile(f"{inc}/hdf5.h"):
        return [f"-I{inc}", f"-L{lib}", "-lhdf5", f"-Wl,-rpath,{lib}"]
    return None


@pytest.fixture(scope="module")
def cnlpio(tmp_path_factory):
    if os.environ.get("CNLPIO_LIB"):
        return os.environ["CNLPIO_LIB"]
    flags = _hdf5_flags()
    if flags is None:
        pytest.skip("no HDF5 development files to build libcnlpio")
    so = tmp_path_factory.mktemp("io") / "libcnlpio.so"
    src = ROOT / "solve" / "cnlpio.c"
    subprocess.run(
        [CC, "-std=c99", "-O2", "-fPIC", "-shared", f"-I{ROOT}", "-o", str(so), str(src), *flags], check=True
    )
    os.environ["CNLPIO_LIB"] = str(so)
    return str(so)


def _result(m, x, y):
    return types.SimpleNamespace(
        solution=x,
        multipliers=y,
        multipliers_lower=np.zeros(m.nvar),
        multipliers_upper=np.zeros(m.nvar),
        objective=m.obj(x),
        status="solved",
        iterations=3,
        solver="test",
        solver_status="ok",
    )


def test_writers_and_readers(lib):
    m = cnlplib.CModel(lib, 4, prefix="tq")
    assert m.writers == ["txt"] and m.default_writer is None
    assert m.readers == {"txtn": "instance", "txtw": "parameters", "txt": "start"}
    assert cnlplib.readers(lib, "tq") == m.readers and cnlplib.default_reader(lib, "tq") is None
    assert cnlplib.readers(lib, "sq") == {}


def test_own_format_round_trip(lib, tmp_path):
    m = cnlplib.CModel(lib, 4, prefix="tq")
    x = np.array([0.5, 0.5, 1.0, 1.0])
    path = tmp_path / "x.txt"
    assert m.write_solution(path, _result(m, x, np.array([1.0])), format="txt") == str(path)
    s = m.read_start(path, reader="txt")
    assert isinstance(s, cnlplib.Start) and s.x.tolist() == x.tolist() and s.y.tolist() == [0.0]
    with pytest.raises(ValueError, match="no writer 'nope'"):
        m.write_solution(path, _result(m, x, np.zeros(1)), format="nope")
    with pytest.raises(ValueError, match="no start reader 'txtw'"):
        m.read_start(path, reader="txtw")


def test_instance_and_parameter_readers(lib, tmp_path):
    (tmp_path / "n.txt").write_text("6\n")
    m = cnlplib.CModel(lib, prefix="tq", input=("txtn", tmp_path / "n.txt"))
    assert m.nvar == 6
    (tmp_path / "w.txt").write_text("3 4\n")
    m.read_parameters("txtw", tmp_path / "w.txt")
    assert m["w"].tolist() == [3.0, 4.0]
    with pytest.raises(TypeError, match="not both"):
        cnlplib.CModel(lib, 4, prefix="tq", input=("txtn", tmp_path / "n.txt"))


def test_standard_file(lib, cnlpio, tmp_path):
    m = cnlplib.CModel(lib, 4, prefix="tq")
    m["w"] = [2.0, 5.0]
    x, y = np.array([0.5, 0.5, 1.0, 1.0]), np.array([1.0])
    path = tmp_path / "sol.h5"
    assert m.write_solution(path, _result(m, x, y)) == str(path)  # no default writer: the standard file
    s = m.read_start(path)
    assert s.x.tolist() == x.tolist() and s.y.tolist() == y.tolist()
    with pytest.raises(RuntimeError, match="the model needs 5"):
        cnlplib.CModel(lib, 5, prefix="tq").read_start(path)
    h5py = pytest.importorskip("h5py")
    with h5py.File(path) as f:
        assert f.attrs["format"] in (b"cnlp-solution", "cnlp-solution")
        assert f["solution/status"][()] in (b"solved", "solved") and f["solution/iterations"][()] == 3
        assert f["blocks/variables/x/value"][()].tolist() == x.tolist()
        assert f["blocks/constraints/budget/value"][()].tolist() == m.cons(x).tolist()
        assert f["blocks/parameters/w/value"][()].tolist() == [2.0, 5.0]
