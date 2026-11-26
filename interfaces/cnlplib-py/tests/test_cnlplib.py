"""Tests against the plain-C fixture, conformance/tinyqp.c."""

import os
import pathlib
import shutil
import subprocess

import numpy as np
import pytest

import cnlplib

# the shared fixture (conformance/tinyqp.c), compiled against this repository's cnlp.h
ROOT = pathlib.Path(__file__).resolve().parents[3]
FIX = ROOT / "conformance" / "tinyqp.c"


@pytest.fixture(scope="module")
def sopath(tmp_path_factory):
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    assert cc, "no C compiler available"
    so = tmp_path_factory.mktemp("lib") / "libtinyqp.so"
    subprocess.run([cc, "-shared", "-fPIC", "-O2", f"-I{ROOT}", "-o", str(so), str(FIX)], check=True)
    return so


@pytest.fixture(scope="module")
def lib(sopath):
    return cnlplib.load(sopath)


@pytest.fixture()
def m(lib):
    return cnlplib.CModel(lib, 4, prefix="tq")


def test_meta(m):
    assert (m.nvar, m.ncon, m.nnzj, m.nnzh) == (4, 1, 2, 4)
    assert np.all(m.x0 == 0.0)
    assert np.all(np.isinf(m.lvar)) and np.all(m.lvar < 0)
    assert np.all(np.isinf(m.uvar)) and np.all(m.uvar > 0)
    assert m.lcon.tolist() == [0.0] and m.ucon.tolist() == [0.0]
    assert m.minimize  # no sense tier, or P_minimize = 1


def test_evaluations_match_closed_form(m):
    x = np.array([0.5, 0.25, 2.0, -1.0])
    assert m.obj(x) == 0.25 + 0.5625 + 1.0 + 4.0
    assert np.array_equal(m.grad(x), 2.0 * (x - 1.0))
    assert m.cons(x).tolist() == [x[0] + x[1] - 1.0]
    rows, cols = m.jac_structure()
    assert rows.tolist() == [0, 0] and cols.tolist() == [0, 1]  # 0-based
    assert m.jac(x).tolist() == [1.0, 1.0]
    hr, hc = m.hess_structure()
    assert hr.tolist() == [0, 1, 2, 3] and hc.tolist() == [0, 1, 2, 3]
    assert m.hess(x, np.array([3.0]), obj_weight=0.7).tolist() == [1.4] * 4


def test_new_failure_raises(lib):
    with pytest.raises(RuntimeError, match="tq_new"):
        cnlplib.CModel(lib, 1, prefix="tq")


def test_multiple_instances_are_independent(lib):
    a = cnlplib.CModel(lib, 4, prefix="tq")
    x = np.array([0.5, 0.25, 2.0, -1.0])
    before = a.obj(x)
    b = cnlplib.CModel(lib, 6, prefix="tq")
    assert b.nvar == 6
    assert a.obj(x) == before


def test_solve_with_scipy(m):
    """Julia-free end-to-end: solve through scipy.optimize (SLSQP).

    min sum((x-1)^2) s.t. x1+x2=1 has optimum 0.5 at (.5, .5, 1, 1)."""
    scipy = pytest.importorskip("scipy.optimize")
    res = scipy.minimize(
        m.obj,
        m.x0,
        jac=m.grad,
        method="SLSQP",
        constraints=[{"type": "eq", "fun": m.cons, "jac": lambda x: np.array([[1.0, 1.0, 0.0, 0.0]])}],
    )
    assert res.success
    assert abs(res.fun - 0.5) < 1e-8
    assert np.allclose(res.x, [0.5, 0.5, 1.0, 1.0], atol=1e-6)


def test_name_based_loading(lib, tmp_path):
    shutil.copy(pathlib.Path(lib._name), tmp_path / "libtoy.so")
    cnlplib.set_path(tmp_path)
    l1 = cnlplib.lib("toy")
    assert cnlplib.lib("toy") is l1  # cached
    m = cnlplib.CModel("@toy", 4, prefix="tq")  # name-based
    assert m.nvar == 4
    # not found anywhere: the registry (a local, empty index) has no such library either
    empty = tmp_path / "index.tsv"
    empty.write_text("")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CNLPLIB_INDEX", empty.as_uri())
        mp.setenv("CNLPLIB_HOME", str(tmp_path / "home"))
        mp.setenv("PATH", f"{ROOT / 'bin'}:{os.environ['PATH']}")
        with pytest.raises(RuntimeError, match="cnlp install nonexistent"):
            cnlplib.lib("nonexistent")


def test_lone_integer_takes_the_one_knob_constructor(lib):
    """`tq` declares one scalar field and also exports tq_new."""
    assert [f["name"] for f in cnlplib.schema(lib, prefix="tq")["fields"]] == ["n"]
    m = cnlplib.CModel(lib, 5, prefix="tq")
    assert m.nvar == 5
    assert m.obj(np.zeros(5)) == 5.0  # sum (0 - 1)^2


def test_arguments_bind_positionally_to_the_schema(lib):
    """`sq` is builder-only (no sq_new) and declares three fields, so the
    arguments bind positionally in the order the schema publishes them."""
    assert [f["name"] for f in cnlplib.schema(lib, prefix="sq")["fields"]] == ["n", "s", "w"]
    n, s, w = 4, 2.0, np.array([1.0, 2.0, 3.0, 4.0])
    m = cnlplib.CModel(lib, n, s, w, prefix="sq")
    assert (m.nvar, m.ncon) == (4, 1)
    x = np.array([0.5, 0.25, 2.0, -1.0])
    assert m.obj(x) == np.sum(w * (x - s) ** 2)  # min sum w_i (x_i - s)^2
    assert np.array_equal(m.grad(x), 2.0 * w * (x - s))
    assert np.array_equal(m.hess(x, np.array([3.0]), obj_weight=0.7), 1.4 * w)
    assert m.cons(x).tolist() == [x[0] + x[1] - 1.0]


def test_argument_order_is_load_bearing(lib):
    """The same values in the wrong order are refused by the slot they land in: n is the
    int64 scalar, so the float meant for s cannot go there."""
    with pytest.raises(TypeError, match=r"'n' is an int64"):
        cnlplib.CModel(lib, 2.0, 4, np.array([1.0, 2.0, 3.0, 4.0]), prefix="sq")
    # The same guard on arrays and table columns, which the fixture's schema
    # has no int64 instance of.
    assert cnlplib.model._as_col([1.0, 2.0], "f64").dtype == np.float64
    with pytest.raises(TypeError, match=r"int64 array"):
        cnlplib.model._as_col(np.array([1.5, 2.5]), "i64", "bus.i")


def test_wrong_arity_names_the_schema(lib):
    with pytest.raises(ValueError, match=r"declares 3 fields"):
        cnlplib.CModel(lib, 4, 2.0, prefix="sq")
    with pytest.raises(ValueError, match=r"declares 1 field"):
        cnlplib.CModel(lib, 5, 6, prefix="tq")
    # No arguments at all is the no-instance-data case; both schemas want some.
    with pytest.raises(ValueError, match=r"given 0 arguments"):
        cnlplib.CModel(lib, prefix="tq")
    with pytest.raises(ValueError, match=r"given 0 arguments"):
        cnlplib.CModel(lib, prefix="sq")


def test_inconsistent_structured_data_is_the_librarys_call(lib):
    """w must be as long as n says — the library decides, not this wrapper."""
    with pytest.raises(RuntimeError, match="incomplete"):
        cnlplib.CModel(lib, 4, 2.0, np.array([1.0, 2.0]), prefix="sq")


def test_a_bool_is_not_an_argument(lib):
    with pytest.raises(TypeError):
        cnlplib.CModel(lib, True, prefix="tq")


def test_fixed_library_needs_no_arguments(lib):
    # fx_nargs() == 0: no arguments instantiate it through fx_new, whose integer is ignored
    m0 = cnlplib.CModel(lib, prefix="fx")
    assert (m0.nvar, m0.ncon) == (3, 1)
    assert m0.obj(np.array([0.5, 0.5, 1.0])) == 0.5
    # An explicit integer still works, and lands on the same fixed model.
    m1 = cnlplib.CModel(lib, 999, prefix="fx")
    assert m1.nvar == 3


def test_string_is_at_name_on_search_path_or_literal_path(sopath, tmp_path, monkeypatch):
    spec = str(sopath)
    # `@name` resolves on the search path and defaults the prefix to the name.
    assert cnlplib.model._is_name("@toy")
    assert cnlplib.model._default_prefix("@toy") == "toy"

    # Anything else is a path; the prefix defaults from the file name (libtinyqp.so → tinyqp).
    assert not cnlplib.model._is_name(spec)
    assert cnlplib.model._default_prefix(spec) == "tinyqp"
    mp = cnlplib.CModel(spec, 4, prefix="tq")
    assert mp.nvar == 4
    # The handle is cached by absolute path: one dlopen per library.
    assert cnlplib.model._resolve_spec(spec) is cnlplib.model._resolve_spec(spec)
    # A fixed model by path needs nothing beyond the path.
    mf = cnlplib.CModel(spec, prefix="fx")
    assert mf.nvar == 3

    # A path to a bundle DIRECTORY finds the library inside it, and the
    # prefix defaults from the directory name.
    bdir = tmp_path / "toyqp2"
    (bdir / "lib").mkdir(parents=True)
    shutil.copy(sopath, bdir / "lib" / "libtoyqp2.so")
    assert cnlplib.model._default_prefix(str(bdir)) == "toyqp2"
    md = cnlplib.CModel(str(bdir), 4, prefix="tq")
    assert md.nvar == 4

    # A bare string without `@` is a file in the current directory, not a search-path name.
    shutil.copy(sopath, tmp_path / "qp.so")
    monkeypatch.chdir(tmp_path)
    mc = cnlplib.CModel("qp.so", 4, prefix="tq")
    assert mc.nvar == 4
    with pytest.raises(FileNotFoundError, match="no shared library at"):
        cnlplib.CModel("toy", 4)

    # A path that is not there fails as a path, never as a search-path miss.
    with pytest.raises(FileNotFoundError, match="no shared library at"):
        cnlplib.CModel(str(sopath.parent / "libnope.so"), 1)


def test_search_path_initializes_from_the_environment(monkeypatch, tmp_path):
    cnlplib.set_path()  # empty the registry
    monkeypatch.setenv("CNLPLIB_PATH", f"/a:{tmp_path}")
    assert cnlplib.model._paths() == ["/a", str(tmp_path)]
    cnlplib.set_path()  # leave no residue for later tests


def test_a_directory_without_a_library_says_what_it_tried(tmp_path):
    with pytest.raises(FileNotFoundError, match="no shared library in"):
        cnlplib.CModel(str(tmp_path), 1)


def test_value_guards_name_the_field_and_kind():
    with pytest.raises(TypeError, match="bool is not a model argument"):
        cnlplib.model._as_scalar("n", True, "i64")
    with pytest.raises(TypeError, match="is a float64"):
        cnlplib.model._as_scalar("s", "x", "f64")
    with pytest.raises(TypeError, match="float64 array"):
        cnlplib.model._as_col(np.array(["a"]), "f64", "w")
    with pytest.raises(ValueError, match="one-dimensional"):
        cnlplib.model._as_col(np.ones((2, 2)), "f64", "w")


def test_fill_data_validates_keys_against_the_schema(lib):
    with pytest.raises(ValueError, match="do not match schema"):
        cnlplib.model._fill_data(lib, "sq", {"n": 4})


def test_a_table_field_binds_column_by_column(lib):
    # min Σ_j w_j (x_{i_j} - 1)^2 over 3 vars: rows (1, 1.0), (2, 2.0),
    # (3, 3.0), (1, 0.5) — variable 1 carries weight 1.5.
    pts = {"i": np.array([1, 2, 3, 1]), "w": np.array([1.0, 2.0, 3.0, 0.5])}
    m = cnlplib.CModel(lib, pts, prefix="tb")
    assert (m.nvar, m.ncon, m.nnzj, m.nnzh) == (3, 0, 0, 3)
    x = np.array([0.0, 2.0, 4.0])
    assert m.obj(x) == 1.5 * 1.0 + 2.0 * 1.0 + 3.0 * 9.0
    assert m.grad(x).tolist() == [2 * 1.5 * -1.0, 2 * 2.0 * 1.0, 2 * 3.0 * 3.0]
    hr, hc = m.hess_structure()
    assert hr.tolist() == [0, 1, 2] and hc.tolist() == [0, 1, 2]
    assert m.hess(x, np.zeros(0), obj_weight=0.5).tolist() == [1.5, 2.0, 3.0]

    # Validation happens before any column crosses the boundary.
    with pytest.raises(ValueError, match="columns"):
        cnlplib.CModel(lib, {"i": [1], "x": [1.0]}, prefix="tb")
    with pytest.raises(ValueError, match="column lengths differ"):
        cnlplib.CModel(lib, {"i": [1, 2], "w": [1.0]}, prefix="tb")
    # A column the library refuses surfaces as its setter's status.
    with pytest.raises(RuntimeError, match="set_col_i64"):
        cnlplib.CModel(lib, {"i": [1] * 9, "w": [1.0] * 9}, prefix="tb")


def test_each_builder_failure_names_its_stage(lib):
    with pytest.raises(RuntimeError, match="data_begin failed"):
        cnlplib.CModel(lib, 4, prefix="bf")
    with pytest.raises(RuntimeError, match="new_from_data failed"):
        cnlplib.CModel(lib, 4, prefix="nf")
    with pytest.raises(RuntimeError, match="no builder surface"):
        cnlplib.CModel(lib, 2.0, prefix="fx")
    with pytest.raises(RuntimeError, match="ns_schema does not"):
        cnlplib.CModel(lib, 4, prefix="ns")
    # A lone integer for a builder-only library falls through to the schema.
    with pytest.raises(ValueError, match="declares 3 fields"):
        cnlplib.CModel(lib, 4, prefix="sq")


def test_a_failing_fixed_constructor_is_reported(lib):
    with pytest.raises(RuntimeError, match=r"zz_new\(0\) failed"):
        cnlplib.CModel(lib, prefix="zz")


def test_evaluation_shape_guards(lib):
    m = cnlplib.CModel(lib, 4, prefix="tq")
    with pytest.raises(ValueError, match=r"x must have shape \(4,\)"):
        m.obj(np.zeros(3))
    with pytest.raises(ValueError, match=r"y must have shape \(1,\)"):
        m.hess(np.zeros(4), np.zeros(2))


# ── Selecting a model by name in a library carrying several ──────────────────
# A leading string argument names one; the name is the symbol prefix.


def test_model_selection_by_name(lib):
    x = np.array([0.5, 0.25, 2.0, -1.0])
    n, s, w = 4, 2.0, np.array([1.0, 2.0, 3.0, 4.0])

    m = cnlplib.CModel(lib, "tq", 4)
    assert m.nvar == 4
    ms = cnlplib.CModel(lib, "sq", n, s, w)  # builder-only sibling
    assert m.obj(x) == ((x - 1.0) ** 2).sum()
    assert ms.obj(x) == (w * (x - s) ** 2).sum()

    # Instances of DIFFERENT models coexist as freely as instances of one.
    m6 = cnlplib.CModel(lib, "tq", 6)
    assert m6.nvar == 6
    assert m.obj(x) == ((x - 1.0) ** 2).sum()
    assert ms.obj(x) == (w * (x - s) ** 2).sum()


def test_unknown_model_name_is_refused_clearly(lib):
    # a mistyped name is reported at selection, not as an `undefined symbol` later
    with pytest.raises(ValueError, match=r"carries no model named 'nosuch'"):
        cnlplib.CModel(lib, "nosuch", 4)
    with pytest.raises(ValueError, match=r"carries no model named"):
        cnlplib.schema(lib, "nosuch")


def test_leading_string_selects_unless_prefix_settles_it(lib):
    # With no prefix=, a leading string selects; a matching one is harmless.
    assert cnlplib.CModel(lib, "tq", 4).nvar == 4
    assert cnlplib.CModel(lib, "tq", 4, prefix="tq").nvar == 4
    # With an explicit different prefix the string is an argument: a lone string routes to
    # the string entry point (sq has none), several arguments to the builder.
    with pytest.raises(ValueError, match=r"no string entry point"):
        cnlplib.CModel(lib, "tq", prefix="sq")
    with pytest.raises(ValueError, match=r"declares 3 fields"):
        cnlplib.CModel(lib, "tq", 4, prefix="sq")


def test_a_string_argument_is_not_mistaken_for_a_model_name(sopath, tmp_path):
    """`ps` instantiates from one string, so on a library whose default prefix (from its
    file name) is `ps`, `CModel(spec, "5")` passes the string through."""
    as_ps = tmp_path / "libps.so"
    shutil.copy(sopath, as_ps)
    assert cnlplib.CModel(str(as_ps), "5").nvar == 5


def test_schema_by_model_name(lib):
    sch = cnlplib.schema(lib, "sq")
    assert [f["name"] for f in sch["fields"]] == ["n", "s", "w"]


def test_at_name_with_model_selection(lib, tmp_path):
    shutil.copy(pathlib.Path(lib._name), tmp_path / "libtoy9.so")
    cnlplib.set_path(tmp_path)
    m = cnlplib.CModel("@toy9", "tq", 4)
    assert m.nvar == 4
    assert cnlplib.lib("toy9") is cnlplib.model._resolve_spec("@toy9")  # one cache entry


# ── Named blocks: the fixture publishes `x` (variable), `budget` (constraint), `w`
# (parameter) ─────────────────────────────────────────────────────────────────


def test_named_blocks(lib):
    m = cnlplib.CModel(lib, 4, prefix="tq")
    assert list(m.variables) == ["x"]
    assert list(m.constraints) == ["budget"]
    assert list(m.parameters) == ["w"]
    assert len(m.variables) == 1 and "x" in m.variables and "w" not in m.variables
    assert dict(m.variables) == {"x": m.variables["x"]}

    b = m.variables["x"]
    assert (b.kind, b.offset, b.length, b.dims) == ("var", 0, 4, (4,))
    # Lengths follow the INSTANCE, not the library.
    assert cnlplib.CModel(lib, 7, prefix="tq").variables["x"].length == 7

    # Result extraction reshapes the block's own slice.
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert cnlplib.solution(x, b).tolist() == [1.0, 2.0, 3.0, 4.0]
    assert cnlplib.multipliers(np.array([7.0]), m.constraints["budget"]).tolist() == [7.0]
    assert cnlplib.multipliers_lower(x, b).tolist() == [1.0, 2.0, 3.0, 4.0]
    assert cnlplib.multipliers_upper(x, b).tolist() == [1.0, 2.0, 3.0, 4.0]


def test_named_block_lookup_errors(lib):
    m = cnlplib.CModel(lib, 4, prefix="tq")
    with pytest.raises(KeyError, match=r"is a parameter \(m.parameters\), not a variable"):
        m.variables["w"]
    with pytest.raises(KeyError, match=r"is a variable \(m.variables\), not a parameter"):
        m.parameters["x"]
    with pytest.raises(KeyError, match=r"no named variable 'nope'"):
        m.variables["nope"]
    assert m.variables.get("nope") is None  # the Mapping protocol


def test_parameter_values(lib):
    m = cnlplib.CModel(lib, 4, prefix="tq")
    p = m.parameters["w"]
    m[p] = [3.0, 4.0]
    assert m[p].tolist() == [3.0, 4.0]
    assert m["w"].tolist() == [3.0, 4.0]  # by name too
    m["w"] = [5.0, 6.0]
    assert m[p].tolist() == [5.0, 6.0]
    with pytest.raises(ValueError, match="has 2 elements, got 1"):
        m[p] = [1.0]
    with pytest.raises(ValueError, match="is a variable, not a parameter"):
        m[m.variables["x"]]
    with pytest.raises(TypeError, match="BlockRef"):
        m[3]


def test_library_without_named_blocks(lib):
    # Named blocks are optional in the ABI: `sq` publishes none, and everything
    # else about it still works.
    ms = cnlplib.CModel(lib, 4, 2.0, np.array([1.0, 2.0, 3.0, 4.0]), prefix="sq")
    assert dict(ms.variables) == {}
    with pytest.raises(KeyError, match="publishes no named blocks"):
        ms.variables["x"]
    assert ms.nvar == 4


def test_catalogue_and_argtype(lib):
    assert cnlplib.available_models(lib) == ["tq", "sq", "fx", "tb", "ps"]
    # a prefix without a published signature answers "", not an exception
    assert cnlplib.argtype(lib, "tq") == "int|size"
    assert cnlplib.argtype(lib, "sq") == "int|n,f64|s,Vector{f64}|w"
    assert cnlplib.argtype(lib, "fx") == ""
    assert cnlplib.argtype(lib, "bf") == ""


def test_string_instantiation(lib):
    # a lone string routes to `<prefix>_new_str`, as a lone integer routes to `<prefix>_new`
    assert cnlplib.argtype(lib, "ps") == "string|size, in decimal"
    m = cnlplib.CModel(lib, "5", prefix="ps")
    assert m.nvar == 5
    assert m.obj(np.zeros(5)) == 4.0 * 5
    # A model without the entry point refuses with its signature.
    with pytest.raises(ValueError, match=r"no string entry point.*int\|size"):
        cnlplib.CModel(lib, "5", prefix="tq")
    # The library's own refusal (unparsable string) surfaces as an error.
    with pytest.raises(RuntimeError, match="ps_new_str"):
        cnlplib.CModel(lib, "not-a-number", prefix="ps")


def test_reports_are_empty_for_a_library_without_them(m):
    assert m.report_instance() == ""
    assert m.report_solution(np.zeros(4)) == ""


def test_backend_is_none_or_cuda(lib):
    with pytest.raises(ValueError, match="'cuda'"):
        cnlplib.CModel(lib, 4, prefix="tq", backend="tpu")
