"""Libraries from the cnlplib registry, against a registry served from local files laid out as
the releases are: an index, a release with its SHA256SUMS, a shared artifact the library names
in its ARTIFACTS file, and a data set the library requires. Everything is installed into a
temporary CNLPLIB_HOME, in the layout the `cnlp` command and CNLPLib.jl use."""
import hashlib
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
import cnlplib
from cnlplib import registry

FIX = pathlib.Path(__file__).parent / "fixtures" / "tinyqp.c"
SHARED = "shared_x-0123456789abcdef"


def _sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def _tgz(src_dir, name, out):
    with tarfile.open(out, "w:gz") as t:
        t.add(os.path.join(src_dir, name), arcname=name)


@pytest.fixture()
def reg(tmp_path, monkeypatch):
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    assert cc, "no C compiler available"
    plat = registry._platform()
    rel, stage, home = tmp_path / "releases", tmp_path / "stage", tmp_path / "home"
    for d in (rel, stage / "regqp" / "lib", stage / SHARED, stage / "regdata" / "cases"):
        d.mkdir(parents=True)
    subprocess.run([cc, "-shared", "-fPIC", "-O2", "-o", str(stage / "regqp" / "lib" / "libregqp.so"),
                    str(FIX)], check=True)
    (stage / "regqp" / "ARTIFACTS").write_text(f"{SHARED} lib/extra\n")
    (stage / SHARED / "note.txt").write_text("shared\n")
    (stage / "regdata" / "cases" / "case_tiny.txt").write_text("4\n")
    _tgz(stage, "regqp", rel / f"regqp-{plat}.tar.gz")
    _tgz(stage, SHARED, rel / f"{SHARED}.artifact.tar.gz")
    _tgz(stage, "regdata", rel / "regdata.tar.gz")
    (rel / f"badqp-{plat}.tar.gz").write_text("not the release")
    sums = [f"{_sha(rel / f)}  {f}" for f in sorted(os.listdir(rel)) if not f.startswith("badqp")]
    sums.append(f"{_sha(rel / f'regqp-{plat}.tar.gz')}  badqp-{plat}.tar.gz")   # bytes that do not match
    (rel / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    u = lambda f: (rel / f).as_uri()
    index = tmp_path / "index.tsv"
    index.write_text("\n".join([
        f"regqp\tlatest\t{plat}\t{u(f'regqp-{plat}.tar.gz')}\t-\ta test library\tlib\tregdata\t-\tcpu",
        f"badqp\tlatest\t{plat}\t{u(f'badqp-{plat}.tar.gz')}\t-\ta tampered library\tlib\t\t-\tcpu",
        f"regdata\tlatest\tany\t{u('regdata.tar.gz')}\t-\ta test data set\tdata\t\t-\t-",
    ]) + "\n")
    monkeypatch.setenv("CNLPLIB_INDEX", index.as_uri())
    monkeypatch.setenv("CNLPLIB_DEVICE", "cpu")
    monkeypatch.setenv("CNLPLIB_HOME", str(home))
    monkeypatch.delenv("CNLPLIB_PATH", raising=False)
    monkeypatch.delenv("CNLPMODELS_PATH", raising=False)
    cnlplib.set_path()
    yield home, index
    cnlplib.set_path()


def test_installed_on_first_use_where_cnlp_puts_it(reg):
    home, _ = reg
    m = cnlplib.CModel("@regqp", 4, prefix="tq")
    assert m.nvar == 4
    d = home / "lib" / "regqp"
    assert d.is_dir()
    assert (d / ".cnlplib-version").read_text().split()[2] == "regqp"
    note = d / "lib" / "extra" / "note.txt"
    assert note.read_text() == "shared\n" and note.is_symlink()
    assert os.readlink(note) == str(home / "artifacts" / SHARED / "note.txt")
    assert (home / "data" / "regdata" / ".cnlplib-version").is_file()


def test_data_files_by_name(reg):
    cnlplib.lib("regqp")
    assert cnlplib.data_file("case_tiny").endswith(os.path.join("cases", "case_tiny.txt"))
    assert cnlplib.data_file("case_tiny.txt") == cnlplib.data_file("case_tiny")
    with pytest.raises(FileNotFoundError, match="did you mean: case_tiny"):
        cnlplib.data_file("tiny")
    assert registry.string_arg("5") == "5"            # not a file: as given


def test_offline_after_install_and_updates(reg):
    _, index = reg
    d = cnlplib.install("regqp")
    index.rename(index.with_suffix(".off"))            # the index is gone, the library is not
    cnlplib.set_path()
    assert cnlplib.CModel("@regqp", 3, prefix="tq").nvar == 3
    index.with_suffix(".off").rename(index)
    assert cnlplib.update("regqp") == d                # same release: kept


def test_bytes_not_matching_the_published_checksum_are_refused(reg):
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        cnlplib.install("badqp")
