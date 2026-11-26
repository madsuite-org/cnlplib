"""Libraries from the cnlplib registry, installed into the one store every cnlp tool shares.

``lib("lv")`` (so ``CModel("@lv", ...)``) falls back to the registry when the library is
neither on the search path nor installed: the registry's index
(https://cnlplib.madsuite.org/index.tsv) names the release asset of each library for each
platform, and the asset is downloaded, checked against the SHA-256 its release publishes, and
installed into ``$CNLPLIB_HOME`` (default ``~/.cnlplib``) exactly as the ``cnlp`` command and
CNLPLib.jl install it: ``lib/NAME``, ``data/NAME``, ``solvers/NAME``, the shared pieces an asset
names in its ARTIFACTS file once in ``artifacts/`` and linked in, and a ``.cnlplib-version``
record. A library is stored once for the ``cnlp`` command, Julia and Python alike.
"""
import glob
import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

__all__ = ["data_file", "install", "update"]

DEFAULT_INDEX = "https://cnlplib.madsuite.org/index.tsv"
_DIRS = {"lib": "lib", "data": "data", "solver": "solvers"}


def _index_url():
    return os.environ.get("CNLPLIB_INDEX", DEFAULT_INDEX)


def _home():
    return os.environ.get("CNLPLIB_HOME", os.path.join(os.path.expanduser("~"), ".cnlplib"))


def _ext():
    return {"win32": ".dll", "darwin": ".dylib"}.get(sys.platform, ".so")


def _platform():
    """This machine as the index names platforms: linux-x86_64, macos-aarch64, ..."""
    os_ = {"linux": "linux", "darwin": "macos", "win32": "windows"}.get(sys.platform)
    if os_ is None:
        raise RuntimeError(f"unsupported operating system {sys.platform} for the cnlplib registry")
    m = platform.machine().lower()
    arch = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}.get(m)
    if arch is None:
        raise RuntimeError(f"unsupported architecture {m} for the cnlplib registry")
    return f"{os_}-{arch}"


def _devices():
    """The devices to prefer, best first: cuda when nvidia-smi lists a GPU
    (CNLPLIB_DEVICE=cpu|cuda overrides)."""
    d = os.environ.get("CNLPLIB_DEVICE", "")
    if d == "cpu":
        return []
    if d == "cuda":
        return ["cuda"]
    if d:
        raise RuntimeError(f"CNLPLIB_DEVICE is cpu or cuda, not {d!r}")
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    try:
        out = subprocess.run([smi, "-L"], capture_output=True, text=True, timeout=30, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return ["cuda"] if re.search(r"^GPU", out, re.MULTILINE) else []


def _fetch(url):
    with urllib.request.urlopen(url) as r:
        return r.read().decode()


def _index():
    try:
        text = _fetch(_index_url())
    except OSError as e:
        raise RuntimeError(f"could not download the cnlplib index from {_index_url()}: {e}") from e
    rows = []
    for line in text.splitlines():
        f = line.split("\t")
        if len(f) < 7:
            continue
        rows.append({
            "name": f[0], "label": f[1], "platform": f[2], "url": f[3], "sha256": f[4],
            "kind": f[6], "requires": [r for r in (f[7].split(",") if len(f) > 7 else []) if r],
            "versioned": f[8] if len(f) > 8 else "-",
        })
    return rows


def _candidates(rows, key):
    plat = _platform()
    order = [f"{plat}-{d}" for d in _devices()] + [plat, "any"]
    return [e for p in order for e in rows if e["name"] == key and e["platform"] == p]


def _published_sha(url, file):
    """The SHA-256 of `file` in the SHA256SUMS published next to `url`, or None."""
    try:
        text = _fetch(url.rsplit("/", 1)[0] + "/SHA256SUMS")
    except OSError:
        return None
    for line in text.splitlines():
        f = line.split()
        if len(f) == 2 and f[1].lstrip("*") == file:
            return f[0]
    return None


def _hex(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _download_checked(url, path, want):
    try:
        with urllib.request.urlopen(url) as r, open(path, "wb") as f:
            shutil.copyfileobj(r, f)
    except OSError as e:
        raise RuntimeError(f"download failed: {url} ({e})") from e
    got = _hex(path)
    if got != want:
        raise RuntimeError(f"checksum mismatch for {url}: expected {want}, got {got}")
    return path


def _unpack(tarball, dest):
    with tarfile.open(tarball, "r:gz") as t:
        if hasattr(tarfile, "data_filter"):   # Python 3.12+: refuse links and paths leaving dest
            t.extractall(dest, filter="tar")
        else:
            t.extractall(dest)
    return dest


def _single_dir(x):
    fs = os.listdir(x)
    return os.path.join(x, fs[0]) if len(fs) == 1 and os.path.isdir(os.path.join(x, fs[0])) else x


def _installed(name):
    """What is installed under `name`: (version, sha256, request), or None."""
    for d in _DIRS.values():
        f = os.path.join(_home(), d, name, ".cnlplib-version")
        if os.path.isfile(f):
            with open(f) as io:
                p = io.readline().split()
            return (p[0] if p else "", p[1] if len(p) > 1 else "", p[2] if len(p) > 2 else name)
    return None


def _place(src, dest):
    """Move `src` to `dest`; another process may have put it there first."""
    try:
        os.rename(src, dest)
    except OSError:
        if not os.path.isdir(dest):
            raise
    return dest


def _shared(a, asset_url):
    """A shared piece an asset names in its ARTIFACTS file, installed once into
    $CNLPLIB_HOME/artifacts: julia-X.Y.Z, the lib/julia of that official Julia release
    (copied from a juliaup install of exactly that version when there is one);
    <name>-<hex>, the <name>-<hex>.artifact.tar.gz of the asset's own release."""
    art = os.path.join(_home(), "artifacts")
    os.makedirs(art, exist_ok=True)
    m = re.fullmatch(r"julia-(\d+\.\d+\.\d+)", a)
    if m:
        v = m[1]
        dest = os.path.join(art, f"{a}-{_platform()}")
        if os.path.isdir(dest):
            return dest
        depot = os.environ.get("JULIAUP_DEPOT_PATH", os.path.join(os.path.expanduser("~"), ".julia"))
        for d in sorted(glob.glob(os.path.join(depot, "juliaup", f"julia-{v}+*"))):
            src = os.path.join(d, "lib", "julia")
            if os.path.isfile(os.path.join(d, "lib", f"libjulia.so.{v}")) and os.path.isdir(src):
                # copied, not linked: removing that Julia with juliaup must not break the libraries
                stage = tempfile.mkdtemp(prefix=f".stage-{a}.", dir=art)
                try:
                    shutil.copytree(src, os.path.join(stage, "julia"), symlinks=True)
                    _place(os.path.join(stage, "julia"), dest)
                finally:
                    shutil.rmtree(stage, ignore_errors=True)
                return dest
        plat = _platform()
        sub = {"linux-x86_64": "linux/x64", "linux-aarch64": "linux/aarch64"}.get(plat)
        if sub is None:
            raise RuntimeError(f"no official Julia download known for {plat}")
        file = f"julia-{v}-{plat}.tar.gz"
        mm = ".".join(v.split(".")[:2])
        url = f"https://julialang-s3.julialang.org/bin/{sub}/{mm}/{file}"
        want = None
        for line in _fetch(f"https://julialang-s3.julialang.org/bin/checksums/julia-{v}.sha256").splitlines():
            f = line.split()
            if len(f) == 2 and f[1] == file:
                want = f[0]
        if want is None:
            raise RuntimeError(f"no published checksum for {file}")
        stage = tempfile.mkdtemp(prefix=f".stage-{a}.", dir=art)
        try:
            t = _download_checked(url, os.path.join(stage, "j.tar.gz"), want)
            x = _unpack(t, os.path.join(stage, "x"))
            _place(os.path.join(x, f"julia-{v}", "lib", "julia"), dest)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
        return dest
    if re.fullmatch(r"[a-z0-9_]+-[0-9a-f]{16,64}", a):
        dest = os.path.join(art, a)
        if os.path.isdir(dest):
            return dest
        file = a + ".artifact.tar.gz"
        url = asset_url.rsplit("/", 1)[0] + "/" + file
        want = _published_sha(url, file)
        if want is None:
            raise RuntimeError(f"no checksum for {file} in the SHA256SUMS of {url.rsplit('/', 1)[0]}")
        stage = tempfile.mkdtemp(prefix=f".stage-{a}.", dir=art)
        try:
            t = _download_checked(url, os.path.join(stage, "a.tar.gz"), want)
            x = _unpack(t, os.path.join(stage, "x"))
            if not os.path.isdir(os.path.join(x, a)):
                raise RuntimeError(f"{file} has no top-level {a}/")
            _place(os.path.join(x, a), dest)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
        return dest
    raise RuntimeError(f"an asset names an invalid shared artifact {a!r}")


def install(req):
    """Install registry library, solver or data set `req` ("lv", "emp@nightly", "lv@1.2.0")
    into $CNLPLIB_HOME (default ~/.cnlplib), as `cnlp install req` does, with the data sets it
    requires, and return its directory. The same bytes already installed are kept. On a
    machine with an NVIDIA GPU a library's CUDA build is taken when its release carries one
    (CNLPLIB_DEVICE=cpu takes the host build)."""
    return _install(_index(), req)


def update(name):
    """Install registry library `name` again if its current release differs from the
    installed one; see `install`."""
    from . import _LIBS
    _LIBS.pop(name, None)
    return install(name)


def _install(rows, req):
    n, _, sel = req.partition("@")
    key = f"{n}@nightly" if sel == "nightly" else n
    cands = _candidates(rows, key)
    if not cands:
        raise FileNotFoundError(f"no {req!r} for {_platform()} in the cnlplib index ({_index_url()})")
    for r in cands[0]["requires"]:
        _install(rows, r)
    entry = url = want = None
    ver = ""
    for e in cands:
        u, v = e["url"], e["label"]
        if sel not in ("", "latest", "nightly"):   # a pinned version
            if e["versioned"] in ("", "-"):
                continue
            u, v = e["versioned"].replace("{version}", sel), sel
        w = _published_sha(u, u.rsplit("/", 1)[1]) if e["sha256"] == "-" or u != e["url"] else e["sha256"]
        if w is not None:
            entry, url, want, ver = e, u, w, v
            break
    if entry is None:
        hint = f"; its nightly build is {n}@nightly" if not sel and _candidates(rows, f"{n}@nightly") else ""
        raise RuntimeError(f"no released build of {req!r} for {_platform()}{hint}")
    dest = os.path.join(_home(), _DIRS[entry["kind"]])
    cur = _installed(n)
    if cur is not None and cur[1] == want:
        # the same bytes: record the request, so `cnlp up` follows the channel asked for last
        with open(os.path.join(dest, n, ".cnlplib-version"), "w") as io:
            io.write(f"{cur[0]} {want} {req}\n")
        return os.path.join(dest, n)
    os.makedirs(dest, exist_ok=True)
    stage = tempfile.mkdtemp(prefix=f".stage-{n}.", dir=dest)
    try:
        t = _download_checked(url, os.path.join(stage, "asset.tar.gz"), want)
        src = _single_dir(_unpack(t, os.path.join(stage, "x")))
        if entry["kind"] == "lib" and not os.path.isfile(os.path.join(src, "lib", f"lib{n}{_ext()}")):
            raise RuntimeError(f"the asset of {req!r} has no lib/lib{n}{_ext()}")
        if entry["kind"] == "solver" and not os.path.isfile(os.path.join(src, "lib", f"libcnlp_{n}{_ext()}")):
            raise RuntimeError(f"the asset of {req!r} has no lib/libcnlp_{n}{_ext()}")
        # shared artifacts, each installed once and its files linked into the library's
        # TARGET (a file the library carries wins)
        af = os.path.join(src, "ARTIFACTS")
        if os.path.isfile(af):
            with open(af) as io:
                lines = io.read().splitlines()
            for line in lines:
                f = line.split()
                if not f:
                    continue
                if len(f) != 2:
                    raise RuntimeError(f"the asset of {req!r} has a malformed ARTIFACTS line: {line}")
                a, tgt = f
                if tgt.startswith("/") or ".." in tgt.split("/"):
                    raise RuntimeError(f"the asset of {req!r} links into {tgt!r}, outside the library")
                ad = _shared(a, url)
                d = os.path.join(src, tgt)
                os.makedirs(d, exist_ok=True)
                for g in os.listdir(ad):
                    p = os.path.join(d, g)
                    if not os.path.lexists(p):
                        os.symlink(os.path.join(ad, g), p)
        # the version the asset says it is, if it says
        vf = os.path.join(src, "VERSION")
        if os.path.isfile(vf):
            with open(vf) as io:
                for line in io:
                    m = re.match(r"^version *= *(.+)$", line.rstrip("\n"))
                    if m:
                        ver = m[1].strip()
                        break
        with open(os.path.join(src, ".cnlplib-version"), "w") as io:
            io.write(f"{ver} {want} {req}\n")
        # swap into place: a failed install never leaves a half-installed library
        old = os.path.join(dest, f".old-{n}")
        shutil.rmtree(old, ignore_errors=True)
        if os.path.isdir(os.path.join(dest, n)):
            os.rename(os.path.join(dest, n), old)
        os.rename(src, os.path.join(dest, n))
        shutil.rmtree(old, ignore_errors=True)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return os.path.join(dest, n)


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
    d = os.path.join(_home(), "data")
    for root, _, files in os.walk(d):
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
