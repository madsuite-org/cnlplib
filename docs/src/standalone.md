# Without cnlp

A library or solver from the registry is an ordinary C shared library, and it can be used
without `cnlp`: from your own C program, or loaded into Julia or Python by path. `cnlp
install` does three things you then do by hand: it downloads the asset, adds the
artifacts the asset lists, and gives the bundled libraries private names. This page covers
the first two. The third is optional (see the end).

## 1. Download and unpack

Each library's or solver's registry entry (`libraries/<name>.toml` or `solvers/<name>.toml`)
names its GitHub repository. Its releases hold `<name>-<platform>.tar.gz` and a
`SHA256SUMS`:

```sh
curl -fLO https://github.com/<owner>/<repo>/releases/download/<tag>/lv-linux-x86_64.tar.gz
curl -fLO https://github.com/<owner>/<repo>/releases/download/<tag>/SHA256SUMS
sha256sum --check --ignore-missing SHA256SUMS
tar -xzf lv-linux-x86_64.tar.gz          # -> lv/
```

## 2. Add the artifacts it lists

If the unpacked directory has an `ARTIFACTS` file, the library is not complete without the
artifacts it lists, one per line: `ARTIFACT[/SUBDIR] TARGET [DEVICE]`.

- `julia-VERSION`: the `lib/julia` directory of that official Julia release
  (julialang.org), linked into `TARGET`. A model library shares its Julia runtime this way,
  and does not load without it.
- `NAME-HASH`: `NAME-HASH.artifact.tar.gz` from the same release (checked against the same
  `SHA256SUMS`), unpacked; the files of `NAME-HASH/` (or of its `SUBDIR`) are linked into
  `TARGET`.
- A line with a `DEVICE` (`cuda`) is needed only on that device: madnlp's CUDA libraries,
  for example. Skip it for CPU use.

[scripts/link-artifacts.sh](../../scripts/link-artifacts.sh) does the linking. Give it
the unpacked directory, and the directory you unpacked the `NAME-HASH` artifacts into;
it takes `julia-VERSION` from the `julia` on `PATH`, which must be that version (or set
`JULIA_LIBDIR` to that Julia's `lib/julia`):

```sh
curl -fLO https://raw.githubusercontent.com/madsuite-org/cnlplib/main/scripts/link-artifacts.sh
JULIA_LIBDIR=/path/to/julia-1.12.6/lib/julia sh link-artifacts.sh lv
```

For madnlp on a GPU:

```sh
curl -fLO .../cuda-<hash>.artifact.tar.gz && mkdir -p artifacts && tar -xzf cuda-<hash>.artifact.tar.gz -C artifacts
sh link-artifacts.sh madnlp artifacts                 # LINK_DEVICES="" leaves the cuda part out
```

The links point to where you unpacked the artifacts, so keep those directories.

## 3. Use it

- **C**: `dlopen` `lib/lib<name>.so` (a model library, [cnlp.h](../../cnlp.h)) or
  `lib/libcnlp_<name>.so` (a solver, [cnlp_solver.h](../../cnlp_solver.h)); the bundle
  finds its own libraries through `$ORIGIN`.
- **Julia**: `CNLPModel("/path/to/lv", :wood, 1000)` (the bundle directory or its library file), see
  [Julia](julia.md).
- **Python**: `cnlplib.CModel("/path/to/lv", "wood", 1000)`, see [Python](python.md).
- **HSL** (madnlp): the solver option `hsl_library=DIR`, or `JULIA_HSL_LIBRARY_PATH`,
  pointing at your own HSL's `lib` directory. See "Your own HSL" in [the cnlp command](cli.md).

## Private names

`cnlp install` also renames the bundled libraries (OpenSSL, libstdc++, BLAS, the CUDA
libraries) to private names, so that they cannot collide with copies that something else in
the same process has loaded under the standard names: Python's OpenSSL, or the CUDA
libraries of CuPy, PyTorch or CUDA.jl. A library set up by hand keeps the standard names.
It works on its own, and in a process that loads nothing else of the same name; for a
process that does, install it with `cnlp`.
