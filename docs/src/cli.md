# The cnlp command

`cnlp` installs compiled models and solvers from the cnlplib registry and solves the models.

## Installing cnlp

```sh
curl -fsSL https://cnlplib.madsuite.org/install | sh
```

This puts `cnlp` and the solve driver `cnlp-solve` in `~/.cnlplib/bin`, and adds a marked
block to `~/.bashrc`, `~/.zshrc` and `~/.profile` that puts `~/.cnlplib/bin` on `PATH` and
`~/.cnlplib/lib` on `CNLPLIB_PATH` (where every cnlp consumer looks for libraries by name).
Options, after `sh -s --`: `-y` (do not ask), `--path DIR` (install elsewhere),
`--add-to-path no` (leave the startup files alone). Run it again to update `cnlp` itself.

`cnlp` is a POSIX shell script and needs `curl`, `tar` and `sha256sum` (or `shasum`).
`cnlp-solve` is a single binary for Linux x86_64 (glibc 2.28 or newer).

## Installing models, solvers and data

```sh
cnlp list                     # what the registry has for this platform, and what is installed
cnlp install lv ipopt         # a model library and a solver
cnlp install lv@nightly       # the latest build of the library's default branch
cnlp install lv@1.2.3         # a specific version
cnlp install --device cpu lv  # the host build even on a GPU machine
cnlp up                       # update everything installed (a pinned version stays)
cnlp remove lv                # remove it
cnlp path                     # the library directory
```

`cnlp list` shows one row per library, solver and data set: its releases, its backends (`cpu`, `cuda`, ...), what is installed (with the device of an installed library), and its description.
A library is published per backend; `cnlp install` takes the CUDA build when `nvidia-smi`
lists a GPU and the host (CPU) build otherwise, and falls back to the host build when a
release has no device build. `--device cpu|cuda` (or `CNLPLIB_DEVICE`) chooses instead.
Installing a library also installs the data sets it needs. Every download is checked
against the `SHA256SUMS` published with it; an install is swapped into place only once it
is complete.

| Installed | Where |
|---|---|
| model libraries | `~/.cnlplib/lib/<name>/` |
| shared artifacts, such as the official Julia runtime (installed once, linked into the libraries that use them) | `~/.cnlplib/artifacts/` |
| solvers | `~/.cnlplib/solvers/<name>/` |
| data sets | `~/.cnlplib/data/<name>/` |

A library carries its own copies of common libraries (OpenSSL, curl, libstdc++, ...).
`cnlp install` gives each runtime's copies private names (`libssl.so.3` is known as, say,
`ct2ssl.so.3` inside the store), so a program that has already loaded its own copies, such as
Python after `import hashlib`, or another library's runtime, keeps them apart. The files keep their names; it runs after
every install, and running it again changes nothing.

## Solving

```
cnlp solve LIB[:MODEL] [INSTANCE...] [-s START] [-o OUTPUT]... [--solver NAME] [key=value...]
```

**`LIB[:MODEL]`** names the model. `@name` is an installed library (looked up on
`CNLPLIB_PATH`, then in `~/.cnlplib/lib`); anything else is a path, to the library file or
its bundle directory. `:MODEL` picks one of the library's models and may be left out when it
has one. `cnlp solve @name` on a library with several models, or with a model it does not
have, lists its models and the instance arguments each takes:

```
$ cnlp solve @lv
cnlp solve: @lv has several models; give one as LIB:MODEL
models:
  @lv:Chained_HS46  int|size
  ...
  @lv:wood  int|size
```

```sh
cnlp solve @lv:wood 10000
cnlp solve ./build/mymodels/lib/libmymodels.so:acp pglib_opf_case118_ieee
```

**`INSTANCE...`** is what the model is built from. The model declares what it takes
(`P_argtype`), and `cnlp solve` reads the arguments accordingly:

| the model takes | write | example |
|---|---|---|
| nothing (a fixed model) | nothing | `cnlp solve @cutest:ROSENBR` |
| one integer | the integer | `cnlp solve @lv:wood 10000` |
| a file | a path, or the name of a file in an installed data set | `cnlp solve @emp:acp pglib_opf_case118_ieee` |
| a file in another format the model reads | `reader:path` | `cnlp solve @emp:acp psse:grid.raw` |
| several values (the builder) | one value per field, in order | `cnlp solve @cops:bearing 50 50` |

A data-set file is found by its full name, with or without the extension
(`pglib_opf_case118_ieee` finds `pglib/pglib_opf_case118_ieee.m`); nothing shorter is
matched. A name that matches no file is an error, listing the installed files whose names
contain it:

```
$ cnlp solve @emp:acp case118
cnlp solve: no file `case118`: not a path, and no installed data-set file of that name; did you mean
  pglib_opf_case118_ieee  (pglib/pglib_opf_case118_ieee.m)
  pglib_opf_case118_ieee__api  (pglib/api/pglib_opf_case118_ieee__api.m)
  pglib_opf_case118_ieee__sad  (pglib/sad/pglib_opf_case118_ieee__sad.m)
```

A model with reports prints them around the solve: what the instance is, before it, and a
summary of the solution after it (for an optimal power flow, its constraint violations by
group, say), as the model's library defines them.

After the solve, `cnlp solve` reports where the time went:

```
times: load model library 0.024 s, instantiate 0.002 s, load solver 2.233 s, solve 0.385 s, write 0.000 s, total 2.644 s
```

*load model library* is loading the model's shared library; a runtime it carries starts on
its first call, so it shows in *instantiate*, which also includes reading the instance's data
and a start. *load solver* includes what the solver does when it loads, such as starting its
runtime and warming up a GPU. *solve* is the call to the solver alone, and *total* runs from
the start of `cnlp solve` to the end of writing the outputs.

**`-s START`** warm-starts from `reader:path` (one of the model's start readers, such as a
solved case) or from a cnlp-solution HDF5 file written by an earlier solve; the start's
sizes must match the model.

**`-o OUTPUT`** writes the solution; give it more than once for several files.

| write | to get |
|---|---|
| `-o sol.h5` | the model's default format if it has one, else the standard cnlp-solution HDF5 file |
| `-o writer:path` | one of the model's own formats (`-o matpower:sol.m`) |
| `-o cnlp:path` | the standard HDF5 file, whatever the model's default |

The standard file is described in [formats/solution.md](../../formats/solution.md); its
multipliers follow the cnlp sign convention.

**`--solver NAME`** picks an installed solver (default `ipopt`), or a solver library by path
(`--solver ./libcnlp_mysolver.so`).

**`--device D`** is where the model and the solver run: `auto` (the default) takes the GPU
when the model's library has a CUDA tier and the solver runs there, `cpu` never does, `cuda`
requires it, and `cuda:N` is `cuda` on GPU N.

**`key=value`** options go to the solver as written: `tol=1e-8 max_iter=500 print_level=0`
for Ipopt. A solver refuses an option it does not know, by name.

**Ipopt with HSL.** The bundled Ipopt uses MUMPS. To use an HSL linear solver
(MA27, MA57, MA86, MA97), bring your own HSL library: Ipopt loads it at run time, so
nothing needs rebuilding.

```sh
cnlp solve @lv:wood 100000 linear_solver=ma27 hsllib=/path/to/libhsl.so
```

HSL is licensed separately: Coin-HSL from STFC (https://licences.stfc.ac.uk/product/coin-hsl,
free for academic use), or the library of Julia's HSL_jll package.

`cnlp solve` prints the model's size, the status, the objective, the iteration count and the
time, and exits with 0 when the status is `solved` or `acceptable`, 1 otherwise, and 2 for an
error before the solve.

### Many instances in one process

`--batch FILE` solves one instance per line of `FILE` (`-` reads standard input), one after
the other in one process, so the solver and each model library are loaded once. A line is
what follows `cnlp solve` for one instance; `#` starts a comment.

```
# runs.txt
@lv:wood 10000
@cops:glider 400
@emp:acp pglib_opf_case118_ieee -o case118.m
```

```sh
cnlp solve --batch runs.txt --results results.tsv --solver madnlp --device cuda:1 tol=1e-8
```

`--solver`, `--device` and `key=value` given on the command line apply to every line, after
the line's own. An error in one line (a missing model or file, a callback that fails) ends
that instance and the batch goes on. `--results` writes a tab-separated row per instance as
it finishes: the line, the status (or `error`), the objective, the iteration count, the
solve time, the time of the instance with its loading, and the error message. The last line
of the output counts the instances solved, not solved and in error; the exit code is 0 when
every instance was solved or acceptable.

## Your own HSL

The HSL linear solvers (MA27, MA57, MA77, MA86, MA97) are free for academic use but may not
be passed on, so no solver carries them and `cnlp` never downloads them. With an HSL_jll you
obtained from [STFC](https://licences.stfc.ac.uk/products/Software/HSL/LibHSL), a solver uses
them through `cnlp hsl add`:

```sh
cnlp hsl add ~/Downloads/HSL_jll.jl.v2025.7.21.tar.gz     # or its directory, or a dir with libhsl
cnlp solve @lv:wood 1000 --solver madnlp linear_solver=Ma57Solver
cnlp hsl                                                  # where it is
cnlp hsl remove
```

It copies the libraries for this platform beside the solver (`solvers/madnlp/lib/hsl`), and
points the libraries they need (libgfortran, libgomp, libmetis, libblastrampoline) at the
solver's own copies. A solver also takes an HSL from `JULIA_HSL_LIBRARY_PATH`,
`artifacts.hsl` in the settings below, or its option `hsl_library=DIR`.

## Settings

Local settings live in `$CNLPLIB_HOME/config.toml`, a TOML file of sections with string
values, kept with `cnlp config`:

```sh
cnlp config set artifacts.hsl /opt/hsl/lib     # where the user's own HSL is
cnlp config get artifacts.hsl
cnlp config list
cnlp config unset artifacts.hsl
```

A key is `SECTION.NAME`. The `[artifacts]` section says where an artifact is, in place of
the store's copy: `hsl`, the HSL libraries a user brings (cnlp never downloads them), or a
CUDA toolkit already installed on the machine. A solver reads it when it first needs the
artifact; an option or environment variable that names the same thing takes precedence.
`cnlp config set` rewrites the file, so comments added by hand are not kept.

```toml
[artifacts]
hsl = "/opt/hsl/lib"
"cuda-12.9" = "/usr/local/cuda/lib64"
```

## Environment

| variable | default | meaning |
|---|---|---|
| `CNLPLIB_HOME` | `~/.cnlplib` | where `cnlp` installs |
| `CNLPLIB_PATH` | (set by the installer) | directories searched for `@name` libraries, `:`-separated |
| `CNLPLIB_INDEX` | `https://cnlplib.madsuite.org/index.tsv` | the registry index |
| `CNLPLIB_BASE`, `CNLPLIB_RELEASES` | the published site and release | where the installer downloads `cnlp` and `cnlp-solve` |

## From other languages

The same libraries work without `cnlp solve`:

- Julia: [CNLPLib.jl](julia.md),
  `CNLPModel("@lv", :wood, 10000)`, solved with any NLPModels solver;
- Python: [cnlplib](python.md) (`import cnlplib`);
- C: link the library and declare its symbols with `cnlp.h` (see [example/solve.c](../../example/solve.c)).
