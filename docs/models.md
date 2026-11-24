# Publishing a model library

This guide is for anyone who wants their models installable with `cnlp install` and usable
from `cnlp solve`, Julia (CNLPLib), Python (cnlplib) and C. It covers the library, its
checks, its release, and its registry entry.

## 1. The library

A cnlp library is a shared library that implements [cnlp.h](../cnlp.h) (cnlp ABI 0.2). The
header is the specification; read its comments. In short:

- **Models and prefixes.** Each model exports its symbols under a prefix `P`
  (`wood_obj`, `wood_grad`, ...). A library with more than one model also exports the
  library level: `cnlp_nmodels`, `cnlp_model_name` and `cnlp_abi_version` (returning 2).
  Export the library level in every library: consumers use it to find the models and to
  tell 0.2 from 0.1.
- **Required per model:** `P_nargs`, `P_argtype`, an entry point (`P_new`, `P_new_str` or
  the builder), and the evaluation core: `P_nvar`, `P_ncon`, `P_nnzj`, `P_nnzh`, `P_meta`,
  `P_obj`, `P_grad`, `P_cons`, `P_jac_structure`, `P_jac`, `P_hess_structure`, `P_hess`.
- **Conventions.** Every integer is `int64_t`. Sparse structures are 1-based COO. The
  Hessian is the lower triangle of `obj_weight * Hess f + sum_i y_i Hess c_i`. Functions
  return 0 on success (ids and lengths are positive; 0 or -1 on failure) and never throw or
  abort across the boundary. Strings are copied out: `P_name(buf, cap)` returns the needed
  length and copies what fits. Instances are integer ids, live for the process, and
  several may coexist.
- **Instantiation.** `P_argtype` says what an instance is built from: `""` for a fixed
  model, `"int|size"` for `P_new(n)`, `"string|..."` for `P_new_str(path)`, or one
  `type|description` item per builder field. `cnlp solve` passes its instance arguments
  accordingly.
- **Optional tiers** (each detected by one witness symbol): named blocks of variables,
  constraints and parameters; the objective sense (`P_minimize`, export it for a
  maximization); solution writers and readers of the model's own file formats; device
  (GPU) evaluation; binary32 and other number formats. Add those you can answer honestly.
- **Multiplier signs** in writers and start readers follow
  [formats/solution.md](../formats/solution.md):
  `s grad f + J'y - zL + zU = 0` with `s = 1` (min) or `-1` (max), `zL, zU >= 0`.

You can write the library in any language that produces a C shared library.
[example/qp.c](../example/qp.c) is a complete one in about 300 lines of C. Julia models
written with ExaModels are compiled into one with ExaModelsCompiler
(`compile_library`, or `compile_all` for a package's whole collection; see its docs).

## 2. Checking it

- `scripts/smoke.c` loads the library in a plain C process, instantiates every model that
  takes one integer or nothing, and evaluates it:
  `cc -std=c99 -o smoke scripts/smoke.c -ldl -lm && ./smoke path/to/libNAME.so 10`.
  A library that carries a runtime must pass this from its packed form (step 3): a process
  that is not the runtime's own language has nothing preloaded.
- `scripts/check-library.jl` evaluates every model through CNLPLib and, when the library
  has the device tier and a GPU is present, compares device callbacks with host ones.
- `cnlp solve path/to/libNAME.so:MODEL ARGS...` solves one with an installed solver.
- [conformance/](../conformance/) holds reference files for the input and solution formats.

## 3. Packing it

One archive per platform, named `<name>-<platform>.tar.gz`, where `<name>` is the registry
name (lowercase, `[a-z0-9_]`) and `<platform>` is `<os>-<arch>` (`linux-x86_64`,
`linux-aarch64`, `macos-aarch64`, `windows-x86_64`). A library with a device tier is also
published per backend: `<name>-<platform>.tar.gz` holds the host (CPU) build only, and
`<name>-<platform>-<device>.tar.gz` (`-cuda`) the host build with that device tier; `cnlp
install` takes the one the machine supports. The name carries no version, so each
release's asset has a stable URL. Its single top-level directory is the bundle:

```
<name>/
  lib/lib<name>.so        the library (lib<name>.dylib, <name>.dll elsewhere)
  lib/...                 anything it loads, found through $ORIGIN (rpath) or the bundle
  LICENSE                 the models' license
  THIRD_PARTY_NOTICES     licenses of everything else in the bundle
  VERSION                 name, version, cnlp_abi, platform, device (key = value lines)
  ARTIFACTS               optional: shared artifacts it needs, `ARTIFACT TARGET` per line
```

A library can share files with other libraries through **artifacts**, listed in its
`ARTIFACTS` file as `ARTIFACT TARGET` lines. `cnlp install` installs each once into
`~/.cnlplib/artifacts/` and links each of its files into the library's `TARGET` directory,
unless the library carries a file of that name; a library installed later that names the
same artifact reuses it. `scripts/link-artifacts.sh` does the linking for a check. An
artifact is either

- `julia-VERSION`: the `lib/julia` of that official Julia release, taken from a juliaup
  install of exactly that version if there is one, else downloaded from julialang.org and
  checked against its published SHA-256; or
- `NAME-HASH`: an archive published in the library's own release as
  `NAME-HASH.artifact.tar.gz` (holding `NAME-HASH/`), HASH being over its content, checked
  against the release's `SHA256SUMS`.

`scripts/package-library.sh` shares the Julia runtime as `julia-VERSION`: every file of
`lib/julia` that is byte for byte the official Julia's is left out of the library, except
each runtime's privatized `libjulia` and `libjulia-internal`, which stay; a file that
differs (a newer `libstdc++` a dependency brings) stays too.

If the bundle carries GPL or LGPL libraries (a bundled Julia runtime does: SuiteSparse,
GMP, MPFR), attach their corresponding source to the same release;
`scripts/gpl-sources.sh JULIA_VERSION OUTDIR NAME` collects it for a Julia runtime.

## 4. Releasing it

Each release of your repository carries the archives and a `SHA256SUMS` listing them
(`sha256sum *.tar.gz > SHA256SUMS`):

- a release tagged `v<version>` for each version (`cnlp install name@1.2.3`; the latest one
  is what `cnlp install name` takes);
- optionally a rolling release tagged `nightly`, replaced on every build of your default
  branch (`cnlp install name@nightly`).

A Julia package whose models are compiled with ExaModelsCompiler can use the reusable
workflow, which builds, checks, packs and publishes all of this:

```yaml
# .github/workflows/Library.yml
on:
  push: { branches: [main], tags: ["v*"] }
  pull_request:
  workflow_dispatch:
jobs:
  library:
    uses: madsuite-org/cnlplib/.github/workflows/library.yml@main
    with: { name: lv }            # also: device, julia-version, runs-on, check-size
    permissions: { contents: write }
```

It calls `ExaModelsCompiler.compile_all(YourPackage; ...)`, so the package implements
`compile_all` for itself (an extension on ExaModelsCompiler). A package with many models can
build them in parallel with `chunks: N`: N jobs each compile one part
(`compile_all(...; chunk = (k, N))`, which the package's method accepts), and a last job
joins them into one bundle whose `lib<name>.so` lists the models and resolves each model's
symbols into its part (`scripts/assemble-chunks.sh`). The archive installs and loads as one
library; each part keeps its own Julia runtime, started when one of its models is first
used. Version tags come from
TagBot; give TagBot an SSH key (`ssh: ${{ secrets.DOCUMENTER_KEY }}`), since a tag pushed
with the default token starts no workflow. Other producers publish the same archives with
their own workflow.

## 5. Registering it

Open a pull request to [madsuite-org/cnlplib](https://github.com/madsuite-org/cnlplib)
adding `libraries/<name>.toml`:

```toml
name        = "lv"                                    # = file name and archive name
description = "Luksan-Vlcek test problems"
producer    = "LuksanVlcekBenchmark.jl"
homepage    = "https://github.com/madsuite-org/LuksanVlcekBenchmark.jl"
license     = "MIT"                                   # SPDX
maintainers = ["github-handle"]
models      = ["wood", "rosenrock"]
repo        = "madsuite-org/LuksanVlcekBenchmark.jl"  # where the releases are
nightly     = true                                    # publishes a `nightly` release
platforms   = ["linux-x86_64"]
cnlp_abi    = "0.2"
device      = ["cuda"]                                # optional: device builds it publishes (NAME-PLATFORM-cuda.tar.gz)
data        = ["pglib"]                               # optional: data sets it needs
```

The entry lists no versions, so later releases need no change here. CI checks the entry;
once merged, `cnlp install lv` works for everyone.
