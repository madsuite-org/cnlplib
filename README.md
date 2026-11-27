# cnlplib

[![check](https://github.com/madsuite-org/cnlplib/actions/workflows/check.yml/badge.svg)](https://github.com/madsuite-org/cnlplib/actions/workflows/check.yml)
[![julia](https://github.com/madsuite-org/cnlplib/actions/workflows/julia.yml/badge.svg)](https://github.com/madsuite-org/cnlplib/actions/workflows/julia.yml)
[![python](https://github.com/madsuite-org/cnlplib/actions/workflows/python.yml/badge.svg)](https://github.com/madsuite-org/cnlplib/actions/workflows/python.yml)
[![registry](https://github.com/madsuite-org/cnlplib/actions/workflows/pages.yml/badge.svg)](https://github.com/madsuite-org/cnlplib/actions/workflows/pages.yml)
[![docs](https://img.shields.io/badge/docs-cnlplib.madsuite.org-blue)](https://cnlplib.madsuite.org)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Compiled nonlinear programs: a C ABI for them ([cnlp.h](cnlp.h)), a registry of libraries
that implement it, and the `cnlp` command that installs and solves them.

```sh
curl -fsSL https://cnlplib.madsuite.org/install | sh    # installs the cnlp command
cnlp list                                              # available and installed libraries
cnlp install lv                                        # a model library
cnlp solve @lv:wood 10000 -o sol.h5                    # solve one of its models
```

Documentation: **[cnlplib.madsuite.org](https://cnlplib.madsuite.org)**: the ABI, the
[registry](https://cnlplib.madsuite.org/registry/), the
[`cnlp` command](https://cnlplib.madsuite.org/cli/), and how to
[publish a model library](https://cnlplib.madsuite.org/models/) or
[add a solver](https://cnlplib.madsuite.org/solvers/).

Interfaces, in [interfaces/](interfaces): [CNLPLib.jl](interfaces/CNLPLib.jl) (Julia) and
[cnlplib](interfaces/cnlplib-py) (Python).

MIT license. Each library in the registry carries its own license, recorded in its entry.
