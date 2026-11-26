# CNLPLib.jl

[![julia](https://github.com/madsuite-org/cnlplib/actions/workflows/julia.yml/badge.svg)](https://github.com/madsuite-org/cnlplib/actions/workflows/julia.yml)

[cnlp libraries](https://cnlplib.madsuite.org) (compiled nonlinear programs) as NLPModels,
for any NLPModels solver, and the Julia side of the cnlp solver interface.

```julia
using Pkg; Pkg.add(url = "https://github.com/madsuite-org/cnlplib", subdir = "interfaces/CNLPLib.jl")

using CNLPLib, MadNLP
m = CNLPModel("@emp", :acp, "pglib_opf_case118_ieee")
madnlp(m)
```

Documentation: [cnlplib.madsuite.org/julia](https://cnlplib.madsuite.org/julia/).
