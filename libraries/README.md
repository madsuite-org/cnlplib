# Library entries

One file per library, `libraries/<name>.toml`. `<name>` is what users install
(`cnlplib install <name>`) and the name consumers resolve (`@<name>`), so the library file
inside each asset must be `lib/lib<name>.<ext>`.

```toml
name        = "emp"                                   # = file name; lowercase, [a-z0-9_]
description = "AC and DC optimal power flow"
producer    = "ExaModelsPower.jl"                     # what built it; any cnlp producer
homepage    = "https://github.com/madsuite-org/ExaModelsPower.jl"
license     = "MIT"                                    # SPDX; upstream notices travel in the asset
maintainers = ["github-handle"]
models      = ["acp", "acr", "dcp"]

[[versions]]                                           # newest last
version  = "0.4.9"
cnlp_abi = "0.2"                                       # the ABI version the library implements
julia    = "1.12.7"                                    # the bundled runtime, if any
device   = ["cuda"]                                    # optional tiers: device evaluation
[versions.assets]                                      # one per platform: <os>-<arch>
"linux-x86_64"  = { url = "https://github.com/…/emp-0.4.9-linux-x86_64.tar.gz",  sha256 = "…" }
"linux-aarch64" = { url = "https://github.com/…/emp-0.4.9-linux-aarch64.tar.gz", sha256 = "…" }
```

An asset is a `.tar.gz` whose content (or whose single top-level directory) is the
library bundle: `lib/lib<name>.<ext>`, its runtime, `LICENSE`, `THIRD_PARTY_NOTICES`.

CI rejects an entry whose fields are missing or malformed, whose name differs from its
file name, or whose platforms are not `<os>-<arch>` with os in linux, macos, windows.
