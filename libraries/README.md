# Library entries

One file per library, `libraries/<name>.toml`. `<name>` is what users install
(`cnlplib install <name>`) and the name consumers resolve (`@<name>`), so the library file
inside each asset must be `lib/lib<name>.<ext>`.

An entry names the repository that releases the library; it lists no versions:

```toml
name        = "lv"                                    # = file name; lowercase, [a-z0-9_]
description = "Luksan-Vlcek test problems"
producer    = "LuksanVlcekBenchmark.jl"               # what built it; any cnlp producer
homepage    = "https://github.com/madsuite-org/LuksanVlcekBenchmark.jl"
license     = "MIT"                                   # SPDX; upstream notices travel in the asset
maintainers = ["github-handle"]
models      = ["wood", "rosenrock"]
repo        = "madsuite-org/LuksanVlcekBenchmark.jl"  # where the releases are
nightly     = true                                    # it also publishes a rolling `nightly` release
platforms   = ["linux-x86_64"]                        # <os>-<arch>
cnlp_abi    = "0.2"
device      = ["cuda"]                                # optional tiers: device evaluation
data        = ["pglib"]                               # optional: data sets it needs (data/*.toml)
```

Every release of `repo` (tag `v<version>`, and `nightly` if published) carries
`<name>-<platform>.tar.gz` and a `SHA256SUMS`. The installer takes

| `cnlplib install` | from |
|---|---|
| `lv` | `releases/latest/download/lv-<platform>.tar.gz` |
| `lv@nightly` | `releases/download/nightly/lv-<platform>.tar.gz` |
| `lv@1.2.3` | `releases/download/v1.2.3/lv-<platform>.tar.gz` |

and checks each against the `SHA256SUMS` of the same release, so releasing a version needs
no change here. An asset is a `.tar.gz` whose single top-level directory is the library
bundle: `lib/lib<name>.<ext>`, its runtime, `LICENSE`, `THIRD_PARTY_NOTICES` and `VERSION`.

Model packages build and publish these releases with the reusable workflow
[`.github/workflows/library.yml`](../.github/workflows/library.yml):

```yaml
on:
  push: { branches: [main], tags: ["v*"] }
  pull_request:
jobs:
  library:
    uses: madsuite-org/cnlplib/.github/workflows/library.yml@main
    with: { name: lv }
    permissions: { contents: write }
```

A library not released this way gives `url`, a template containing `{platform}`, instead of
`repo`; its `SHA256SUMS` must sit next to the asset.

CI rejects an entry whose fields are missing or malformed, whose name differs from its
file name, or whose platforms are not `<os>-<arch>` with os in linux, macos, windows.
