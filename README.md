# cnlplib

A registry and installer for compiled optimization model libraries: shared libraries
implementing the [cnlp ABI](https://github.com/madsuite-org/cnlp-abi), such as the
compiled ExaModelsPower, COPS and Luksan–Vlček models. Any library implementing the ABI
can be listed, whatever produced it.

```sh
curl -fsSL https://cnlplib.madsuite.org/install | sh     # installs the cnlplib command
cnlplib list                                            # available and installed libraries
cnlplib install emp cops                                # into ~/.cnlplib/lib
cnlplib up                                              # update everything installed
```

Installed libraries are found by name by every cnlp consumer, because the installer
puts `~/.cnlplib/lib` on `CNLPMODELS_PATH`:

```julia
using CNLPModels
m = CNLPModel("@emp", :acp; input = :matpower => "case1354.m")
```

```sh
madnlp emp acp case1354_pglib --format matpower --output sol.m
```

Julia users of a modelling package usually do not need cnlplib: the package's own
`Artifacts.toml` fetches its compiled library on first use. cnlplib is for the command
line, C, Python and cluster installs, and for libraries from other producers.

## How it works

- Each library is released by its own project, as one `.tar.gz` per platform on its
  GitHub Releases. cnlplib does not rehost binaries.
- This repository lists them: one TOML file per library in [libraries/](libraries/),
  with the download URL and SHA-256 of each platform's asset.
- CI validates every entry and publishes the index ([index.json](https://cnlplib.madsuite.org/index.json),
  and `index.tsv` for the installer) and the installer to `https://cnlplib.madsuite.org/`.

## Adding a library

Open a pull request adding `libraries/<name>.toml`; see [libraries/README.md](libraries/README.md)
for the fields. CI checks the entry; a maintainer reviews and merges.

## License

MIT for this repository. Each library carries its own license, recorded in its entry.
