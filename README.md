# cnlplib

A registry and installer for compiled optimization model libraries: shared libraries
implementing the [cnlp ABI](https://github.com/madsuite-org/cnlp-abi), such as the
compiled ExaModelsPower, COPS and Luksan–Vlček models. Any library implementing the ABI
can be listed, whatever produced it.

```sh
curl -fsSL https://cnlplib.madsuite.org/install | sh     # installs the cnlplib command
cnlplib list                                            # available and installed libraries
cnlplib install lv                                      # the latest release, into ~/.cnlplib/lib
cnlplib install lv@nightly                              # or the latest build of the default branch
cnlplib up                                              # update everything installed
```

Installed libraries are found by name by every cnlp consumer, because the installer
puts `~/.cnlplib/lib` on `CNLPMODELS_PATH`:

```julia
using CNLPModels
m = CNLPModel("@lv", :wood, 100_000)
```

## How it works

- Each library is released by its own project, as `<name>-<platform>.tar.gz` and a
  `SHA256SUMS` on its GitHub Releases: every version, and a rolling `nightly`. cnlplib
  does not rehost binaries.
- This repository lists the libraries, one TOML file each in [libraries/](libraries/),
  naming the repository that releases it. It lists no versions, so a release needs no
  change here; the installer checks every download against the release's `SHA256SUMS`.
- Model packages build and publish their releases with one reusable workflow,
  [`.github/workflows/library.yml`](.github/workflows/library.yml).
- CI validates every entry and publishes the index ([index.json](https://cnlplib.madsuite.org/index.json),
  and `index.tsv` for the installer) and the installer to `https://cnlplib.madsuite.org/`.

## Adding a library

Open a pull request adding `libraries/<name>.toml`; see [libraries/README.md](libraries/README.md)
for the fields. CI checks the entry; a maintainer reviews and merges.

## License

MIT for this repository. Each library carries its own license, recorded in its entry.
