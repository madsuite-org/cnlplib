#!/usr/bin/env python3
"""Validate libraries/*.toml and data/*.toml and write the index (index.json, index.tsv)
into site/.

A library is released by its own repository: `repo` names it, and every release carries
`<name>-<platform>.tar.gz` and a `SHA256SUMS`. The index therefore lists no versions: the
installer takes the latest release, the rolling `nightly` release, or a tagged version,
and checks each against the SHA256SUMS of the same release. (`url`, a template with
`{platform}`, replaces `repo` for a library that is not released that way.)

index.tsv has one line per library, channel and platform, and one per data set: name,
label, platform, url, sha256 (`-`: in the SHA256SUMS next to the asset), description,
kind, requires, versioned url (`{version}` substituted; `-` if none), backends
(comma-separated: a solver's `backends`, or `cpu` plus a library's `device` tiers),
tab-separated. The
nightly channel's lines are named `<name>@nightly`. Data sets have platform `any`, kind
`data` and a sha256. Solvers (solvers/*.toml) are listed the same way with kind `solver`; a `rolling` entry has
no versioned releases, so its plain name installs the `nightly` release. The installer
reads it with plain shell tools.
"""
import json, os, re, sys, tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAME = re.compile(r"^[a-z0-9_]+$")
REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
PLATFORM = re.compile(r"^(linux|macos|windows)-(x86_64|aarch64)$")
SHA = re.compile(r"^[0-9a-f]{64}$")
DATA_REQUIRED = ("name", "description", "homepage", "license", "maintainers", "versions")
REQUIRED = ("name", "description", "producer", "homepage", "license", "maintainers", "models",
            "cnlp_abi", "platforms")
SOLVER_REQUIRED = ("name", "description", "homepage", "license", "maintainers", "solver_interface",
                   "platforms", "repo", "backends")
BACKENDS = {"cpu", "cuda", "rocm", "oneapi", "metal"}


def check(path, required=REQUIRED):
    with open(path, "rb") as f:
        e = tomllib.load(f)
    errs = [f"missing `{k}`" for k in required if k not in e]
    if errs:
        return None, errs
    stem = os.path.splitext(os.path.basename(path))[0]
    if e["name"] != stem:
        errs.append(f"`name` is {e['name']!r} but the file is {stem}.toml")
    if not NAME.match(e["name"]):
        errs.append("`name` must be lowercase [a-z0-9_]")
    if ("repo" in e) == ("url" in e):
        errs.append("give exactly one of `repo` (GitHub releases) and `url` (a template)")
    if "repo" in e and not REPO.match(e["repo"]):
        errs.append("`repo` must be owner/name")
    if "url" in e and not (e["url"].startswith("https://") and "{platform}" in e["url"]):
        errs.append("`url` must be an https template containing {platform}")
    if e.get("rolling") and not e.get("nightly"):
        errs.append("a `rolling` entry needs `nightly`")
    if e.get("nightly") and "repo" not in e:
        errs.append("`nightly` needs `repo`")
    for plat in e["platforms"]:
        if not PLATFORM.match(plat):
            errs.append(f"platform {plat!r} is not <os>-<arch>")
    for b in e.get("backends", []):
        if b not in BACKENDS:
            errs.append(f"backend {b!r} is not one of {sorted(BACKENDS)}")
    for d in e.get("data", []):
        if not os.path.isfile(os.path.join(ROOT, "data", f"{d}.toml")):
            errs.append(f"`data` names {d!r}, which has no data/{d}.toml")
    return e, errs


def check_data(path):
    with open(path, "rb") as f:
        e = tomllib.load(f)
    errs = [f"missing `{k}`" for k in DATA_REQUIRED if k not in e]
    if errs:
        return None, errs
    stem = os.path.splitext(os.path.basename(path))[0]
    if e["name"] != stem:
        errs.append(f"`name` is {e['name']!r} but the file is {stem}.toml")
    if not NAME.match(e["name"]):
        errs.append("`name` must be lowercase [a-z0-9_]")
    for v in e["versions"]:
        if not str(v.get("url", "")).startswith("https://"):
            errs.append(f"version {v.get('version')}: url must be https")
        if not SHA.match(str(v.get("sha256", ""))):
            errs.append(f"version {v.get('version')}: sha256 must be 64 hex digits")
    return e, errs


def load_dir(sub, checker):
    d = os.path.join(ROOT, sub)
    entries, failed = [], False
    for fn in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if not fn.endswith(".toml"):
            continue
        e, errs = checker(os.path.join(d, fn))
        for m in errs:
            print(f"{sub}/{fn}: {m}", file=sys.stderr)
        failed |= bool(errs)
        if not errs:
            entries.append(e)
    return entries, failed


def main():
    site = os.path.join(ROOT, "site")
    entries, f1 = load_dir("libraries", check)
    data, f2 = load_dir("data", check_data)
    solvers, f3 = load_dir("solvers", lambda p: check(p, SOLVER_REQUIRED))
    if f1 or f2 or f3:
        sys.exit(1)
    os.makedirs(site, exist_ok=True)
    with open(os.path.join(site, "index.json"), "w") as f:
        json.dump({"format": "cnlplib-index", "format_version": "2", "libraries": entries, "data": data, "solvers": solvers}, f, indent=1)
    with open(os.path.join(site, "index.tsv"), "w") as f:
        for kind, group in (("lib", entries), ("solver", solvers)):
            for e in group:
                desc = e["description"].replace("\t", " ")
                req = ",".join(e.get("data", []))
                n = e["name"]
                be = ",".join(e["backends"] if kind == "solver" else ["cpu", *e.get("device", [])])
                # a library is published per backend: `<platform>` holds the host (CPU) tier only,
                # `<platform>-<device>` the host tier and that device tier
                variants = [(plat, be if kind == "solver" else "cpu") for plat in sorted(e["platforms"])]
                if kind == "lib" and "url" not in e:
                    variants += [(f"{plat}-{d}", f"cpu,{d}") for plat in sorted(e["platforms"]) for d in e.get("device", [])]
                for plat, be in variants:
                    asset = f"{n}-{plat}.tar.gz"
                    if "url" in e:
                        f.write(f"{n}\t{e.get('label', 'fixed')}\t{plat}\t{e['url'].replace('{platform}', plat)}\t-\t{desc}\t{kind}\t{req}\t-\t{be}\n")
                        continue
                    base = f"https://github.com/{e['repo']}/releases"
                    if e.get("rolling"):
                        f.write(f"{n}\tnightly\t{plat}\t{base}/download/nightly/{asset}\t-\t{desc}\t{kind}\t{req}\t-\t{be}\n")
                        continue
                    f.write(f"{n}\tlatest\t{plat}\t{base}/latest/download/{asset}\t-\t{desc}\t{kind}\t{req}\t{base}/download/v{{version}}/{asset}\t{be}\n")
                    if e.get("nightly"):
                        f.write(f"{n}@nightly\tnightly\t{plat}\t{base}/download/nightly/{asset}\t-\t{desc}\t{kind}\t{req}\t-\t{be}\n")
        for e in data:
            v = e["versions"][-1]
            desc = e["description"].replace("\t", " ")
            f.write(f"{e['name']}\t{v['version']}\tany\t{v['url']}\t{v['sha256']}\t{desc}\tdata\t\t-\t-\n")
    print(f"index: {len(entries)} libraries, {len(solvers)} solvers, {len(data)} data sets")


if __name__ == "__main__":
    main()
