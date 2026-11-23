#!/usr/bin/env python3
"""Validate libraries/*.toml and data/*.toml and write the index (index.json, index.tsv)
into site/.

index.tsv has one line per library and platform, and one per data set, for the newest
version: name, version, platform, url, sha256, description, kind, requires
(tab-separated). A library's rolling `[nightly]` build adds lines named `<name>@nightly`,
version `nightly` and sha256 `-`: the installer checks those against the `SHA256SUMS`
published next to the asset. Data sets have platform `any` and kind `data`; `requires` lists the data
sets a library needs (comma-separated). The installer reads it with plain shell tools.
"""
import json, os, re, sys, tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAME = re.compile(r"^[a-z0-9_]+$")
PLATFORM = re.compile(r"^(linux|macos|windows)-(x86_64|aarch64)$")
SHA = re.compile(r"^[0-9a-f]{64}$")
DATA_REQUIRED = ("name", "description", "homepage", "license", "maintainers", "versions")
REQUIRED = ("name", "description", "producer", "homepage", "license", "maintainers", "models", "versions")


def check(path):
    with open(path, "rb") as f:
        e = tomllib.load(f)
    errs = [f"missing `{k}`" for k in REQUIRED if k not in e]
    if errs:
        return None, errs
    stem = os.path.splitext(os.path.basename(path))[0]
    if e["name"] != stem:
        errs.append(f"`name` is {e['name']!r} but the file is {stem}.toml")
    if not NAME.match(e["name"]):
        errs.append("`name` must be lowercase [a-z0-9_]")
    if not e["versions"] and "nightly" not in e:
        errs.append("`versions` is empty and there is no `[nightly]`")
    if "nightly" in e:
        n = e["nightly"]
        if "cnlp_abi" not in n or not n.get("assets"):
            errs.append("`[nightly]` needs `cnlp_abi` and `assets`")
        for plat, a in n.get("assets", {}).items():
            if not PLATFORM.match(plat):
                errs.append(f"nightly: platform {plat!r} is not <os>-<arch>")
            if not str(a.get("url", "")).startswith("https://"):
                errs.append(f"nightly {plat}: url must be https")
    for v in e["versions"]:
        for k in ("version", "cnlp_abi", "assets"):
            if k not in v:
                errs.append(f"version {v.get('version', '?')}: missing `{k}`")
        for plat, a in v.get("assets", {}).items():
            if not PLATFORM.match(plat):
                errs.append(f"version {v.get('version')}: platform {plat!r} is not <os>-<arch>")
            if not str(a.get("url", "")).startswith("https://"):
                errs.append(f"version {v.get('version')} {plat}: url must be https")
            if not SHA.match(str(a.get("sha256", ""))):
                errs.append(f"version {v.get('version')} {plat}: sha256 must be 64 hex digits")
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
    if f1 or f2:
        sys.exit(1)
    os.makedirs(site, exist_ok=True)
    with open(os.path.join(site, "index.json"), "w") as f:
        json.dump({"format": "cnlplib-index", "format_version": "1", "libraries": entries, "data": data}, f, indent=1)
    with open(os.path.join(site, "index.tsv"), "w") as f:
        for e in entries:
            desc = e["description"].replace("\t", " ")
            req = ",".join(e.get("data", []))
            if e["versions"]:
                v = e["versions"][-1]
                for plat, a in sorted(v["assets"].items()):
                    f.write(f"{e['name']}\t{v['version']}\t{plat}\t{a['url']}\t{a['sha256']}\t{desc}\tlib\t{req}\n")
            for plat, a in sorted(e.get("nightly", {}).get("assets", {}).items()):
                f.write(f"{e['name']}@nightly\tnightly\t{plat}\t{a['url']}\t-\t{desc}\tlib\t{req}\n")
        for e in data:
            v = e["versions"][-1]
            desc = e["description"].replace("\t", " ")
            f.write(f"{e['name']}\t{v['version']}\tany\t{v['url']}\t{v['sha256']}\t{desc}\tdata\t\n")
    print(f"index: {len(entries)} libraries, {len(data)} data sets")


if __name__ == "__main__":
    main()
