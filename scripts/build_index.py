#!/usr/bin/env python3
"""Validate libraries/*.toml and write the index (index.json, index.tsv) into site/.

index.tsv has one line per library and platform for the newest version:
name, version, platform, url, sha256, description (tab-separated). The installer reads
it with plain shell tools.
"""
import json, os, re, sys, tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAME = re.compile(r"^[a-z0-9_]+$")
PLATFORM = re.compile(r"^(linux|macos|windows)-(x86_64|aarch64)$")
SHA = re.compile(r"^[0-9a-f]{64}$")
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
    if not e["versions"]:
        errs.append("`versions` is empty")
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
    return e, errs


def main():
    lib_dir, site = os.path.join(ROOT, "libraries"), os.path.join(ROOT, "site")
    entries, failed = [], False
    for fn in sorted(os.listdir(lib_dir)):
        if not fn.endswith(".toml"):
            continue
        e, errs = check(os.path.join(lib_dir, fn))
        if errs:
            failed = True
            for m in errs:
                print(f"libraries/{fn}: {m}", file=sys.stderr)
        else:
            entries.append(e)
    if failed:
        sys.exit(1)
    os.makedirs(site, exist_ok=True)
    with open(os.path.join(site, "index.json"), "w") as f:
        json.dump({"format": "exalib-index", "format_version": "1", "libraries": entries}, f, indent=1)
    with open(os.path.join(site, "index.tsv"), "w") as f:
        for e in entries:
            v = e["versions"][-1]
            for plat, a in sorted(v["assets"].items()):
                desc = e["description"].replace("\t", " ")
                f.write(f"{e['name']}\t{v['version']}\t{plat}\t{a['url']}\t{a['sha256']}\t{desc}\n")
    print(f"index: {len(entries)} libraries")


if __name__ == "__main__":
    main()
