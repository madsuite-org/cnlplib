#!/usr/bin/env python3
"""Render the documentation into site/ (next to the index that build_index.py writes):
docs/index.md as the home page, the rest of docs/ and formats/, and a registry page listing every library,
solver and data set with its backends. Links between the rendered pages become .html links;
links to other files of the repository (cnlp.h, the examples) point to GitHub.

    python3 scripts/build_site.py          (needs the `markdown` package; run build_index.py first)
"""
import html, json, os, re
import markdown

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = os.path.join(ROOT, "site")
REPO = "https://github.com/madsuite-org/cnlplib/blob/main"
PAGES = [f"docs/{f}" for f in sorted(os.listdir(os.path.join(ROOT, "docs")))] \
      + [f"formats/{f}" for f in sorted(os.listdir(os.path.join(ROOT, "formats")))]
PAGES = [p for p in PAGES if p.endswith(".md")]
NAV = [("index.html", "Home"), ("registry.html", "Registry"), ("docs/cli.html", "The cnlp command"),
       ("docs/models.html", "Publishing models"), ("docs/solvers.html", "Adding solvers"),
       ("formats/README.html", "Formats"), (REPO.replace("/blob/main", ""), "GitHub")]
STYLE = """body{font:16px/1.55 system-ui,sans-serif;max-width:52rem;margin:0 auto;padding:1rem 1.25rem;color:#222}
nav{border-bottom:1px solid #ddd;padding-bottom:.6rem;margin-bottom:1rem}nav a{margin-right:1rem}
a{color:#1a5fb4;text-decoration:none}a:hover{text-decoration:underline}
pre{background:#f6f8fa;padding:.75rem;overflow:auto;border-radius:6px}code{font-size:.9em}
:not(pre)>code{background:#f6f8fa;padding:.1em .3em;border-radius:4px}
table{border-collapse:collapse;margin:1rem 0}td,th{border:1px solid #ddd;padding:.3rem .6rem;text-align:left;vertical-align:top}
th{background:#f6f8fa}h1,h2,h3{line-height:1.25}"""


def out_path(page):
    return "index.html" if page == "docs/index.md" else page[:-3] + ".html"


def fix_links(body, page):
    """Rewrite href targets: rendered .md pages -> .html, other repository files -> GitHub."""
    here = os.path.dirname(page)

    def sub(m):
        url = m.group(1)
        if re.match(r"^[a-z]+:|^#", url):
            return m.group(0)
        path, _, frag = url.partition("#")
        target = os.path.normpath(os.path.join(here, path)) if path else page
        if target.endswith("/") or os.path.isdir(os.path.join(ROOT, target)):
            if not os.path.isfile(os.path.join(ROOT, target, "README.md")):   # a directory: GitHub lists it
                return f'href="{REPO.replace("/blob/", "/tree/")}/{target.rstrip("/")}"'
            target = os.path.join(target, "README.md")
        if target in PAGES:
            rel = os.path.relpath(out_path(target), os.path.dirname(out_path(page)) or ".")
        else:
            rel = f"{REPO}/{target}"
        return f'href="{rel}{"#" + frag if frag else ""}"'
    return re.sub(r'href="([^"]*)"', sub, body)


def wrap(title, body, page):
    up = os.path.relpath(".", os.path.dirname(out_path(page)) or ".")
    nav = " ".join(f'<a href="{h if h.startswith("http") else os.path.join(up, h)}">{t}</a>' for h, t in NAV)
    return (f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{html.escape(title)} · cnlplib</title><style>{STYLE}</style><nav>{nav}</nav>{body}</html>\n")


def registry_page():
    with open(os.path.join(SITE, "index.json")) as f:
        idx = json.load(f)
    def rows(items, kind):
        out = []
        for e in items:
            be = ", ".join(e["backends"] if kind == "solver" else ["cpu", *e.get("device", [])])
            ch = "nightly" if e.get("rolling") else ("latest, nightly" if e.get("nightly") else "latest")
            out.append(f'<tr><td><code>{e["name"]}</code></td><td>{html.escape(e["description"])}</td><td>{be}</td>'
                       f'<td>{", ".join(e["platforms"])}</td><td>{ch}</td><td>{e["license"]}</td></tr>')
        return "".join(out) or '<tr><td colspan="6">(none yet)</td></tr>'
    head = "<tr><th>name</th><th>description</th><th>backends</th><th>platforms</th><th>releases</th><th>license</th></tr>"
    data = "".join(f'<tr><td><code>{e["name"]}</code></td><td>{html.escape(e["description"])}</td><td>{e["license"]}</td></tr>'
                   for e in idx.get("data", []))
    body = (f"<h1>Registry</h1><p>Install with <code>cnlp install NAME</code> "
            f"(<code>NAME@nightly</code> for the latest build). Raw: <a href=\"index.json\">index.json</a>.</p>"
            f"<h2>Model libraries</h2><table>{head}{rows(idx['libraries'], 'lib')}</table>"
            f"<h2>Solvers</h2><table>{head}{rows(idx.get('solvers', []), 'solver')}</table>"
            f"<h2>Data sets</h2><table><tr><th>name</th><th>description</th><th>license</th></tr>{data}</table>")
    return wrap("Registry", body, "registry.md")


def main():
    for page in PAGES:
        with open(os.path.join(ROOT, page)) as f:
            text = f.read()
        body = markdown.markdown(text, extensions=["tables", "fenced_code"])
        title = re.search(r"^# (.+)$", text, re.M)
        dest = os.path.join(SITE, out_path(page))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w") as f:
            f.write(wrap(title.group(1) if title else page, fix_links(body, page), page))
    with open(os.path.join(SITE, "registry.html"), "w") as f:
        f.write(registry_page())
    print(f"site: {len(PAGES)} pages and the registry")


if __name__ == "__main__":
    main()
