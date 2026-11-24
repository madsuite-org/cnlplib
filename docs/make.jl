# The documentation site, https://cnlplib.madsuite.org, built with Documenter:
#
#   python3 scripts/build_index.py                      # site/index.json, the registry
#   julia --project=docs docs/make.jl                   # the pages, into site/
#
# The pages are docs/src and the format specifications in formats/; links to other files of
# the repository (cnlp.h, the examples) point to GitHub, and the registry page is generated
# from site/index.json. The site is written into site/, next to the index and the installer
# the registry workflow publishes at the root of the domain.
using Documenter, JSON

const ROOT = dirname(@__DIR__)
const REPO = "https://github.com/madsuite-org/cnlplib"
const STAGE = mktempdir()

# docs/src and formats/ into one source tree, with links rewritten: to a page of the site
# (relative), or to the file on GitHub
const PAGES_FROM = Dict{String, String}()     # source path (from ROOT) => staged path
for (dir, sub) in (("docs/src", ""), ("formats", "formats"))
    for f in readdir(joinpath(ROOT, dir))
        endswith(f, ".md") && (PAGES_FROM[joinpath(dir, f)] = joinpath(sub, f))
    end
end

function rewrite(text, src)
    here = dirname(src)
    return replace(text, r"(\]\s*)\(([^)\s]+)\)" => function (m)
        mm = match(r"(\]\s*)\(([^)\s]+)\)", m)
        lead, url = mm.captures
        occursin(r"^[a-z]+:|^#", url) && return m
        path, frag = (p = split(url, '#'; limit = 2); (p[1], length(p) == 2 ? "#" * p[2] : ""))
        target = normpath(joinpath(here, path))
        if isdir(joinpath(ROOT, target)) && isfile(joinpath(ROOT, target, "README.md")) &&
                haskey(PAGES_FROM, joinpath(target, "README.md"))
            target = joinpath(target, "README.md")
        end
        if haskey(PAGES_FROM, target)
            rel = relpath(PAGES_FROM[target], dirname(PAGES_FROM[src]) == "" ? "." : dirname(PAGES_FROM[src]))
            return string(lead, "(", rel, frag, ")")
        end
        kind = isdir(joinpath(ROOT, target)) ? "tree" : "blob"
        return string(lead, "(", REPO, "/", kind, "/main/", rstrip(target, '/'), frag, ")")
    end)
end

for (src, dst) in PAGES_FROM
    mkpath(dirname(joinpath(STAGE, dst)))
    write(joinpath(STAGE, dst), rewrite(read(joinpath(ROOT, src), String), src))
end
cp(joinpath(ROOT, "docs", "src", "assets"), joinpath(STAGE, "assets"))

# the registry page, from the index build_index.py writes
idx = JSON.parsefile(joinpath(ROOT, "site", "index.json"))
function table(items, kind)
    rows = ["| name | description | backends | platforms | releases | license |", "|:--|:--|:--|:--|:--|:--|"]
    for e in items
        be = join(kind == :solver ? e["backends"] : ["cpu"; get(e, "device", [])], ", ")
        rel = get(e, "rolling", false) ? "nightly" : get(e, "nightly", false) ? "latest, nightly" : "latest"
        push!(rows, "| `$(e["name"])` | $(e["description"]) | $be | $(join(e["platforms"], ", ")) | $rel | $(e["license"]) |")
    end
    isempty(items) && push!(rows, "| (none yet) | | | | | |")
    return join(rows, "\n")
end
data = ["| name | description | license |", "|:--|:--|:--|"]
for e in get(idx, "data", [])
    push!(data, "| `$(e["name"])` | $(e["description"]) | $(e["license"]) |")
end
write(joinpath(STAGE, "registry.md"), """
# Registry

Install with `cnlp install NAME` (`NAME@nightly` for the latest build of the default
branch). Machine-readable: [index.json](https://cnlplib.madsuite.org/index.json),
[index.tsv](https://cnlplib.madsuite.org/index.tsv).

## Model libraries

$(table(idx["libraries"], :lib))

## Solvers

$(table(get(idx, "solvers", []), :solver))

## Data sets

$(join(data, "\n"))
""")

makedocs(
    sitename = "cnlplib",
    source = STAGE,
    build = joinpath(ROOT, "site"),
    clean = false,                    # site/ also holds the index and the installer
    format = Documenter.HTML(
        assets = ["assets/favicon.ico"],
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://cnlplib.madsuite.org",
        repolink = REPO,
        edit_link = nothing,
        inventory_version = "0.2",
    ),
    remotes = nothing,
    pages = [
        "Home" => "index.md",
        "Registry" => "registry.md",
        "The cnlp command" => "cli.md",
        "Publishing models" => "models.md",
        "Adding solvers" => "solvers.md",
        "Formats" => [
            "Overview" => "formats/README.md",
            "Input" => "formats/input.md",
            "Parameters" => "formats/parameters.md",
            "Solutions" => "formats/solution.md",
            "Number formats" => "formats/numbers.md",
        ],
    ],
)
