# Libraries from the cnlplib registry, installed into the one store every cnlp tool shares.
#
# `lib("lv")` (so `CNLPModel("@lv", ...)` and `cnlp"lv"`) falls back to the registry when the
# library is neither on the search path nor installed: the registry's index
# (https://cnlplib.madsuite.org/index.tsv) names the release asset of each library for each
# platform, and the asset is downloaded, checked against the SHA-256 its release publishes,
# and installed into `$CNLPLIB_HOME` (default `~/.cnlplib`) exactly as the `cnlp` command
# installs it: `lib/NAME`, `data/NAME`, `solvers/NAME`, the shared pieces an asset names in its
# ARTIFACTS file once in `artifacts/` and linked in, and a `.cnlplib-version` record. A library
# is therefore stored once for the `cnlp` command, Julia and Python alike, and found offline
# afterwards.

using Downloads: Downloads
using SHA: sha256
using Tar: Tar
using p7zip_jll: p7zip_jll

const DEFAULT_INDEX = "https://cnlplib.madsuite.org/index.tsv"
_index_url() = get(ENV, "CNLPLIB_INDEX", DEFAULT_INDEX)
_home() = get(ENV, "CNLPLIB_HOME", joinpath(homedir(), ".cnlplib"))
const _DIRS = Dict("lib" => "lib", "data" => "data", "solver" => "solvers")

# this machine as the index names platforms: linux-x86_64, macos-aarch64, ...
function _platform()
    os = Sys.islinux() ? "linux" : Sys.isapple() ? "macos" : Sys.iswindows() ? "windows" :
         error("unsupported operating system for the cnlplib registry")
    arch = Sys.ARCH === :x86_64 ? "x86_64" : Sys.ARCH === :aarch64 ? "aarch64" :
           error("unsupported architecture $(Sys.ARCH) for the cnlplib registry")
    return "$os-$arch"
end

# the devices to prefer, best first: `cuda` when nvidia-smi lists a GPU (CNLPLIB_DEVICE=cpu|cuda overrides)
function _devices()
    d = get(ENV, "CNLPLIB_DEVICE", "")
    d == "cpu" && return String[]
    d == "cuda" && return ["cuda"]
    isempty(d) || error("CNLPLIB_DEVICE is cpu or cuda, not `$d`")
    smi = Sys.which("nvidia-smi")
    smi === nothing && return String[]
    out = try read(pipeline(`$smi -L`; stderr = devnull), String) catch; "" end
    return occursin(r"^GPU"m, out) ? ["cuda"] : String[]
end

# one row of index.tsv: name label platform url sha256 description kind requires versioned-url backends
struct IndexEntry
    name::String
    label::String
    platform::String
    url::String
    sha256::String     # "-": in the SHA256SUMS next to the asset
    kind::String
    requires::Vector{String}
    versioned::String  # the asset's URL with `{version}`, or "-"
end

function _fetch(url)
    io = IOBuffer()
    Downloads.download(url, io)
    return String(take!(io))
end

function _index()
    text = try
        _fetch(_index_url())
    catch e
        error("could not download the cnlplib index from $(_index_url()): $(sprint(showerror, e))")
    end
    rows = IndexEntry[]
    for line in eachline(IOBuffer(text))
        f = split(line, '\t')
        length(f) >= 7 || continue
        req = length(f) >= 8 ? filter(!isempty, split(f[8], ',')) : SubString{String}[]
        push!(rows, IndexEntry(f[1], f[2], f[3], f[4], f[5], f[7], String.(req), length(f) >= 9 ? f[9] : "-"))
    end
    return rows
end

# the entries for `key` (`lv`, `lv@nightly`) on this machine, best build first
function _candidates(rows, key)
    plat = _platform()
    order = [["$plat-$d" for d in _devices()]; plat; "any"]
    return [e for p in order for e in rows if e.name == key && e.platform == p]
end

# the SHA-256 of `file` in the SHA256SUMS published next to `url`, or nothing
function _published_sha(url, file)
    text = try _fetch(dirname(url) * "/SHA256SUMS") catch; return nothing end
    for line in eachline(IOBuffer(text))
        f = split(line)
        length(f) == 2 && lstrip(f[2], '*') == file && return String(f[1])
    end
    return nothing
end

_hex(file) = bytes2hex(open(sha256, file))

function _download_checked(url, file, want)
    try
        Downloads.download(url, file)
    catch e
        error("download failed: $url ($(sprint(showerror, e)))")
    end
    got = _hex(file)
    got == want || error("checksum mismatch for $url: expected $want, got $got")
    return file
end

function _unpack(tarball, dir)
    exe = p7zip_jll.p7zip()
    open(pipeline(`$exe x $tarball -so`; stderr = devnull)) do io
        Tar.extract(io, dir)
    end
    return dir
end

# an asset's content, or its single top-level directory
function _single_dir(x)
    fs = readdir(x)
    return length(fs) == 1 && isdir(joinpath(x, fs[1])) ? joinpath(x, fs[1]) : x
end

# what is installed under `name`: (version, sha256, request), or nothing
function _installed(name)
    for d in values(_DIRS)
        f = joinpath(_home(), d, name, ".cnlplib-version")
        isfile(f) || continue
        p = split(readline(f))
        return (get(p, 1, ""), get(p, 2, ""), get(p, 3, name))
    end
    return nothing
end

# move `src` to `dest`; another process may have put it there first
function _place(src, dest)
    try
        mv(src, dest)
    catch
        isdir(dest) || rethrow()
    end
    return dest
end

# A shared piece an asset names in its ARTIFACTS file, installed once into
# `$CNLPLIB_HOME/artifacts`: `julia-X.Y.Z`, the lib/julia of that official Julia release (copied
# from a juliaup install of exactly that version when there is one); `<name>-<hex>`, the
# `<name>-<hex>.artifact.tar.gz` of the asset's own release.
function _shared(a, asset_url)
    art = mkpath(joinpath(_home(), "artifacts"))
    if (m = match(r"^julia-(\d+\.\d+\.\d+)$", a)) !== nothing
        v = m[1]
        dest = joinpath(art, "$a-$(_platform())")
        isdir(dest) && return dest
        jdir = joinpath(get(ENV, "JULIAUP_DEPOT_PATH", joinpath(homedir(), ".julia")), "juliaup")
        if isdir(jdir)
            for d in readdir(jdir; join = true)
                startswith(basename(d), "julia-$v+") || continue
                src = joinpath(d, "lib", "julia")
                if isfile(joinpath(d, "lib", "libjulia.so.$v")) && isdir(src)
                    # copied, not linked: removing that Julia with juliaup must not break the libraries
                    stage = mktempdir(art; prefix = ".stage-$a.")
                    try
                        cp(src, joinpath(stage, "julia"))
                        _place(joinpath(stage, "julia"), dest)
                    finally
                        rm(stage; recursive = true, force = true)
                    end
                    return dest
                end
            end
        end
        plat = _platform()
        sub = plat == "linux-x86_64" ? "linux/x64" : plat == "linux-aarch64" ? "linux/aarch64" :
              error("no official Julia download known for $plat")
        file = "julia-$v-$plat.tar.gz"
        mm = join(split(v, '.')[1:2], '.')
        url = "https://julialang-s3.julialang.org/bin/$sub/$mm/$file"
        want = nothing
        for line in eachline(IOBuffer(_fetch("https://julialang-s3.julialang.org/bin/checksums/julia-$v.sha256")))
            f = split(line)
            length(f) == 2 && f[2] == file && (want = String(f[1]))
        end
        want === nothing && error("no published checksum for $file")
        stage = mktempdir(art; prefix = ".stage-$a.")
        try
            t = _download_checked(url, joinpath(stage, "j.tar.gz"), want)
            x = _unpack(t, mkpath(joinpath(stage, "x")))
            _place(joinpath(x, "julia-$v", "lib", "julia"), dest)
        finally
            rm(stage; recursive = true, force = true)
        end
        return dest
    elseif occursin(r"^[a-z0-9_]+-[0-9a-f]{16,64}$", a)
        dest = joinpath(art, a)
        isdir(dest) && return dest
        file = a * ".artifact.tar.gz"
        url = dirname(asset_url) * "/" * file
        want = _published_sha(url, file)
        want === nothing && error("no checksum for $file in $(dirname(url))/SHA256SUMS")
        stage = mktempdir(art; prefix = ".stage-$a.")
        try
            t = _download_checked(url, joinpath(stage, "a.tar.gz"), want)
            x = _unpack(t, mkpath(joinpath(stage, "x")))
            isdir(joinpath(x, a)) || error("$file has no top-level $a/")
            _place(joinpath(x, a), dest)
        finally
            rm(stage; recursive = true, force = true)
        end
        return dest
    end
    error("an asset names an invalid shared artifact `$a`")
end

"""
    install(name) -> String

Install registry library, solver or data set `name` (`"lv"`, `"emp@nightly"`, `"lv@1.2.0"`)
into `\$CNLPLIB_HOME` (default `~/.cnlplib`), as `cnlp install name` does, with the data sets
it requires, and return its directory. The same bytes already installed are kept. On a
machine with an NVIDIA GPU a library's CUDA build is taken when its release carries one
(`CNLPLIB_DEVICE=cpu` takes the host build). [`lib`](@ref) calls this for a name it finds
nowhere else, so `CNLPModel("@lv", :wood, 1000)` works on first use.
"""
install(req::AbstractString) = _install(_index(), String(req))

function _install(rows, req)
    n, sel = occursin('@', req) ? String.(split(req, '@'; limit = 2)) : (req, "")
    key = sel == "nightly" ? "$n@nightly" : n
    cands = _candidates(rows, key)
    isempty(cands) && error("no `$req` for $(_platform()) in the cnlplib index ($(_index_url()))")
    for r in cands[1].requires
        _install(rows, r)
    end
    entry, url, want, ver = nothing, "", nothing, ""
    for e in cands
        u, v = e.url, e.label
        if !(sel in ("", "latest", "nightly"))      # a pinned version
            e.versioned in ("", "-") && continue
            u, v = replace(e.versioned, "{version}" => sel), sel
        end
        w = e.sha256 == "-" || u != e.url ? _published_sha(u, basename(u)) : e.sha256
        if w !== nothing
            entry, url, want, ver = e, u, w, v
            break
        end
    end
    if entry === nothing
        hint = isempty(sel) && !isempty(_candidates(rows, "$n@nightly")) ? "; its nightly build is `$n@nightly`" : ""
        error("no released build of `$req` for $(_platform())$hint")
    end
    dest = joinpath(_home(), _DIRS[entry.kind])
    cur = _installed(n)
    if cur !== nothing && cur[2] == want
        # the same bytes: record the request, so `cnlp up` follows the channel asked for last
        write(joinpath(dest, n, ".cnlplib-version"), "$(cur[1]) $want $req\n")
        return joinpath(dest, n)
    end
    mkpath(dest)
    stage = mktempdir(dest; prefix = ".stage-$n.")
    try
        t = _download_checked(url, joinpath(stage, "asset.tar.gz"), want)
        x = mkpath(joinpath(stage, "x"))
        _unpack(t, x)
        src = _single_dir(x)
        ext = Libdl.dlext
        entry.kind == "lib" && !isfile(joinpath(src, "lib", "lib$n.$ext")) &&
            error("the asset of `$req` has no lib/lib$n.$ext")
        entry.kind == "solver" && !isfile(joinpath(src, "lib", "libcnlp_$n.$ext")) &&
            error("the asset of `$req` has no lib/libcnlp_$n.$ext")
        # shared artifacts, each installed once and its files linked into the library's TARGET
        # (a file the library carries wins)
        af = joinpath(src, "ARTIFACTS")
        if isfile(af)
            for line in eachline(af)
                f = split(line)
                isempty(f) && continue
                length(f) == 2 || error("the asset of `$req` has a malformed ARTIFACTS line: $line")
                tgt = String(f[2])
                (startswith(tgt, "/") || ".." in splitpath(tgt)) && error("the asset of `$req` links into `$tgt`, outside the library")
                ad = _shared(String(f[1]), url)
                d = mkpath(joinpath(src, tgt))
                for g in readdir(ad)
                    p = joinpath(d, g)
                    ispath(p) || islink(p) || symlink(joinpath(ad, g), p)
                end
            end
        end
        # the version the asset says it is, if it says
        vf = joinpath(src, "VERSION")
        if isfile(vf)
            for line in eachline(vf)
                m = match(r"^version *= *(.+)$", line)
                m === nothing || (ver = strip(m[1]); break)
            end
        end
        write(joinpath(src, ".cnlplib-version"), "$ver $want $req\n")
        # swap into place: a failed install never leaves a half-installed library
        old = joinpath(dest, ".old-$n")
        rm(old; recursive = true, force = true)
        isdir(joinpath(dest, n)) && mv(joinpath(dest, n), old)
        mv(src, joinpath(dest, n))
        rm(old; recursive = true, force = true)
    finally
        rm(stage; recursive = true, force = true)
    end
    return joinpath(dest, n)
end

"""
    update!(name) -> String

Install registry library `name` again if its current release differs from the installed
one; see [`install`](@ref).
"""
update!(name::AbstractString) = (delete!(_LIBS, String(name)); install(name))

# a registry library's file: installed (for `name`, under any channel; for `name@channel`,
# installed with that request), else installed now
function _registry_lib(name)
    n = String(first(split(name, '@')))
    f = joinpath(_home(), "lib", n, "lib", "lib$n.$(Libdl.dlext)")
    cur = _installed(n)
    ok = isfile(f) && (n == name || (cur !== nothing && cur[3] == name))
    ok || install(name)
    isfile(f) || error("the registry asset of `$name` has no lib/lib$n.$(Libdl.dlext)")
    return f
end

"""
    data_file(name) -> String

A file of an installed data set by name, as `cnlp solve` resolves one: a path as it is, else
the file of that exact name, or that name without its extension, under `\$CNLPLIB_HOME/data`.
A name that matches no file is an error that lists the files whose names contain it.
"""
function data_file(name::AbstractString)
    f, near = _find_data(name)
    f === nothing || return f
    error("no file `$name`: not a path, and no installed data-set file of that name" *
          (isempty(near) ? "" : "; did you mean: " * join(near, ", ")))
end

# the data-set file `name`, or nothing with the names that contain it
function _find_data(name::AbstractString)
    (isfile(name) || occursin('/', name)) && return String(name), String[]
    near = String[]
    d = joinpath(_home(), "data")
    isdir(d) || return nothing, near
    for (root, _, files) in walkdir(d), f in files
        (f == name || splitext(f)[1] == name) && return joinpath(root, f), near
        occursin(name, f) && push!(near, splitext(f)[1])
    end
    return nothing, first(sort!(unique!(near)), 10)
end

# a model's string argument: a data-set file when one has that name, else the string as given
# (the model's own function decides what it means)
_string_arg(s::AbstractString) = something(first(_find_data(s)), String(s))
