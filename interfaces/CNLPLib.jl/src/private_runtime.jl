# ── Loading UNBUNDLED juliac libraries in-process ────────────────────────────
#
# A `juliac --output-lib` library compiled without bundling links the standard
# `libjulia.so.x.y` and initializes that runtime lazily on its first call. Loaded into a
# process that is already Julia, the shared `libjulia` is already initialized, so the
# library's preamble falls through to `jl_adopt_thread`, whose thread-local guard aborts
# the process. The library needs a second runtime, distinct from the host's.
#
# JuliaC's bundler ships a privatized runtime copy at compile time (~80 MB per library).
# This file replays that transformation at load time, in scratch, from the consumer's own
# Julia installation:
#
#   1. copy `libjulia` and `julia/libjulia-internal` under salted names;
#   2. rewrite libjulia's `dep_libs` blob (how it finds libjulia-internal) to them;
#   3. symlink the rest of `lib/julia` beside them (support libraries keep their
#      standard sonames and dedupe with the host's copies, as a bundle's do);
#   4. Linux: re-stamp the ELF symbol version `JL_LIBJULIA_x.y` to `JL_<salt>_x.y` in both
#      copies, so nothing binds across to the host's libjulia;
#   5. patch a scratch copy of the model library onto the salted names (NEEDED entries or
#      load commands, rpath, version stamp) and `dlopen` that copy.
#
# The salt is fresh per load: two identically named runtimes cannot coexist in one
# process. Loads are cached per file, so one library costs one runtime.
#
# On macOS the edits go through `install_name_tool`, and no string may grow (a juliac
# dylib has no Mach-O header padding), so the salt replaces the `libjulia` stem with one of
# the same length; the edited binaries are re-signed ad hoc, which arm64 requires. The
# two-level namespace binds each symbol to the install name it was found under, so no
# symbol-version stamp is needed. Windows has no privatization: the unbundled form is
# refused there.

import .PatchVersion
import ObjectFile
import Patchelf_jll
import Random
import Mmap

# Model-library handles by realpath: one private runtime per file.
const _PRIVATE_HANDLES = Dict{String, Ptr{Cvoid}}()

_patchelf(args...) =
    run(`$(Patchelf_jll.patchelf()) $(collect(String, args))`)

# The version of the standard libjulia a library links, or `nothing` (a bundled library
# links a salted name; a plain C library none; an unparsable file is left to dlopen).
function _stdlinked_libjulia(path::AbstractString)
    occursin(r"\.(so|dylib)", basename(path)) || return nothing
    return try
        open(path, "r") do io
            oh = only(ObjectFile.readmeta(io))
            for dl in ObjectFile.DynamicLinks(oh)
                m = match(r"^libjulia\.(?:so\.(\d+\.\d+)|(\d+\.\d+)\.dylib)$",
                          basename(ObjectFile.path(dl)))
                m === nothing || return something(m.captures...)
            end
            return nothing
        end
    catch
        nothing
    end
end

# dlopen for model libraries: unbundled juliac libraries take the private-
# runtime route; everything else (bundles, plain C libraries) loads as-is.
function _dlopen_model(path::AbstractString, flags)
    ver = _stdlinked_libjulia(path)
    ver === nothing && return Libdl.dlopen(path, flags)
    key = realpath(path)
    return get!(_PRIVATE_HANDLES, key) do
        _dlopen_private(key, ver, flags)
    end
end

# The platform-form basenames of the runtime pair a library of version `ver`
# links: what the copies are named (salted), and what is matched when
# rewriting references.
_rt_basenames(ver) =
    Sys.isapple() ? ("libjulia." * ver * ".dylib", "libjulia-internal." * ver * ".dylib") :
    ("libjulia.so." * ver, "libjulia-internal.so." * ver)

function _dlopen_private(path::AbstractString, ver::AbstractString, flags)
    hostver = "$(VERSION.major).$(VERSION.minor)"
    ver == hostver || error(
        "$path is an unbundled juliac library linked against Julia $ver, but " *
        "this process runs Julia $hostver. Loading it here needs a runtime " *
        "sourced from a matching installation — run under Julia $ver, or " *
        "recompile the library (compile_library(...; bundle = true) makes it " *
        "self-contained).")
    (Sys.islinux() || Sys.isapple()) || error(
        "$path is an unbundled juliac library, which this Julia process can " *
        "only load on Linux or macOS (where CNLPLib can give it a private " *
        "copy of the installed runtime; loaded as-is it would abort the " *
        "process). Compile with bundle = true for this platform, or consume " *
        "the library from Python/C, where it works as-is.")

    libdir = dirname(Libdl.dlpath("libjulia"))     # <prefix>/lib
    # The scratch runtime is removed at process exit; CNLPLIB_KEEP_RUNTIME=1 keeps it for
    # inspection.
    keep = get(ENV, "CNLPLIB_KEEP_RUNTIME", "") == "1"
    scratch = mktempdir(prefix = "cnlplib_rt_", cleanup = !keep)
    lib = joinpath(scratch, "lib");     mkpath(lib)
    priv = joinpath(lib, "julia");      mkpath(priv)

    # Entropy from the OS: the task-local RNG has produced repeated salts across calls,
    # and equal salts mean one shared, aborting runtime.
    salt = String(rand(Random.RandomDevice(), ['a':'z'; 'A':'Z'; '0':'9'], 8))

    # The salted runtime pair. Linux prepends the salt (`<salt>_libjulia.so.x.y`; patchelf
    # grows ELF strings freely). On macOS every rewrite must keep its length, so the salt
    # replaces the 8-char stem: `libjulia` → `lib<salt5>`, `libjulia-internal.dylib` →
    # `libXXXXX-internal.dylib`.
    base_lj, base_int = _rt_basenames(ver)
    src_lj  = joinpath(libdir, base_lj)
    src_int = joinpath(libdir, "julia", base_int)
    (isfile(src_lj) && isfile(src_int)) || error(
        "this Julia installation has no $base_lj/$base_int pair at $libdir " *
        "to source a private runtime from")
    if Sys.islinux()
        s_lj  = joinpath(lib,  salt * "_" * base_lj)
        s_int = joinpath(priv, salt * "_" * base_int)
    else
        stem = "lib" * lowercase(salt[1:5])
        stem == "libjulia" && (stem = "libjulja")   # never collide with the host stem
        s_lj  = joinpath(lib,  _restem(base_lj, stem))
        s_int = joinpath(priv, _restem(base_int, stem))
    end
    for (src, dst) in ((src_lj, s_lj), (src_int, s_int))
        cp(realpath(src), dst)
        chmod(dst, 0o755)
    end

    if Sys.islinux()
        rpath = lib * ":" * priv
        oldver = "JL_LIBJULIA_" * hostver
        newver = "JL_" * salt * "_" * hostver
        for dst in (s_lj, s_int)
            _patchelf("--set-soname", basename(dst), dst)
            _patchelf("--set-rpath", rpath, dst)
        end
        _replace_dep_libs(s_lj, "libjulia" => salt * "_libjulia")
        for dst in (s_lj, s_int)
            _rewrite_needed(dst, ver, s_lj, s_int)
            PatchVersion.patch_version!(dst, oldver, newver)
        end
    else
        # macOS: identity is the install name. No rpath is added and no string grows. The
        # salted pair needs no rpath edits: its `@loader_path`-relative rpaths resolve
        # inside the scratch mirror of the real layout. Every edit precedes the re-sign.
        for dst in (s_lj, s_int)
            _install_name_tool("-id", "@rpath/" * basename(dst), dst)
        end
        _replace_dep_libs(s_lj, "libjulia" => stem)
        for dst in (s_lj, s_int)
            _rewrite_loads_macos(dst, stem)
            _codesign_adhoc(dst)
        end
        # References use versioned and unversioned forms; cover the aliases with symlinks,
        # as the real installation's chain does.
        for (alias, dst) in (
            (stem * ".dylib", s_lj),
            (_restem(basename(realpath(src_lj)), stem), s_lj),
            (stem * "-internal.dylib", s_int),
            (_restem(basename(realpath(src_int)), stem), s_int),
        )
            p = joinpath(dirname(dst), alias)
            alias == basename(dst) || ispath(p) || symlink(basename(dst), p)
        end
    end

    # The support libraries, found relative to the salted runtime exactly as
    # they are relative to the real one.
    for f in readdir(joinpath(libdir, "julia"))
        startswith(f, "libjulia") && continue
        symlink(joinpath(libdir, "julia", f), joinpath(priv, f))
    end

    # The model itself, onto the salted runtime.
    model = joinpath(lib, basename(path))
    cp(path, model)
    chmod(model, 0o755)
    if Sys.islinux()
        rpath = lib * ":" * priv
        _rewrite_needed(model, ver, s_lj, s_int)
        _patchelf("--set-rpath", rpath, model)
        PatchVersion.patch_version!(model, "JL_LIBJULIA_" * hostver,
                                    "JL_" * salt * "_" * hostver)
    else
        _rewrite_loads_macos(model, stem)
        _retarget_rpaths_macos(model, lib, priv)
        _codesign_adhoc(model)
    end

    return Libdl.dlopen(model, flags)
end

# ── macOS binary edits ───────────────────────────────────────────────────────

# The tool's stderr is the diagnostic (it names padding problems), so a failure carries it.
function _install_name_tool(args...)
    tool = Sys.which("install_name_tool")
    tool === nothing && error(
        "install_name_tool not found — the Xcode command-line tools are " *
        "needed to load an unbundled juliac library on macOS")
    cmd = `$tool $(collect(String, args))`
    buf = IOBuffer()
    success(pipeline(cmd; stdout = buf, stderr = buf)) ||
        error("install_name_tool failed: ", strip(String(take!(buf))),
              "\n(command: ", cmd, ")")
    return nothing
end

# `libjulia<rest>` → `<stem><rest>`, length-identical by construction.
function _restem(name::AbstractString, stem::AbstractString)
    startswith(name, "libjulia") ||
        error("$name does not carry the libjulia stem to substitute")
    return stem * name[9:end]
end

# Rewrite the load commands that reference the runtime onto the salted stem. The old
# string must be exactly as recorded, so the references are read out with ObjectFile. An
# `@`-relative reference keeps its prefix (same length); an absolute one becomes
# `@rpath/` + name (shorter).
function _rewrite_loads_macos(bin::AbstractString, stem::AbstractString)
    loads = open(bin, "r") do io
        oh = only(ObjectFile.readmeta(io))
        [ObjectFile.path(dl) for dl in ObjectFile.DynamicLinks(oh)]
    end
    for p in loads
        b = basename(p)
        occursin(r"^libjulia(-internal)?(\.\d+)*\.dylib$", b) || continue
        prefix = startswith(p, "@") ? p[1:(end - length(b))] : "@rpath/"
        _install_name_tool("-change", p, prefix * _restem(b, stem), bin)
    end
    return nothing
end

# The LC_RPATH entries, as recorded (`otool` ships beside `install_name_tool`).
function _rpaths_macos(bin::AbstractString)
    tool = Sys.which("otool")
    tool === nothing && return String[]
    out = read(`$tool -l $bin`, String)
    paths = String[]
    lines = split(out, '\n')
    for (i, l) in enumerate(lines)
        occursin(r"\bcmd LC_RPATH\b", l) || continue
        for j in (i + 1):min(i + 3, length(lines))
            m = match(r"^\s*path (.*?) \(offset \d+\)\s*$", lines[j])
            m === nothing || (push!(paths, String(m.captures[1])); break)
        end
    end
    return paths
end

# An unbundled juliac library carries absolute rpaths into the Julia installation it was
# linked against (`<prefix>/lib` and `<prefix>/lib/julia`). Retarget them in place onto the
# scratch layout: `-rpath old new` needs no header padding as long as the new path is no
# longer, and scratch paths are short.
function _retarget_rpaths_macos(bin::AbstractString, lib, priv)
    existing = _rpaths_macos(bin)
    isempty(existing) && error(
        "$bin carries no LC_RPATH to retarget — cannot point it at a " *
        "private runtime")
    hit = false
    for p in unique(existing)
        new = endswith(rstrip(p, '/'), "/julia") ? priv : lib
        p == new && continue
        _install_name_tool("-rpath", p, new, bin)
        hit = true
    end
    hit || error("no rpath of $bin could be retargeted onto $lib")
    return nothing
end

# install_name_tool invalidates the signature, and arm64 kills an invalidly signed binary
# at load. Ad hoc (`-s -`) is what JuliaC's bundler applies. A missing codesign is skipped
# (Intel tolerates it; on arm64 the dlopen failure then names the binary).
function _codesign_adhoc(bin::AbstractString)
    cs = Sys.which("codesign")
    cs === nothing && return nothing
    cmd = `$cs -f -s - $bin`
    buf = IOBuffer()
    success(pipeline(cmd; stdout = buf, stderr = buf)) ||
        error("codesign failed: ", strip(String(take!(buf))),
              "\n(command: ", cmd, ")")
    return nothing
end

function _rewrite_needed(bin::AbstractString, ver, s_lj, s_int)
    needed = filter(!isempty, split(
        read(`$(Patchelf_jll.patchelf()) --print-needed $bin`, String), '\n'))
    for n in needed
        if n == "libjulia.so." * ver
            _patchelf("--replace-needed", n, basename(s_lj), bin)
        elseif n == "libjulia-internal.so." * ver
            _patchelf("--replace-needed", n, basename(s_int), bin)
        end
    end
end

# libjulia locates libjulia-internal through its embedded `dep_libs` path list, not through
# NEEDED. Same transformation and buffer cap as JuliaC's bundler (`replace_dep_libs`).
const _DEP_LIBS_LENGTH = 512

function _replace_dep_libs(file::AbstractString, subst::Pair{<:AbstractString, <:AbstractString})
    off = open(file, "r") do io
        obj = only(ObjectFile.readmeta(io))
        syms = collect(ObjectFile.Symbols(obj))
        i = findfirst(==(ObjectFile.mangle_symbol_name(obj, "dep_libs")),
                      ObjectFile.symbol_name.(syms))
        i === nothing && error(
            "no dep_libs symbol in $file — cannot privatize this libjulia")
        ObjectFile.symbol_offset(syms[i])
    end
    open(file, "r+") do io
        data = Mmap.mmap(io)
        blob = String(data[off:(off + _DEP_LIBS_LENGTH - 1)])
        patched = Vector{UInt8}(replace(blob, subst)[1:_DEP_LIBS_LENGTH])
        data[off:(off + _DEP_LIBS_LENGTH - 1)] .= patched
        Mmap.sync!(data)
    end
    return nothing
end
