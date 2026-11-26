# Libraries from the cnlplib registry, through the `cnlp` command.
#
# `lib("lv")` (so `CNLPModel("@lv", ...)` and `cnlp"lv"`) falls back to the libraries installed
# in `$CNLPLIB_HOME` (default `~/.cnlplib`), and installs one that is not there with
# `cnlp install`: one store, and one installer, for the `cnlp` command, Julia and Python.

_home() = get(ENV, "CNLPLIB_HOME", joinpath(homedir(), ".cnlplib"))
const _KINDS = ("lib", "data", "solvers")

# the `cnlp` command: the installer's, else on PATH
function _cnlp()
    c = joinpath(_home(), "bin", "cnlp")
    isfile(c) && return c
    c = Sys.which("cnlp")
    c === nothing && error("the `cnlp` command is not installed; install it with\n" *
                           "    curl -fsSL https://cnlplib.madsuite.org/install | sh")
    return c
end

# what is installed under `name`: (version, sha256, request), or nothing
function _installed(name)
    for d in _KINDS
        f = joinpath(_home(), d, name, ".cnlplib-version")
        isfile(f) || continue
        p = split(readline(f))
        return (get(p, 1, ""), get(p, 2, ""), get(p, 3, name))
    end
    return nothing
end

"""
    install(name) -> String

Install registry library, solver or data set `name` (`"lv"`, `"emp@nightly"`, `"lv@1.2.0"`)
with `cnlp install name`, into `\$CNLPLIB_HOME` (default `~/.cnlplib`), and return its
directory. [`lib`](@ref) calls this for a name it finds nowhere else, so
`CNLPModel("@lv", :wood, 1000)` works on first use.
"""
function install(name::AbstractString)
    # cnlp says what went wrong on stderr; here, that it did
    success(`$(_cnlp()) install $name`) || error("`cnlp install $name` failed")
    n = first(split(name, '@'))
    for d in _KINDS
        p = joinpath(_home(), d, n)
        isdir(p) && return p
    end
    error("`cnlp install $name` installed nothing under $(_home())")
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
    throw(ArgumentError("no file `$name`: not a path, and no installed data-set file of that name" *
                        (isempty(near) ? "" : "; did you mean: " * join(near, ", "))))
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
