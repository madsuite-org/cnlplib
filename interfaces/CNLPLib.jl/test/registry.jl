# Libraries from the cnlplib registry, against a registry served from local files laid out as
# the releases are: an index, a release with its SHA256SUMS, a shared artifact the library
# names in its ARTIFACTS file, and a data set the library requires. Everything is installed
# into a temporary CNLPLIB_HOME by this repository's `cnlp` command (bin/cnlp).
using CNLPLib: install, update!, data_file
using SHA: sha256
_sha(f) = bytes2hex(open(sha256, f))

@testset "the registry, installed into CNLPLIB_HOME" begin
    root = mktempdir()
    home = mktempdir()
    try
        plat = (Sys.islinux() ? "linux" : "macos") * "-" * (Sys.ARCH === :x86_64 ? "x86_64" : "aarch64")
        rel = mkpath(joinpath(root, "releases"))
        stage = mktempdir()
        tgz(dir, name, out) = run(`tar -czf $out -C $dir $name`)
        # the library: lib/libregqp.so, with an ARTIFACTS line linking a shared piece into lib/extra
        mkpath(joinpath(stage, "regqp", "lib"))
        cp(LIBPATH, joinpath(stage, "regqp", "lib", "libregqp.so"))
        write(joinpath(stage, "regqp", "ARTIFACTS"), "shared_x-0123456789abcdef lib/extra\n")
        tgz(stage, "regqp", joinpath(rel, "regqp-$plat.tar.gz"))
        mkpath(joinpath(stage, "shared_x-0123456789abcdef"))
        write(joinpath(stage, "shared_x-0123456789abcdef", "note.txt"), "shared\n")
        tgz(stage, "shared_x-0123456789abcdef", joinpath(rel, "shared_x-0123456789abcdef.artifact.tar.gz"))
        # the data set
        mkpath(joinpath(stage, "regdata", "cases"))
        write(joinpath(stage, "regdata", "cases", "case_tiny.txt"), "4\n")
        tgz(stage, "regdata", joinpath(rel, "regdata.tar.gz"))
        open(joinpath(rel, "SHA256SUMS"), "w") do io
            for f in readdir(rel)
                f == "SHA256SUMS" || println(io, _sha(joinpath(rel, f)), "  ", f)
            end
            # a release whose bytes are not what its SHA256SUMS says
            println(io, _sha(joinpath(rel, "regqp-$plat.tar.gz")), "  badqp-$plat.tar.gz")
        end
        write(joinpath(rel, "badqp-$plat.tar.gz"), "not the release")
        u(f) = "file://" * joinpath(rel, f)
        write(joinpath(root, "index.tsv"), join([
            "regqp\tlatest\t$plat\t$(u("regqp-$plat.tar.gz"))\t-\ta test library\tlib\tregdata\t-\tcpu",
            "badqp\tlatest\t$plat\t$(u("badqp-$plat.tar.gz"))\t-\ta tampered library\tlib\t\t-\tcpu",
            "regdata\tlatest\tany\t$(u("regdata.tar.gz"))\t-\ta test data set\tdata\t\t-\t-",
        ], "\n") * "\n")
        withenv("CNLPLIB_INDEX" => "file://" * joinpath(root, "index.tsv"), "CNLPLIB_DEVICE" => "cpu",
                "CNLPLIB_HOME" => home, "CNLPLIB_PATH" => nothing,
                "PATH" => joinpath(ROOT, "bin") * ":" * ENV["PATH"]) do
            CNLPLib.set_path!()
            m = CNLPModel("@regqp", 4; prefix = "tq")      # downloaded on first use
            @test m.meta.nvar == 4
            dir = install("regqp")
            # where and how the `cnlp` command installs it
            @test dir == joinpath(home, "lib", "regqp")
            @test split(readline(joinpath(dir, ".cnlplib-version")))[3] == "regqp"
            @test read(joinpath(dir, "lib", "extra", "note.txt"), String) == "shared\n"
            @test readlink(joinpath(dir, "lib", "extra", "note.txt")) ==
                  joinpath(home, "artifacts", "shared_x-0123456789abcdef", "note.txt")
            @test isfile(joinpath(home, "data", "regdata", ".cnlplib-version"))
            # the data set came with it, and its files resolve by name
            @test endswith(data_file("case_tiny"), joinpath("cases", "case_tiny.txt"))
            @test data_file("case_tiny.txt") == data_file("case_tiny")
            err = try data_file("tiny"); nothing catch e; sprint(showerror, e) end
            @test err !== nothing && occursin("did you mean: case_tiny", err)
            # found offline afterwards: the index is gone, the installed library is not
            mv(joinpath(root, "index.tsv"), joinpath(root, "index.off"))
            empty!(CNLPLib._LIBS)
            @test CNLPModel("@regqp", 3; prefix = "tq").meta.nvar == 3
            # the release has not changed: update! keeps what is there
            mv(joinpath(root, "index.off"), joinpath(root, "index.tsv"))
            @test update!("regqp") == dir
            # bytes that do not match the published checksum are refused
            @test_throws ErrorException install("badqp")
            @test !isdir(joinpath(home, "lib", "badqp"))
            # an unknown name says so
            @test_throws ErrorException CNLPLib.lib("nonexistent")
        end
    finally
        CNLPLib.set_path!()
        empty!(CNLPLib._LIBS)
    end
end
