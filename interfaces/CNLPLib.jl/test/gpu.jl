# GPU check, run by hand on a machine with CUDA (CI has none):
#
#   julia --project=<env with CNLPLib, CUDA, NLPModels, HDF5> test/gpu.jl LIB MODEL ARG...
#
# LIB is a library compiled with the device tier (`compile_library(...; device = :cuda)`).
# Every evaluation of `CNLPModel{CuVector}` must match `CNLPModel{Vector}` on the same instance.
using Test, CNLPLib, CUDA, NLPModels, HDF5

lib = CNLPLib.load(ARGS[1])
model = Symbol(ARGS[2])
args = parse.(Int, ARGS[3:end])
h = CNLPModel(lib, model, args...)
d = CNLPModel(lib, model, args...; backend = CUDABackend())
n, m = h.meta.nvar, h.meta.ncon
close(a, b) = maximum(abs, Array(a) .- b; init = 0.0) <= 1e-12 * max(1.0, maximum(abs, b; init = 0.0))

@testset "CNLPModel{CuVector} vs CNLPModel{Vector}: $model" begin
    @test d isa CNLPModel{<:CuVector{Float64}} && h isa CNLPModel{Vector{Float64}}
    @test d.meta.x0 isa CuVector && Array(d.meta.x0) == h.meta.x0 && d.meta.minimize == h.meta.minimize
    x = 0.5 .+ 0.3 .* sin.(1:n); y = cos.(1:m); v = sin.(2.0 .* (1:n)); w = cos.(3.0 .* (1:m))
    X, Y, V, W = CuVector(x), CuVector(y), CuVector(v), CuVector(w)
    @test obj(d, X) ≈ obj(h, x) rtol = 1e-12
    @test close(grad(d, X), grad(h, x))
    @test close(cons(d, X), cons(h, x))
    @test close(jac_coord(d, X), jac_coord(h, x))
    @test close(hess_coord(d, X, Y; obj_weight = 0.7), hess_coord(h, x, y; obj_weight = 0.7))
    @test close(hess_coord(d, X), hess_coord(h, x))
    @test jac_structure(d) == jac_structure(h) && hess_structure(d) == hess_structure(h)
    @test close(jprod(d, X, V), jprod(h, x, v))
    @test close(jtprod(d, X, W), jtprod(h, x, w))
    @test close(hprod(d, X, Y, V), hprod(h, x, y, v))
    @test keys(get_vars(d)) == keys(get_vars(h))
    # a result on the device is written from the host, like any other
    r = (; solution = X, multipliers = Y, multipliers_L = CUDA.zeros(n), multipliers_U = CUDA.zeros(n),
         objective = obj(h, x), status = :first_order, iter = 0)
    p = write_solution(d, joinpath(mktempdir(), "sol.h5"), r)
    @test h5open(f -> read(f["solution/solution"]), p) == x
end

@testset "the solver interface on the device: ProblemModel{CuVector}" begin
    p = CNLPLib.problem(h)
    q = CNLPLib.device_problem(p, d)
    GC.@preserve p q begin
        pm = CNLPLib.ProblemModel(Base.unsafe_convert(Ptr{CNLPLib.CNLPProblem}, p),
                                  Base.unsafe_convert(Ptr{CNLPLib.CNLPDeviceProblem}, q))
        @test pm isa CNLPLib.ProblemModel{<:CuVector{Float64}}
        x = 0.5 .+ 0.3 .* sin.(1:n); y = cos.(1:m); v = sin.(2.0 .* (1:n)); w = cos.(3.0 .* (1:m))
        X, Y, V, W = CuVector(x), CuVector(y), CuVector(v), CuVector(w)
        @test Array(pm.meta.x0) == h.meta.x0 && pm.meta.minimize == h.meta.minimize
        @test obj(pm, X) ≈ obj(h, x) rtol = 1e-12
        @test close(grad(pm, X), grad(h, x))
        @test close(cons(pm, X), cons(h, x))
        @test close(jac_coord(pm, X), jac_coord(h, x))
        @test close(hess_coord(pm, X, Y; obj_weight = 0.7), hess_coord(h, x, y; obj_weight = 0.7))
        @test jac_structure(pm) == jac_structure(h) && hess_structure(pm) == hess_structure(h)
        @test close(jprod(pm, X, V), jprod(h, x, v))
        @test close(jtprod(pm, X, W), jtprod(h, x, w))
        @test close(hprod(pm, X, Y, V), hprod(h, x, y, v))
    end
end
