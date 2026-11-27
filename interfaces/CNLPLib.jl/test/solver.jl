# The solver interface (cnlp_solver.h): a CNLPModel handed out as a cnlp_problem, read back
# as a ProblemModel, solved, and the result written in the cnlp sign convention.
# (The sign convention, failing callbacks and options are covered in callbacks.jl.)
using CNLPLib: CNLPProblem, CNLPResult, ProblemModel, problem, set_result!, set_failure!, CNLP_SOLVED, CNLP_ERROR

@testset "solver interface" begin
    lib = CNLPLib.load(LIBPATH)
    m = CNLPModel(lib, 4; prefix = "tq")
    p = problem(m)
    GC.@preserve m p begin
        pm = ProblemModel(Base.unsafe_convert(Ptr{CNLPProblem}, p))

        @testset "the problem model evaluates as the model" begin
            @test pm.meta.nvar == 4 && pm.meta.ncon == 1 && pm.meta.nnzj == 2 && pm.meta.nnzh == 4
            @test pm.meta.x0 == m.meta.x0 && pm.meta.lcon == m.meta.lcon && pm.meta.minimize
            x = [0.5, 0.25, 2.0, -1.0]
            @test obj(pm, x) == obj(m, x)
            @test grad(pm, x) == grad(m, x)
            @test cons(pm, x) == cons(m, x)
            @test jac_structure(pm) == jac_structure(m)
            @test jac_coord(pm, x) == jac_coord(m, x)
            @test hess_structure(pm) == hess_structure(m)
            @test hess_coord(pm, x, [3.0]; obj_weight = 0.7) == hess_coord(m, x, [3.0]; obj_weight = 0.7)
            v = [1.0, -2.0, 3.0, 0.5]
            @test jprod(pm, x, v) == jac(m, x) * v
            @test jtprod(pm, x, [2.0]) == jac(m, x)' * [2.0]
        end

        @testset "solve and report" begin
            res = madnlp(pm; print_level = MadNLP.ERROR)
            @test res.status == MadNLP.SOLVE_SUCCEEDED
            x, y, zl, zu = zeros(4), zeros(1), zeros(4), zeros(4)
            r = Ref(CNLPResult(pointer(x), pointer(y), pointer(zl), pointer(zu), 0.0, -1, -1, ntuple(_ -> 0x00, 128)))
            GC.@preserve x y zl zu r begin
                rp = Base.unsafe_convert(Ptr{CNLPResult}, r)
                set_result!(rp, pm, res.solution, res.multipliers, res.multipliers_L, res.multipliers_U,
                            res.objective, CNLP_SOLVED, res.iter, "MadNLP: SOLVE_SUCCEEDED")
                @test x ≈ [0.5, 0.5, 1, 1] atol = 1e-6
                # minimize: s = 1, grad f + J'y = 0 at x, so y = -(x1 - 1) * 2 = 1
                @test grad(m, x) + jac(m, x)' * y ≈ zeros(4) atol = 1e-6
                @test r[].status == CNLP_SOLVED
                @test r[].objective ≈ 0.5 atol = 1e-6
                @test String(filter(!iszero, collect(r[].solver_status))) == "MadNLP: SOLVE_SUCCEEDED"
                set_failure!(rp, "bad option")
                @test r[].status == CNLP_ERROR && isnan(r[].objective) && r[].iterations == -1
            end
        end
    end
end
