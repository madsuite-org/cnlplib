# The solver interface (cnlp_solver.h): a CNLPModel handed out as a cnlp_problem, read back
# as a ProblemModel, solved, and the result written in the cnlp sign convention.
using CNLPLib: CNLPProblem, CNLPResult, ProblemModel, problem, set_result!, set_failure!,
    solver_options, CallbackError, CNLP_SOLVED, CNLP_ERROR

failing_obj(c::Ptr{Cvoid}, x::Ptr{Float64}, f::Ptr{Float64}) = Int64(7)

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

    @testset "maximization: multipliers in the cnlp convention" begin
        # max -(sum (x - 1)^2): the model negated, minimize = 0; same x and, in the cnlp
        # convention s grad f + J'y - zL + zU = 0 with s = -1, the same y as the minimization
        q = p[]
        pmax = ProblemModel(CNLPProblem(q.ctx, q.nvar, q.ncon, q.nnzj, q.nnzh, 0, q.x0, q.lvar, q.uvar,
                                        q.lcon, q.ucon, q.y0, q.zl0, q.zu0, q.obj, q.grad, q.cons,
                                        q.jac_structure, q.jac, q.hess_structure, q.hess))
        @test !pmax.meta.minimize
        # (evaluations are the minimization's; set_result! only applies the sign)
        x, y, zl, zu = zeros(4), zeros(1), zeros(4), zeros(4)
        r = Ref(CNLPResult(pointer(x), pointer(y), pointer(zl), pointer(zu), 0.0, -1, -1, ntuple(_ -> 0x00, 128)))
        GC.@preserve m p x y zl zu r begin
            set_result!(Base.unsafe_convert(Ptr{CNLPResult}, r), pmax, ones(4), [2.0], fill(3.0, 4), fill(4.0, 4),
                        -1.0, CNLP_SOLVED, 7, "ok")
        end
        @test y == [-2.0] && zl == fill(-3.0, 4) && zu == fill(-4.0, 4) && r[].iterations == 7
    end

    @testset "a failing callback throws CallbackError" begin
        q = p[]
        bad = ProblemModel(CNLPProblem(q.ctx, q.nvar, q.ncon, q.nnzj, q.nnzh, 1, q.x0, q.lvar, q.uvar,
                                       q.lcon, q.ucon, C_NULL, C_NULL, C_NULL,
                                       @cfunction(failing_obj, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64})),
                                       q.grad, q.cons, q.jac_structure, q.jac, q.hess_structure, q.hess))
        GC.@preserve m p @test_throws CallbackError obj(bad, zeros(4))
    end

    @testset "options" begin
        ks = ["tol", "max_iter"]; vs = ["1e-8", "50"]
        kp = [pointer(k) for k in ks]; vp = [pointer(v) for v in vs]
        GC.@preserve ks vs kp vp @test solver_options(2, pointer(kp), pointer(vp)) == ["tol" => "1e-8", "max_iter" => "50"]
    end
end
