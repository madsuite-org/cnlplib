# A problem given as C callbacks, as a cnlp solver receives it, built here from Julia
# functions: min (x1 - 1)^2 + (x2 - 2)^2 subject to x1 + x2 = 1, 0 <= x <= 5.
using CNLPLib: CNLPProblem, CNLPResult, ProblemModel, set_result!, set_failure!, solver_options,
    CallbackError, CNLP_SOLVED, CNLP_ERROR

const FAIL = Ref(false)
wrap(p, n) = unsafe_wrap(Array, p, n)
function c_obj(::Ptr{Cvoid}, x::Ptr{Float64}, f::Ptr{Float64})
    FAIL[] && return Int64(7)
    v = wrap(x, 2)
    unsafe_store!(f, (v[1] - 1)^2 + (v[2] - 2)^2)
    return Int64(0)
end
c_grad(::Ptr{Cvoid}, x::Ptr{Float64}, g::Ptr{Float64}) =
    (v = wrap(x, 2); w = wrap(g, 2); w[1] = 2(v[1] - 1); w[2] = 2(v[2] - 2); Int64(0))
c_cons(::Ptr{Cvoid}, x::Ptr{Float64}, c::Ptr{Float64}) = (v = wrap(x, 2); unsafe_store!(c, v[1] + v[2]); Int64(0))
c_js(::Ptr{Cvoid}, r::Ptr{Int64}, k::Ptr{Int64}) = (wrap(r, 2) .= [1, 1]; wrap(k, 2) .= [1, 2]; Int64(0))
c_jac(::Ptr{Cvoid}, ::Ptr{Float64}, v::Ptr{Float64}) = (wrap(v, 2) .= 1.0; Int64(0))
c_hs(::Ptr{Cvoid}, r::Ptr{Int64}, k::Ptr{Int64}) = (wrap(r, 2) .= [1, 2]; wrap(k, 2) .= [1, 2]; Int64(0))
c_hess(::Ptr{Cvoid}, ::Ptr{Float64}, ::Ptr{Float64}, w::Float64, v::Ptr{Float64}) = (wrap(v, 2) .= 2w; Int64(0))

const X0, LV, UV, LC, UC = [0.5, 0.5], [0.0, 0.0], [5.0, 5.0], [1.0], [1.0]

function the_problem(minimize = 1)
    fs = (@cfunction(c_obj, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64})),
          @cfunction(c_grad, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64})),
          @cfunction(c_cons, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64})),
          @cfunction(c_js, Int64, (Ptr{Cvoid}, Ptr{Int64}, Ptr{Int64})),
          @cfunction(c_jac, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64})),
          @cfunction(c_hs, Int64, (Ptr{Cvoid}, Ptr{Int64}, Ptr{Int64})),
          @cfunction(c_hess, Int64, (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Float64, Ptr{Float64})))
    p = CNLPProblem(C_NULL, 2, 1, 2, 2, minimize, pointer(X0), pointer(LV), pointer(UV),
                    pointer(LC), pointer(UC), C_NULL, C_NULL, C_NULL,
                    fs...)
    return p, fs
end

@testset "a problem given as Julia callbacks" begin
    p, fs = the_problem()
    GC.@preserve fs begin
        m = ProblemModel(p)

        @testset "the problem as an NLPModel" begin
            @test m isa AbstractNLPModel{Float64, Vector{Float64}}
            @test m.meta.nvar == 2 && m.meta.ncon == 1 && m.meta.nnzj == 2 && m.meta.nnzh == 2
            @test m.meta.x0 == X0 && m.meta.lcon == LC && m.meta.minimize
            x = [0.25, 0.75]
            @test obj(m, x) ≈ 0.75^2 + 1.25^2
            @test grad(m, x) ≈ [-1.5, -2.5]
            @test cons(m, x) ≈ [1.0]
            @test jac_structure(m) == ([1, 1], [1, 2]) && jac_coord(m, x) == [1.0, 1.0]
            @test hess_structure(m) == ([1, 2], [1, 2]) && hess_coord(m, x, [0.0]) == [2.0, 2.0]
            @test jprod(m, x, [1.0, 2.0]) ≈ [3.0]
            @test jtprod(m, x, [2.0]) ≈ [2.0, 2.0]
            @test hprod(m, x, [0.0], [1.0, -1.0]) ≈ [2.0, -2.0]
        end

        @testset "a failing callback throws, by name" begin
            FAIL[] = true
            err = try obj(m, [0.0, 1.0]); nothing catch e; e end
            FAIL[] = false
            @test err isa CallbackError && err.name == :obj && err.status == 7
        end

        @testset "the result in the cnlp sign convention" begin
            for (minimize, s) in ((1, 1.0), (0, -1.0))
                q, gs = the_problem(minimize)
                GC.@preserve gs begin
                    pm = ProblemModel(q)
                    x, y, zl, zu = zeros(2), zeros(1), zeros(2), zeros(2)
                    r = Ref(CNLPResult(pointer(x), pointer(y), pointer(zl), pointer(zu), 0.0, -1, -1,
                                       ntuple(_ -> 0x00, 128)))
                    GC.@preserve x y zl zu r begin
                        rp = Base.unsafe_convert(Ptr{CNLPResult}, r)
                        set_result!(rp, pm, [0.0, 1.0], [2.0], [0.5, 0.0], [0.0, 0.25], 1.0, CNLP_SOLVED, 4, "Optimal")
                        out = r[]
                        @test x == [0.0, 1.0] && y == [s * 2.0] && zl == [s * 0.5, 0.0] && zu == [0.0, s * 0.25]
                        @test out.objective == 1.0 && out.status == CNLP_SOLVED && out.iterations == 4
                        @test String(UInt8[c for c in out.solver_status if c != 0]) == "Optimal"
                        set_failure!(rp, "it broke")
                        @test r[].status == CNLP_ERROR && isnan(r[].objective) &&
                              String(UInt8[c for c in r[].solver_status if c != 0]) == "it broke"
                    end
                end
            end
        end

        @testset "options" begin
            ks, vs = ["tol", "max_iter"], ["1e-8", "50"]
            kp, vp = pointer.(ks), pointer.(vs)
            GC.@preserve ks vs kp vp begin
                @test solver_options(2, pointer(kp), pointer(vp)) == ["tol" => "1e-8", "max_iter" => "50"]
            end
        end
    end
end
