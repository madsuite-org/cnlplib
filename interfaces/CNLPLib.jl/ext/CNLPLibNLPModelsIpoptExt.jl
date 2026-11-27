"Precompiles an Ipopt solve (NLPModelsIpopt) of a `CNLPModel` and of a `ProblemModel` (see src/precompile.jl)."
module CNLPLibNLPModelsIpoptExt

using CNLPLib, NLPModelsIpopt
using PrecompileTools: @setup_workload, @compile_workload

# the reports and the solution file for a result of this solver (`CNLPModel` from a library;
# the toy model has none to report)
function _precompile_result(R)
    M = CNLPLib.CNLPModel{Vector{Float64}, Vector{Int}}
    precompile(CNLPLib.report_solution, (M, R))
    precompile(CNLPLib._write_cnlp, (M, String, R))
    precompile(CNLPLib.write_solution, (M, String, R))
end

@setup_workload begin
    @compile_workload begin
        for fixed in (false, true)
            ipopt(CNLPLib._toy_model(; fixed); print_level = 0, sb = "yes")
            ipopt(CNLPLib._toy_problem(; fixed); print_level = 0, sb = "yes")
        end
        # the keyword lists a first call is likely to use (each is its own method instance)
        ipopt(CNLPLib._toy_model(); print_level = 0)
        s = ipopt(CNLPLib._toy_model(); print_level = 0, sb = "yes", tol = 1e-8, max_iter = 3000)
        _precompile_result(typeof(s))
    end
end

end # module
