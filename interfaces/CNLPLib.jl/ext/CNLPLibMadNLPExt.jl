"Precompiles a MadNLP solve of a `CNLPModel` and of a `ProblemModel` (see src/precompile.jl)."
module CNLPLibMadNLPExt

using CNLPLib, MadNLP
using PrecompileTools: @setup_workload, @compile_workload

# the reports and the solution file for a result of this solver (`CNLPModel` from a library;
# the toy model has none to report)
function _precompile_result(R)
    M = CNLPLib.CNLPModel{Vector{Float64}, Vector{Int}}
    precompile(CNLPLib.report_solution, (M, R))
    precompile(CNLPLib._solution_record, (M, R))
    precompile(CNLPLib.write_solution, (M, String, R))
end

@setup_workload begin
    @compile_workload begin
        for fixed in (false, true)
            madnlp(CNLPLib._toy_model(; fixed); print_level = MadNLP.ERROR)
            madnlp(CNLPLib._toy_problem(; fixed); print_level = MadNLP.ERROR)
        end
        _precompile_result(typeof(madnlp(CNLPLib._toy_model(); print_level = MadNLP.ERROR)))
    end
end

end # module
