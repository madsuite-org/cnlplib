"Precompiles a MadNLP solve of a `CNLPModel` and of a `ProblemModel` (see src/precompile.jl)."
module CNLPLibMadNLPExt

using CNLPLib, MadNLP
using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    @compile_workload begin
        for fixed in (false, true)
            madnlp(CNLPLib._toy_model(; fixed); print_level = MadNLP.ERROR)
            madnlp(CNLPLib._toy_problem(; fixed); print_level = MadNLP.ERROR)
        end
    end
end

end # module
