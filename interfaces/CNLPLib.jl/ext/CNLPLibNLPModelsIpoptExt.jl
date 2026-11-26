"Precompiles an Ipopt solve (NLPModelsIpopt) of a `CNLPModel` and of a `ProblemModel` (see src/precompile.jl)."
module CNLPLibNLPModelsIpoptExt

using CNLPLib, NLPModelsIpopt
using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    @compile_workload begin
        for fixed in (false, true)
            ipopt(CNLPLib._toy_model(; fixed); print_level = 0, sb = "yes")
            ipopt(CNLPLib._toy_problem(; fixed); print_level = 0, sb = "yes")
        end
    end
end

end # module
