"""
Precompiles a cuMadNLP solve of a `CNLPModel{CuVector}` (see src/precompile.jl), on a machine
whose GPU works at precompile time; elsewhere it compiles nothing.
"""
module CNLPLibcuMadNLPExt

using CNLPLib, cuMadNLP, CUDACore
using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    cx = Base.get_extension(CNLPLib, :CNLPLibCUDAExt)
    if cx !== nothing && CUDACore.functional()
        @compile_workload begin
            cuMadNLP.madnlp(cx._toy_device_model(); print_level = cuMadNLP.MadCore.ERROR)
        end
    end
end

end # module
