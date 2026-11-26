"""
The standard solution format, cnlp-solution 0.2 (formats/solution.md of cnlplib): root
attributes, the `/solution` group ExaModelsIO also uses, and `/blocks/<kind>/<name>`.
"""
module CNLPLibHDF5Ext

using CNLPLib, HDF5
import PrecompileTools

function CNLPLib._write_standard(path::AbstractString, r)
    h5open(path, "w") do f
        a = attributes(f)
        a["format"] = "cnlp-solution"; a["format_version"] = "0.2"
        a["library"] = r.library; a["model"] = r.model; a["nvar"] = r.nvar; a["ncon"] = r.ncon
        g = create_group(f, "solution")
        g["solution"] = r.x; g["multipliers"] = r.y; g["multipliers_L"] = r.zl; g["multipliers_U"] = r.zu
        g["objective"] = r.objective; g["minimize"] = Int8(r.minimize); g["status"] = r.status
        g["solver_status"] = r.solver_status; g["iterations"] = r.iterations
        isempty(r.blocks) && return
        b = create_group(f, "blocks")
        for (kind, name, fields) in r.blocks
            k = haskey(b, kind) ? b[kind] : create_group(b, kind)
            h = create_group(k, name)
            attributes(h)["dims"] = collect(Int64, size(last(first(fields))))
            for (field, arr) in fields
                h[field] = arr
            end
        end
    end
    return path
end

# the standard solution file, with one block of each field layout
PrecompileTools.@compile_workload begin
    v = Pair{String, Array{Float64}}["value" => zeros(2), "multiplier_L" => zeros(2)]
    r = (; library = "toy", model = "toy", nvar = 2, ncon = 1, minimize = true, objective = 0.0,
         status = "solved", solver_status = "toy", iterations = 1, x = zeros(2), y = zeros(1),
         zl = zeros(2), zu = zeros(2), blocks = [("variables", "x", v)])
    path = tempname() * ".h5"
    CNLPLib._write_standard(path, r)
    rm(path; force = true)
end

end # module
