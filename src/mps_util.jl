function evaluate_mps(
    mps::Union{ITensors.MPS,ITensors.MPO},
    indexspecs::Vararg{AbstractVector{<:Tuple{ITensors.Index,Int}}})
    V = ITensor(1.0)
    for j in eachindex(indexspecs[1])
        states = prod(state(spec[j]...) for spec in indexspecs)
        V *= mps[j] * states
    end
    return scalar(V)
end

function evaluate_mps(
    mps::Union{ITensors.MPS,ITensors.MPO},
    indices::AbstractVector{<:ITensors.Index},
    indexvalues::AbstractVector{Int})
    return evaluate_mps(mps, collect(zip(indices, indexvalues)))
end