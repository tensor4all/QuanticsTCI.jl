function evaluate_mps(
    mps::Union{ITensors.MPS,ITensors.MPO},
    indexspecs::Vararg{AbstractVector{<:Tuple{ITensors.Index,Number}}})
    V = ITensor(1.0)
    for j in eachindex(indexspecs[1])
        states = prod(state(spec[j]...) for spec in indexspecs)
        V *= mps[j] * states
    end
    return scalar(V)
end