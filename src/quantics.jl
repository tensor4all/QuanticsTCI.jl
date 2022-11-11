function quantics_to_index(bitlist::Union{Array{Int},NTuple{N,Int}}) where {N}
    # Must be signed int to avoid https://github.com/JuliaLang/julia/issues/44895
    result::Int = 0
    for i in eachindex(bitlist)
        result += (bitlist[i] - 1) * (1 << (i - 1))
    end
    return result + 1
end

function index_to_quantics(index::Int, n::Int)
    index -= 1
    result = ones(Int, n)
    for i in 1:n
        result[i] += (index & (1 << (i - 1))) != 0
    end
    return result
end
