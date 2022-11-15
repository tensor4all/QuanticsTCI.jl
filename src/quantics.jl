"""
Split up a d-dimensional quantics index into d 1-dimensional quantics indices.
"""
function split_dimensions(bitlist, d)
    dimensions_bitmask = 2 .^ (0:(d-1))
    return [
        (((bitlist .- 1) .& dimensions_bitmask[i]) .!= 0) .+ 1
        for i in 1:d
    ]
end

"""
Convert a d-dimensional index from quantics representation to d Integers.

bitlist     binary representation
d           number of dimensions
"""
function quantics_to_index(
    bitlist::Union{Array{Int},NTuple{N,Int}};
    d=1
) where {N}
    # Must be signed int to avoid https://github.com/JuliaLang/julia/issues/44895
    result = zeros(Int, d)

    dimensions_bitmask = 2 .^ (0:(d-1))
    for i in eachindex(bitlist)
        result .+= (((bitlist[i] - 1) .& dimensions_bitmask) .!= 0) .* (1 << (i - 1))
    end
    return result .+ 1
end

"""
Convert an integer to its binary representation.

index       an integer
numdigits   how many digits to zero-pad to
"""
function binary_representation(index::Int; numdigits=8)
    return [(index & (1 << (i - 1))) != 0 for i in 1:numdigits]
end

"""
Convert d indices to quantics representation, with n digits.
"""
function index_to_quantics(indices::Array{Int}, n::Int)
    result = [binary_representation(indices[i] - 1; numdigits=n) * 2^(i - 1)
              for i in eachindex(indices)]
    return [sum(r[i] for r in result) for i in 1:n] .+ 1
end

"""
Convert a single index to quantics representation.
"""
function index_to_quantics(index::Int, n::Int)
    return index_to_quantics([index], n)
end
