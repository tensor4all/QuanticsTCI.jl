"""
    merge_dimensions(bitlists...)

Merge d bitlists that represent a quantics index into a bitlist where each bit
has dimension 2^d. This merges legs for different dimensions, but equal length
scale.
"""
function merge_dimensions(bitlists...)
    return sum(
        [(bitlists[d] .- 1) .* (2^(d - 1)) for d in eachindex(bitlists)];
        dims=1)[1] .+ 1
end

"""
    split_dimensions(bitlist, d)

Split up a merged bitlist with bits of dimension 2^d into d bitlists where each
bit has dimension 2. This is the inverse of `merge_dimensions`.
"""
function split_dimensions(bitlist, d)
    dimensions_bitmask = 2 .^ (0:(d-1))
    return [
        (((bitlist .- 1) .& dimensions_bitmask[i]) .!= 0) .+ 1
        for i in 1:d
    ]
end

"""
    interleave_dimensions(bitlists...)

Interleaves the indices of all bitlists into one long bitlist. Use this for
quantics representation of multidimensional objects without merging indices.
"""
function interleave_dimensions(bitlists...)
    return [bitlists[d][i] for i in eachindex(bitlists[1]) for d in eachindex(bitlists)]
end

"""
    deinterleave_dimensions(bitlist, d)

Reverses the interleaving of bits, i.e. yields bitlists for each dimension from
a long interleaved bitlist. Inverse of `interleave_dimensions()`.
"""
function deinterleave_dimensions(bitlist, d)
    return [bitlist[i:d:end] for i in 1:d]
end

"""
    quantics_to_index(
        bitlist::Union{Array{Int},NTuple{N,Int}};
        d=1
    ) where {N}

Convert a d-dimensional index from quantics representation to d Integers.

 * `bitlist`     binary representation
 * `d`           number of dimensions
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
    binary_representation(index::Int; numdigits=8)

Convert an integer to its binary representation.

 * `index`       an integer
 * `numdigits`   how many digits to zero-pad to
"""
function binary_representation(index::Int; numdigits=8)
    return [(index & (1 << (i - 1))) != 0 for i in 1:numdigits]
end

"""
    index_to_quantics(indices::Array{Int}, n::Int)

Convert d indices to quantics representation with n digits.
"""
function index_to_quantics(indices::Array{Int}, n::Int)
    result = [binary_representation(indices[d] - 1; numdigits=n) * 2^(d - 1)
              for d in eachindex(indices)]
    return [sum(r[i] for r in result) for i in 1:n] .+ 1
end

"""
    index_to_quantics(index::Int, n::Int)

Convert a single index to quantics representation.
"""
function index_to_quantics(index::Int, n::Int)
    return index_to_quantics([index], n)
end
