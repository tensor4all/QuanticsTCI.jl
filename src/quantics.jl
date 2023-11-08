"""
    module UnfoldingSchemes

Contains an enum to choose between interleaved and fused representation during quantics
conversion / unfolding. Choose between `UnfoldingSchemes.interleaved` and
`UnfoldingSchemes.fused`.
"""
module UnfoldingSchemes
@enum UnfoldingScheme begin
    interleaved
    fused
end
end

"""
    fuse_dimensions([base=Val(2)], bitlists...)

Merge d bitlists that represent a quantics index into a bitlist where each bit
has dimension base^d. This fuses legs for different dimensions that have equal length
scale (see QTCI paper).

Inverse of [`split_dimensions`](@ref).
"""
function fuse_dimensions(base::Val{B}, bitlists...) where {B}
    result = ones(Int, length(bitlists[1]))
    return fuse_dimensions!(base, result, bitlists...)
end

fuse_dimensions(bitlists...) = fuse_dimensions(Val(2), bitlists...)

function fuse_dimensions!(::Val{B}, fused::AbstractArray{<:Integer}, bitlists...) where {B}
    p = 1
    for d in eachindex(bitlists)
        @. fused += (bitlists[d] - 1) * p
        p *= B
    end
    return fused
end

fuse_dimensions!(fused::AbstractArray{<:Integer}, bitlists...) = fuse_dimensions!(Val(2), fused::AbstractArray{<:Integer}, bitlists...)

"""
    function merge_dimensions(bitlists...)

See [`fuse_dimensions`](@ref).
"""
function merge_dimensions(bitlists...)
    return fuse_dimensions(bitlists...)
end

"""
    split_dimensions([base=Val(2)], bitlist, d)

Split up a merged bitlist with bits of dimension base^d into d bitlists where each bit has dimension `base`.
Inverse of [`fuse_dimensions`](@ref).
"""
function split_dimensions(base::Val{B}, bitlist, d) where {B}
    result = [zeros(Int, length(bitlist)) for _ in 1:d]
    return split_dimensions!(base, result, bitlist)
end

function split_dimensions!(base::Val{B}, bitlists, bitlist) where {B}
    d = length(bitlists)
    p = 1
    for i in 1:d
        bitlists[i] .= (((bitlist .- 1) .& p) .!= 0) .+ 1
        p *= B
    end
    return bitlists
end

split_dimensions(bitlist, d) = split_dimensions(Val(2), bitlist, d)

split_dimensions!(bitlists, bitlist) = split_dimensions!(Val(2), bitlists, bitlist)

"""
    interleave_dimensions(bitlists...)

Interleaves the indices of all bitlists into one long bitlist. Use this for
quantics representation of multidimensional objects without fusing indices.
Inverse of [`deinterleave_dimensions`](@ref).
"""
function interleave_dimensions(bitlists...)::Vector{Int}
    results = Vector{Int}(undef, length(bitlists[1]) * length(bitlists))
    return interleave_dimensions!(results, bitlists...)
end

function interleave_dimensions!(interleaved_bitlist::AbstractArray{<:Integer}, bitlists...)
    idx = 1
    for i in eachindex(bitlists[1])
        for d in eachindex(bitlists)
            interleaved_bitlist[idx] = bitlists[d][i]
            idx += 1
        end
    end
    return interleaved_bitlist
end

"""
    deinterleave_dimensions(bitlist, d)

Reverses the interleaving of bits, i.e. yields bitlists for each dimension from
a long interleaved bitlist. Inverse of [`interleave_dimensions`](@ref).
"""
function deinterleave_dimensions(bitlist, d)
    return [bitlist[i:d:end] for i in 1:d]
end

function deinterleave_dimensions!(deinterleaved_bitlists::AbstractArray{<:AbstractArray{I}}, bitlist) where {I<:Integer}
    d = length(deinterleaved_bitlists)
    for i in 1:d
        @. deinterleaved_bitlists[i] = bitlist[i:d:end]
    end
    return deinterleaved_bitlists
end


"""
    quantics_to_index_fused(
        ::Val{B}, ::Val{d}, bitlist::AbstractVector{<:Integer}
    )::NTuple{d,Int} where {B, d}

Convert a d-dimensional index from fused quantics representation to d Integers.

* `B`           base for quantics (default: 2)
* `d`           number of dimensions
* `bitlist`     binary representation

See also [`quantics_to_index_interleaved`](@ref).
"""
function quantics_to_index_fused(
    ::Val{B}, ::Val{d}, bitlist::AbstractVector{<:Integer}
)::NTuple{d,Int} where {B, d}
    n = length(bitlist)
    dimensions_bitmask = ntuple(i->B .^ (i-1), d)
    result = ones(MVector{d,Int})
    for i in eachindex(bitlist)
        result .+= (((bitlist[i] - 1) .& dimensions_bitmask) .!= 0) .* (B^(n - i))
    end
    return tuple(result...)
end

function quantics_to_index_fused(
    ::Val{d}, bitlist::AbstractVector{<:Integer}
)::NTuple{d,Int} where {d}
    return quantics_to_index_fused(Val{2}, Val(d), bitlist)
end


"""
    binary_representation(index::Int; numdigits=8)

Convert an integer to its binary representation.

 * `index`       an integer
 * `numdigits`   how many digits to zero-pad to
"""
function binary_representation(index::Int; numdigits=8) where {Base}
    return [(index & (1 << (numdigits - i))) != 0 for i in 1:numdigits]
end

function bitlist(::Val{B}, index::Int; numdigits=8) where {B}
    return [mod(index - 1, B^(numdigits-i+1)) ÷ B^(numdigits-i) + 1 for i in 1:numdigits]
end