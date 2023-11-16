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

function _checkdigits(::Val{B}, digitlist) where {B}
    maximum(digitlist) <= B || error("maximum(digitlist) <= B")
    minimum(digitlist) >= 0 || error("minimum(digitlist) >= 0")
end

"""
    fuse_dimensions([base=Val(2)], digitlists...)

Fuse d digitlists that represent a quantics index into a digitlist where each bit
has dimension base^d. This fuses legs for different dimensions that have equal length
scale (see QTCI paper).

Inverse of [`unfuse_dimensions`](@ref).
"""
function fuse_dimensions(::Val{B}, digitlists...) where {B}
    _checkdigits.(Val(B), digitlists)
    result = ones(Int, length(digitlists[1]))
    return fuse_dimensions!(Val(B), result, digitlists...)
end

function fuse_dimensions!(::Val{B}, fused::AbstractArray{<:Integer}, digitlists...) where {B}
    p = 1
    for d in eachindex(digitlists)
        @. fused += (digitlists[d] - 1) * p
        p *= B
    end
    return fused
end


"""
    unfuse_dimensions([base=Val(2)], digitlist, d)

Unfuse up a fused digitlist with bits of dimension base^d into d digitlists where each bit has dimension `base`.
Inverse of [`fuse_dimensions`](@ref).
"""
function unfuse_dimensions(::Val{B}, digitlist, d) where {B}
    result = [zeros(Int, length(digitlist)) for _ in 1:d]
    return unfuse_dimensions!(Val(B), result, digitlist)
end

function unfuse_dimensions!(::Val{B}, digitlists, digitlist) where {B}
    ndim = length(digitlists)
    R  = length(digitlist)
    for i in 1:ndim
        for j in 1:R
           digitlists[i][j] = _digit_at_index(Val(B), digitlist[j], ndim-i+1; numdigits=ndim) 
        end
    end
    return digitlists
end


"""
    interleave_dimensions(digitlists...)

Interleaves the indices of all digitlists into one long digitlist. Use this for
quantics representation of multidimensional objects without fusing indices.
Inverse of [`deinterleave_dimensions`](@ref).
"""
function interleave_dimensions(digitlists...)::Vector{Int}
    results = Vector{Int}(undef, length(digitlists[1]) * length(digitlists))
    return interleave_dimensions!(results, digitlists...)
end

function interleave_dimensions!(interleaved_digitlist::AbstractArray{<:Integer}, digitlists...)
    idx = 1
    for i in eachindex(digitlists[1])
        for d in eachindex(digitlists)
            interleaved_digitlist[idx] = digitlists[d][i]
            idx += 1
        end
    end
    return interleaved_digitlist
end

"""
    deinterleave_dimensions(digitlist, d)

Reverses the interleaving of bits, i.e. yields digitlists for each dimension from
a long interleaved digitlist. Inverse of [`interleave_dimensions`](@ref).
"""
function deinterleave_dimensions(digitlist, d)
    return [digitlist[i:d:end] for i in 1:d]
end


function deinterleave_dimensions!(deinterleaved_digitlists::AbstractArray{<:AbstractArray{I}}, digitlist) where {I<:Integer}
    d = length(deinterleaved_digitlists)
    for i in 1:d
        @. deinterleaved_digitlists[i] = digitlist[i:d:end]
    end
    return deinterleaved_digitlists
end


"""
    quantics_to_index_fused(
        ::Val{B}, ::Val{d}, digitlist::AbstractVector{<:Integer}
    )::NTuple{d,Int} where {B, d}

Convert a d-dimensional index from fused quantics representation to d Integers.

* `B`           base for quantics (default: 2)
* `d`           number of dimensions
* `digitlist`     base-b representation

See also [`quantics_to_index_interleaved`](@ref).
"""
function quantics_to_index_fused(
    digitlist::AbstractVector{<:Integer};
    base::Val{B}=Val(2), dims::Val{d}=Val(1)
)::NTuple{d,Int} where {B, d}
    R = length(digitlist)
    result = ones(MVector{d,Int})

    maximum(digitlist) <= B^d || error("maximum(digitlist) <= B^d")
    minimum(digitlist) >= 0 || error("minimum(digitlist) >= 0")

    for n in 1:R # from the least to most significant digit
        scale = B^(n-1) # length scale
        tmp = digitlist[R-n+1] - 1
        for i in 1:d # in the order of 1st dim, 2nd dim, ...
            div_, rem_ = divrem(tmp, B)
            result[i] += rem_ * scale
            tmp = div_
        end
    end

    return tuple(result...)
end


"""
* `digitlist`     base-b representation (1d vector)
* `B`           base for quantics (default: 2)
"""
function index_to_quantics!(digitlist, index::Integer; base::Val{B}=Val(2)) where {B}
    numdigits = length(digitlist)
    for i in 1:numdigits
        digitlist[i] = mod(index - 1, B^(numdigits-i+1)) ÷ B^(numdigits-i) + 1
    end
    return digitlist
end

"""
Does the opposite of [`quantics_to_index_fused!`](@ref)

* `D`  
"""
function index_to_quantics_fused!(digitlist::AbstractVector{<:Integer}, index::NTuple{D,<:Integer}; base::Val{B}=Val(2)) where {B,D}
    R = length(digitlist)
    ndims = length(index)
    digitlist .= 1
    for dim in 1:ndims
        for i in 1:R # from the left to right
           digitlist[i] += (B^(dim-1)) * (_digit_at_index(index[dim], i; numdigits=R, base=base) - 1)
        end
    end
    return digitlist
end


"""
    index_to_quantics(::Val{B}, index::Integer; numdigits=8)

Does the same as [`index_to_quantics!`](@ref) but returns a new vector.
"""
function index_to_quantics(index::Integer; numdigits=8, base::Val{B}=Val(2)) where {B}
    digitlist = Vector{Int}(undef, numdigits)
    return index_to_quantics!(digitlist, index; base=base)
end


"""
`index`, `position` and the result are one-based.

`B`: base
`index`: The integer to look at.
`position=1`: Specify the position of the digit to look at. 1 is the most significant (most left) digit.
`numdigits=8`: Specify the number of digits in the number `index`.
"""
function _digit_at_index(index, position; numdigits=8, base::Val{B}=Val(2)) where {B}
    p_ = numdigits - position + 1
    return mod((index-1), B^p_) ÷ B^(p_-1) + 1
end