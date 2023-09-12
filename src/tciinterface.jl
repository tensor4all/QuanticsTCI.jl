struct QuanticsTensorCI2{ValueType}
    tt::TensorCrossInterpolation.TensorCI2{ValueType}
    unfoldingscheme::UnfoldingSchemes.UnfoldingScheme
end

function evaluate(
    qtci::QuanticsTensorCI2{ValueType},
    indices::Union{Array{Int},NTuple{N,Int}}
)::ValueType where {N,ValueType}
    bitlist = index_to_quantics(
        indices, length(qtci.tt.localset); unfoldingscheme=qtci.unfoldingscheme)
    return TensorCrossInterpolation.evaluate(qtci.tt, bitlist)
end

function evaluate(qtci::QuanticsTensorCI2{V}, indices::Int...)::V where {V}
    return evaluate(qtci, collect(indices))
end

function (qtci::QuanticsTensorCI2{V})(indices)::V where {V}
    return evaluate(qtci, indices)
end

function quantics_to_x(
    bitlist::Union{Array{Int},NTuple{N,Int}},
    xvals::AbstractVector{<:AbstractVector};
    unfoldingscheme::UnfoldingSchemes.UnfoldingScheme=UnfoldingSchemes.fused
) where {N}
    indices = quantics_to_index(bitlist, length(xvals); unfoldingscheme=unfoldingscheme)
    return [x[i] for (x, i) in zip(xvals, indices)]
end

function quanticscrossinterpolate(
    ::Type{ValueType},
    f,
    xvals::AbstractVector{<:AbstractVector},
    initialpivots::AbstractVector{<:AbstractVector}=[ones(Int, length(xvals))];
    unfoldingscheme::UnfoldingSchemes.UnfoldingScheme=UnfoldingSchemes.interleaved,
    kwargs...
) where {ValueType}
    localdimensions = log2.(length.(xvals))
    if !allequal(localdimensions)
        throw(ArgumentError(
            "This method only supports grids with equal number of points in each direction. If you need a different grid, please use index_to_quantics and quantics_to_index and determine the index ordering yourself."))
    elseif !all(isinteger.(localdimensions))
        throw(ArgumentError("This method only supports grid sizes that are powers of 2."))
    end
    n = length(xvals)
    R = Int(first(localdimensions))
    L = n * R

    qf(q) = f(quantics_to_x(q, xvals, unfoldingscheme=unfoldingscheme))
    qinitialpivots = index_to_quantics.(initialpivots, R; unfoldingscheme=unfoldingscheme)
    qtt, ranks, errors = TensorCrossInterpolation.crossinterpolate2(
        ValueType, qf, fill(2, L), qinitialpivots; kwargs...)
    return QuanticsTensorCI2{ValueType}(qtt, unfoldingscheme), ranks, errors
end

function quanticscrossinterpolate(
    ::Type{ValueType},
    f,
    xvals::AbstractVector,
    initialpivots::AbstractVector=[1];
    kwargs...
) where {ValueType}
    return quanticscrossinterpolate(
        ValueType,
        x -> f(first(x)),
        [xvals],
        [initialpivots];
        kwargs...)
end
