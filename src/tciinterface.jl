struct QuanticsTensorCI2{ValueType}
    tt::TensorCrossInterpolation.TensorCI2{ValueType}
    unfoldingscheme::UnfoldingSchemes.UnfoldingScheme
end

function evaluate(
    qtci::QuanticsTensorCI2{ValueType},
    indices::Union{Array{Int},NTuple{N,Int}}
)::ValueType where {N,ValueType}
    R = if qtci.unfoldingscheme == UnfoldingSchemes.interleaved
        div(length(qtci.tt), length(indices))
    else
        length(qtci.tt)
    end
    bitlist = index_to_quantics(indices, R; unfoldingscheme=qtci.unfoldingscheme)
    return TensorCrossInterpolation.evaluate(qtci.tt, bitlist)
end

function evaluate(qtci::QuanticsTensorCI2{V}, indices::Int...)::V where {V}
    return evaluate(qtci, collect(indices)::Vector{Int})
end

function (qtci::QuanticsTensorCI2{V})(indices)::V where {V}
    return evaluate(qtci, indices)
end

function (qtci::QuanticsTensorCI2{V})(indices::Int...)::V where {V}
    return evaluate(qtci, indices...)
end

function quantics_to_x(
    bitlist::Union{Array{Int},NTuple{N,Int}},
    xvals::AbstractVector{<:AbstractVector};
    unfoldingscheme::UnfoldingSchemes.UnfoldingScheme=UnfoldingSchemes.fused
) where {N}
    indices = quantics_to_index(bitlist, length(xvals); unfoldingscheme=unfoldingscheme)
    return [x[i] for (x, i) in zip(xvals, indices)]
end

@doc raw"""
    function quanticscrossinterpolate(
        ::Type{ValueType},
        f,
        xvals::AbstractVector{<:AbstractVector},
        initialpivots::AbstractVector{<:AbstractVector}=[ones(Int, length(xvals))];
        unfoldingscheme::UnfoldingSchemes.UnfoldingScheme=UnfoldingSchemes.interleaved,
        kwargs...
    ) where {ValueType}

Interpolate a function ``f(\mathbf{x})`` as a quantics tensor train. The tensor train itself is constructed using the 2-site tensor cross interpolation algorithm implemented in [`TensorCrossInterpolation.crossinterpolate2`](https://tensors4fields.gitlab.io/tensorcrossinterpolation.jl/dev/documentation/#TensorCrossInterpolation.crossinterpolate2-Union{Tuple{N},%20Tuple{ValueType},%20Tuple{Type{ValueType},%20Any,%20Union{Tuple{Vararg{Int64,%20N}},%20Vector{Int64}}},%20Tuple{Type{ValueType},%20Any,%20Union{Tuple{Vararg{Int64,%20N}},%20Vector{Int64}},%20Vector{Vector{Int64}}}}%20where%20{ValueType,%20N}).

Arguments:
- `ValueType` is the return type of `f`. Automatic inference is too error-prone.
- `f` is the function to be interpolated. `f` may take multiple arguments. The return type should be `ValueType`.
- `xvals` is a vector of discretization grids, where the `i`th grid corresponds to the `i`th argument of `f`.
- `initialpivots` is a vector of pivots to be used for initialization. Default: `[[1, 1, ...]]`.
- `unfoldingscheme` determines whether the *interleaved* or *fused* representation is used. (See the [quantics TCI paper](http://arxiv.org/abs/2303.11819).)

All other arguments are forwareded to `crossinterpolate2`. Most importantly:
- `tolerance::Float64` is a float specifying the target tolerance for the interpolation. Default: `1e-8`.
- `pivottolerance::Float64` is a float that specifies the tolerance for adding new pivots, i.e. the truncation of tensor train bonds. It should be <= tolerance, otherwise convergence may be impossible. Default: `tolerance`.
- `maxbonddim::Int` specifies the maximum bond dimension for the TCI. Default: `typemax(Int)`, i.e. effectively unlimited.
- `maxiter::Int` is the maximum number of iterations (i.e. optimization sweeps) before aborting the TCI construction. Default: `200`.

For all other arguments, see the documentation for [`TensorCrossInterpolation.crossinterpolate2`](https://tensors4fields.gitlab.io/tensorcrossinterpolation.jl/dev/documentation/#TensorCrossInterpolation.crossinterpolate2-Union{Tuple{N},%20Tuple{ValueType},%20Tuple{Type{ValueType},%20Any,%20Union{Tuple{Vararg{Int64,%20N}},%20Vector{Int64}}},%20Tuple{Type{ValueType},%20Any,%20Union{Tuple{Vararg{Int64,%20N}},%20Vector{Int64}},%20Vector{Vector{Int64}}}}%20where%20{ValueType,%20N}).
"""
function quanticscrossinterpolate(
    ::Type{ValueType},
    f,
    xvals::AbstractVector{<:AbstractVector},
    initialpivots::AbstractVector{<:AbstractVector}=[ones(Int, length(xvals))];
    unfoldingscheme::UnfoldingSchemes.UnfoldingScheme=UnfoldingSchemes.interleaved,
    nrandominitpivot = 5,
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

    qlocaldimensions = if unfoldingscheme == UnfoldingSchemes.interleaved
        fill(2, n * R)
    else
        fill(2^n, R)
    end

    qf = TensorCrossInterpolation.CachedFunction{ValueType}(
        q -> f(quantics_to_x(q, xvals, unfoldingscheme=unfoldingscheme)...),
        qlocaldimensions
    )
    qinitialpivots = index_to_quantics.(initialpivots, R; unfoldingscheme=unfoldingscheme)

    # For stabity
    kwargs_ = Dict{Symbol,Any}(kwargs)
    if !(:nsearchglobalpivot ∈ keys(kwargs))
        kwargs_[:nsearchglobalpivot] = 5
    end
    if !(:partialnesting ∈ keys(kwargs))
        kwargs_[:partialnesting] = false
    end

    # random initial pivot
    for _ in 1:nrandominitpivot
       pivot = [rand(1:d) for d in qlocaldimensions]
       push!(
           qinitialpivots,
           TensorCrossInterpolation.optfirstpivot(qf, qlocaldimensions, pivot)
       )
    end

    qtt, ranks, errors = TensorCrossInterpolation.crossinterpolate2(
        ValueType, qf, qlocaldimensions, qinitialpivots; kwargs_...)
    return QuanticsTensorCI2{ValueType}(qtt, unfoldingscheme), ranks, errors
end

@doc raw"""
    function quanticscrossinterpolate(
        ::Type{ValueType},
        f,
        xvals::AbstractVector,
        initialpivots::AbstractVector=[1];
        kwargs...
    ) where {ValueType}

Interpolate a function ``f(x)`` as a quantics tensor train. The tensor train itself is constructed using the 2-site tensor cross interpolation algorithm implemented in [`TensorCrossInterpolation.crossinterpolate2`](https://tensors4fields.gitlab.io/tensorcrossinterpolation.jl/dev/documentation/#TensorCrossInterpolation.crossinterpolate2-Union{Tuple{N},%20Tuple{ValueType},%20Tuple{Type{ValueType},%20Any,%20Union{Tuple{Vararg{Int64,%20N}},%20Vector{Int64}}},%20Tuple{Type{ValueType},%20Any,%20Union{Tuple{Vararg{Int64,%20N}},%20Vector{Int64}},%20Vector{Vector{Int64}}}}%20where%20{ValueType,%20N}).

Arguments:
- `ValueType` is the return type of `f`. Automatic inference is too error-prone.
- `f` is the function to be interpolated. `f` should take a single parameter (otherwise, use the other overload). The return type should be `ValueType`.
- `xvals` is a vector of discretization grids, where the `i`th grid corresponds to the `i`th argument of `f`.
- `initialpivots` is a vector of pivots to be used for initialization. Default: `[[1, 1, ...]]`.

All other arguments are forwarded to `crossinterpolate2`. Most importantly:
- `tolerance::Float64` is a float specifying the target tolerance for the interpolation. Default: `1e-8`.
- `pivottolerance::Float64` is a float that specifies the tolerance for adding new pivots, i.e. the truncation of tensor train bonds. It should be <= tolerance, otherwise convergence may be impossible. Default: `tolerance`.
- `maxbonddim::Int` specifies the maximum bond dimension for the TCI. Default: `typemax(Int)`, i.e. effectively unlimited.
- `maxiter::Int` is the maximum number of iterations (i.e. optimization sweeps) before aborting the TCI construction. Default: `200`.

For all other arguments, see the documentation for [`TensorCrossInterpolation.crossinterpolate2`](https://tensors4fields.gitlab.io/tensorcrossinterpolation.jl/dev/documentation/#TensorCrossInterpolation.crossinterpolate2-Union{Tuple{N},%20Tuple{ValueType},%20Tuple{Type{ValueType},%20Any,%20Union{Tuple{Vararg{Int64,%20N}},%20Vector{Int64}}},%20Tuple{Type{ValueType},%20Any,%20Union{Tuple{Vararg{Int64,%20N}},%20Vector{Int64}},%20Vector{Vector{Int64}}}}%20where%20{ValueType,%20N}).
"""
function quanticscrossinterpolate(
    ::Type{ValueType},
    f,
    xvals::AbstractVector,
    initialpivots::AbstractVector=[1];
    nrandominitpivot = 5,
    kwargs...
) where {ValueType}
    return quanticscrossinterpolate(
        ValueType,
        f,
        [xvals],
        [initialpivots];
        nrandominitpivot=nrandominitpivot,
        kwargs...)
end
