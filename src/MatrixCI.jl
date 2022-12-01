module MatrixCIutil
    using LinearAlgebra

    """
        AtimesBinv(A::Matrix, B::Matrix)
    
    Calculates the matrix product ``A B^{-1}``, given a rectangular matrix ``A``
    and a square matrix ``B`` in a numerically stable way using QR
    decomposition. This is useful in case ``B^{-1}`` is ill-conditioned.
    """
    function AtimesBinv(A::AbstractMatrix, B::AbstractMatrix)
        m, n = size(A)
        AB = vcat(A, B)
        decomposition = LinearAlgebra.qr(AB)
        QA = decomposition.Q[1:m, 1:n]
        QB = decomposition.Q[(m+1):end, 1:n]
        return QA * inv(QB)
    end

    """
        AtimesBinv(A::Matrix, B::Matrix)
    
    Calculates the matrix product ``A^{-1} B``, given a square matrix ``A``
    and a rectangular matrix ``B`` in a numerically stable way using QR
    decomposition. This is useful in case ``A^{-1}`` is ill-conditioned.
    """
    function AinvtimesB(A::AbstractMatrix, B::AbstractMatrix)
        return AtimesBinv(B', A')'
    end
end

"""
    mutable struct matrix_cross_interpolation{T}

Represents a cross interpolation of a matrix - or, to be more precise, the data 
necessary to evaluate a cross interpolation. This data can be fed into various
methods such as eval(c, i, j) to get interpolated values, and be improved by
dynamically adding pivots.
"""
mutable struct matrix_cross_interpolation{T}
    # "Number of rows of the original matrix."
    # n_rows::Int
    # "Number of columns of the original matrix."
    # n_cols::Int
    "Same as `CrossData.Iset` in xfac code, or ``\\mathcal{I}`` in TCI paper."
    row_indices::Vector{Int}
    "Same as `CrossData.Jset` in xfac code, or ``\\mathcal{J}`` in TCI paper."
    col_indices::Vector{Int}
    "Same as `CrossData.C` in xfac code, or ``A(\\mathbb{I}, \\mathcal{J})`` in TCI paper."
    pivot_cols::Matrix{T}
    "Same as `CrossData.R` in xfac code, or ``A(\\mathcal{I}, \\mathbb{J})`` in TCI paper."
    pivot_rows::Matrix{T}

    function matrix_cross_interpolation{T}(n_rows::Int, n_cols::Int) where {T<:Number}
        return new{T}([], [], zeros(n_rows, 0), zeros(0, n_cols))
    end

    function matrix_cross_interpolation(
        row_indices::AbstractVector{Int}, col_indices::AbstractVector{Int},
        pivot_cols::AbstractMatrix{T}, pivot_rows::AbstractMatrix{T}
        ) where {T<:Number}
        return new{T}(row_indices, col_indices, pivot_cols, pivot_rows)
    end
end

function n_rows(ci::matrix_cross_interpolation{T}) where T
    return size(ci.pivot_cols, 1)
end

function n_cols(ci::matrix_cross_interpolation{T}) where T
    return size(ci.pivot_rows, 2)
end

function pivot_matrix(ci::matrix_cross_interpolation{T}) where T
    return ci.pivot_cols[ci.row_indices, :]
end

function left_matrix(ci::matrix_cross_interpolation{T}) where T
    return MatrixCIutil.AtimesBinv(ci.pivot_cols, pivot_matrix(ci))
end

function right_matrix(ci::matrix_cross_interpolation{T}) where T
    return MatrixCIutil.AinvtimesB(pivot_matrix(ci), ci.pivot_rows)
end

function avail_rows(ci::matrix_cross_interpolation{T}) where T
    return setdiff(1:n_rows(ci), ci.row_indices)
end

function avail_cols(ci::matrix_cross_interpolation{T}) where T
    return setdiff(1:n_cols(ci), ci.col_indices)
end

function rank(ci::matrix_cross_interpolation{T}) where T
    return length(ci.row_indices)
end

function isempty(ci::matrix_cross_interpolation{T}) where T
    return Base.isempty(ci.pivot_cols)
end

function first_pivot_value(ci::matrix_cross_interpolation{T}) where T
    return isempty(ci) ? 1.0 : ci.pivot_cols[ci.row_indices[1], 1]
end

function eval(ci::matrix_cross_interpolation{T}, i::Int, j::Int) where T
    if isempty(ci)
        return T(0)
    else
        return dot(left_matrix(ci)[i, :], ci.pivot_rows[:, j])
    end
end

function Base.getindex(
    ci::matrix_cross_interpolation{T}, i::Int, j::Int) where T
    return eval(ci, i, j)
end

function submatrix(
    ci::matrix_cross_interpolation{T},
    rows::Union{AbstractVector{Int}, Colon},
    cols::Union{AbstractVector{Int}, Colon}) where T
    if isempty(ci)
        return zeros(T, length(rows), length(cols))
    else
        return left_matrix(ci)[rows, :] * ci.pivot_rows[:, cols]
    end
end

function row(
    ci::matrix_cross_interpolation{T},
    i::Int;
    cols::Union{AbstractVector{Int}, Colon}=Colon()) where T
    return submatrix(ci, [i], cols)[:]
end

function col(
    ci::matrix_cross_interpolation{T},
    j::Int;
    rows::Union{AbstractVector{Int}, Colon}=Colon()) where T
    return submatrix(ci, rows, [j])[:]
end

function Base.getindex(
    ci::matrix_cross_interpolation{T},
    rows::AbstractVector{Int},
    cols::AbstractVector{Int}) where T
    return submatrix(ci, rows, cols)
end

function Base.getindex(
    ci::matrix_cross_interpolation{T},
    i::Int,
    cols::AbstractVector{Int}) where T
    return row(ci, i; cols=cols)
end

function Base.getindex(
    ci::matrix_cross_interpolation{T},
    rows::AbstractVector{Int},
    j::Int) where T
    return col(ci, j; rows=rows)
end

function matrix(ci::matrix_cross_interpolation{T}) where T
    return left_matrix(ci) * ci.pivot_rows
end
