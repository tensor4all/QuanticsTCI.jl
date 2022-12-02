module QuanticsTCI

using LinearAlgebra
using ITensors
using PyCall

export qtt
export qtt_to_mps
export quantics_to_index, index_to_quantics
export split_dimensions, merge_dimensions,
    interleave_dimensions, deinterleave_dimensions
export binary_addition_mpo, binary_subtraction_mpo, kroneckerdelta_mpo
export evaluate_mps
export MatrixCrossInterpolation, n_rows, n_cols, size,
    pivot_matrix, left_matrix, right_matrix, rank, matrix,
    local_error, add_pivot!

include("quantics.jl")
include("binaryops.jl")
include("propagators.jl")
include("xfac.jl")
include("mps_util.jl")
include("MatrixCI.jl")

end
