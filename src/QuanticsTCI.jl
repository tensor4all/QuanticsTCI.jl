module QuanticsTCI

using TensorCrossInterpolation
using ITensors

# To add methods to rank
import LinearAlgebra: rank

export qtt
export quantics_to_index, index_to_quantics
export split_dimensions, merge_dimensions,
    interleave_dimensions, deinterleave_dimensions
export binary_addition_mpo, binary_subtraction_mpo, kroneckerdelta_mpo

include("quantics.jl")
include("binaryops.jl")
include("propagators.jl")

end
