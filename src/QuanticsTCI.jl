module QuanticsTCI

using TensorCrossInterpolation
using ITensors

# To add methods to rank
import LinearAlgebra: rank

export qtt
export qtt_to_mps
export quantics_to_index, index_to_quantics
export quantics_to_index_fused, index_to_quantics_fused
export quantics_to_index_interleaved, index_to_quantics_interleaved
export split_dimensions, merge_dimensions, fuse_dimensions,
    interleave_dimensions, deinterleave_dimensions
export binary_addition_mpo, binary_subtraction_mpo, kroneckerdelta_mpo
export evaluate_mps
export QuanticsFunction, QuanticsFunctionFused, QuanticsFunctionInterleaved

include("quantics.jl")
include("binaryops.jl")
include("propagators.jl")
include("xfac.jl")
include("mps_util.jl")

end
