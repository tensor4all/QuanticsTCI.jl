module QuanticsTCI

using StaticArrays
#using TensorCrossInterpolation
#using ITensors

# To add methods to rank
#import LinearAlgebra: rank

#export QuanticsFunction, QuanticsFunctionFused, QuanticsFunctionInterleaved
#export UnfoldingSchemes
#export quantics_to_index, index_to_quantics
#export quantics_to_index_fused, index_to_quantics_fused
#export quantics_to_index_interleaved, index_to_quantics_interleaved
#export split_dimensions, merge_dimensions, fuse_dimensions,
    #interleave_dimensions, deinterleave_dimensions
#export evaluate_mps
#export quanticscrossinterpolate

include("quantics.jl")
include("grid.jl")

end
