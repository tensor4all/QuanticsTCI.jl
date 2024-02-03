module QuanticsTCI

using TensorCrossInterpolation

# To add methods to rank
import LinearAlgebra: rank

import QuanticsGrids as QG
import QuanticsGrids: UnfoldingSchemes

export evaluate_mps
export quanticscrossinterpolate

include("tciinterface.jl")

end
