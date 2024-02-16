module QuanticsTCI

using TensorCrossInterpolation

# To add methods to rank
import LinearAlgebra: rank

import QuanticsGrids as QG
import QuanticsGrids: UnfoldingSchemes

export evaluate
export quanticscrossinterpolate
export UnfoldingSchemes

include("tciinterface.jl")

end
