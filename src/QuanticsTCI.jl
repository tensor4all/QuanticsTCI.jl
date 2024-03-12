module QuanticsTCI

using TensorCrossInterpolation
import TensorCrossInterpolation as TCI

# To add methods to rank
import LinearAlgebra: rank

import QuanticsGrids as QG
import QuanticsGrids: UnfoldingSchemes

export evaluate
export quanticscrossinterpolate
export UnfoldingSchemes
export cachedata

include("tciinterface.jl")

end
