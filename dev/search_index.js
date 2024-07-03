var documenterSearchIndex = {"docs":
[{"location":"apireference/#Documentation","page":"API Reference","title":"Documentation","text":"","category":"section"},{"location":"apireference/","page":"API Reference","title":"API Reference","text":"Modules = [QuanticsTCI]","category":"page"},{"location":"apireference/#QuanticsTCI.quanticscrossinterpolate-Union{Tuple{Array{ValueType, d}}, Tuple{d}, Tuple{ValueType}, Tuple{Array{ValueType, d}, AbstractVector{<:AbstractVector}}} where {ValueType, d}","page":"API Reference","title":"QuanticsTCI.quanticscrossinterpolate","text":"function quanticscrossinterpolate(\n    ::Type{ValueType},\n    f,\n    size::NTuple{d,Int},\n    initialpivots::AbstractVector{<:AbstractVector}=[ones(Int, d)];\n    unfoldingscheme::Symbol=:interleaved,\n    kwargs...\n) where {ValueType,d}\n\nInterpolate a Tensor F as a quantics tensor train. For an explanation of arguments, etc., see the documentation of the main overload.\n\n\n\n\n\n","category":"method"},{"location":"apireference/#QuanticsTCI.quanticscrossinterpolate-Union{Tuple{ValueType}, Tuple{Type{ValueType}, Any, AbstractVector{<:AbstractVector}}, Tuple{Type{ValueType}, Any, AbstractVector{<:AbstractVector}, Union{Nothing, AbstractVector{<:AbstractVector}}}} where ValueType","page":"API Reference","title":"QuanticsTCI.quanticscrossinterpolate","text":"function quanticscrossinterpolate(\n    ::Type{ValueType},\n    f,\n    xvals::AbstractVector{<:AbstractVector},\n    initialpivots::Union{Nothing,AbstractVector{<:AbstractVector}}=nothing;\n    unfoldingscheme::Symbol=:interleaved,\n    nrandominitpivot=5,\n    kwargs...\n) where {ValueType}\n\nInterpolate a function f(mathbfx) as a quantics tensor train. This overload automatically constructs a Grid object from the mathbfx points given in xvals.\n\nArguments:\n\nxvals::AbstractVector{<:AbstractVector}: A set of discrete points where f can be evaluated, given as a set of arrays, where xvals[i] describes the ith axis. Each array in xvals should contain 2^R points for some integer R.\nFor all other arguments, see the documentation of the main overload.\n\n\n\n\n\n","category":"method"},{"location":"apireference/#QuanticsTCI.quanticscrossinterpolate-Union{Tuple{ValueType}, Tuple{Type{ValueType}, Any, AbstractVector}, Tuple{Type{ValueType}, Any, AbstractVector, AbstractVector}} where ValueType","page":"API Reference","title":"QuanticsTCI.quanticscrossinterpolate","text":"function quanticscrossinterpolate(\n    ::Type{ValueType},\n    f,\n    xvals::AbstractVector,\n    initialpivots::AbstractVector=[1];\n    kwargs...\n) where {ValueType}\n\nInterpolate a function f(x) as a quantics tensor train. This is an overload for 1d functions. For an explanation of arguments and return type, see the documentation of the main overload.\n\n\n\n\n\n","category":"method"},{"location":"apireference/#QuanticsTCI.quanticscrossinterpolate-Union{Tuple{d}, Tuple{ValueType}, Tuple{Type{ValueType}, Any, Tuple{Vararg{Int64, d}}}, Tuple{Type{ValueType}, Any, Tuple{Vararg{Int64, d}}, AbstractVector{<:AbstractVector}}} where {ValueType, d}","page":"API Reference","title":"QuanticsTCI.quanticscrossinterpolate","text":"function quanticscrossinterpolate(\n    ::Type{ValueType},\n    f,\n    size::NTuple{d,Int},\n    initialpivots::AbstractVector{<:AbstractVector}=[ones(Int, d)];\n    unfoldingscheme::Symbol=:interleaved,\n    kwargs...\n) where {ValueType,d}\n\nInterpolate a function f(mathbfx) as a quantics tensor train. This overload automatically constructs a Grid object using the information contained in size. Here, the ith argument runs from 1 to size[i].\n\n\n\n\n\n","category":"method"},{"location":"apireference/#QuanticsTCI.quanticscrossinterpolate-Union{Tuple{n}, Tuple{ValueType}, Tuple{Type{ValueType}, Any, QuanticsGrids.Grid{n}}, Tuple{Type{ValueType}, Any, QuanticsGrids.Grid{n}, Union{Nothing, AbstractVector{<:AbstractVector}}}} where {ValueType, n}","page":"API Reference","title":"QuanticsTCI.quanticscrossinterpolate","text":"function quanticscrossinterpolate(\n    ::Type{ValueType},\n    f,\n    grid::QuanticsGrids.Grid{n},\n    initialpivots::Union{Nothing,AbstractVector{<:AbstractVector}}=nothing;\n    nrandominitpivot=5,\n    kwargs...\n) where {ValueType}\n\nInterpolate a function f(mathbfx) as a quantics tensor train. The tensor train itself is constructed using the 2-site tensor cross interpolation algorithm implemented in TensorCrossInterpolation.crossinterpolate2.\n\nArguments:\n\nValueType is the return type of f. Automatic inference is too error-prone.\nf is the function to be interpolated. f may take multiple arguments. The return type should be ValueType.\ngrid is a Grid{n} object from QuanticsGrids.jl that describes a d-dimensional grid of discrete points indexed by binary digits. To avoid constructing a grid explicitly, use one of the other overloads.\ninitialpivots is a vector of pivots to be used for initialization.\nnrandominitpivot determines how many random pivots should be used for initialization if no initial pivot is given.\n\nAll other arguments are forwareded to crossinterpolate2. Most importantly:\n\ntolerance::Float64 is a float specifying the target tolerance for the interpolation. Default: 1e-8.\npivottolerance::Float64 is a float that specifies the tolerance for adding new pivots, i.e. the truncation of tensor train bonds. It should be <= tolerance, otherwise convergence may be impossible. Default: tolerance.\nmaxbonddim::Int specifies the maximum bond dimension for the TCI. Default: typemax(Int), i.e. effectively unlimited.\nmaxiter::Int is the maximum number of iterations (i.e. optimization sweeps) before aborting the TCI construction. Default: 200.\n\nFor all other arguments, see the documentation for TensorCrossInterpolation.crossinterpolate2.\n\n\n\n\n\n","category":"method"},{"location":"apireference/#QuanticsTCI.quanticsfouriermpo-Tuple{Int64}","page":"API Reference","title":"QuanticsTCI.quanticsfouriermpo","text":"function quanticsfouriermpo(\n    R::Int;\n    sign::Float64=-1.0,\n    maxbonddim::Int=12,\n    tolerance::Float64=1e-14,\n    K::Int=25,\n    method::Symbol=:SVD,\n    normalize::Bool=true\n)::TCI.TensorTrain{ComplexF64}\n\nGenerate a quantics Fourier transform operator in tensor train form. When contracted with a quantics tensor train F_boldsymbolsigma representing a function, the result will be the fourier transform of the function in quantics tensor train form, tildeF_boldsymbolsigma = sum_boldsymbolsigma F_boldsymbolsigma exp(-2pi i (k_boldsymbolsigma-1) (m_boldsymbolsigma - 1)M), where k_boldsymbolsigma = sum_ell=1^R 2^R-ell sigma_ell, m_boldsymbolsigma = sum_ell=1^R 2^R-ell sigma_ell, and M=2^R.\n\nnote: Index ordering\nBefore the Fourier transform, the left most index corresponds to sigma_1, which describes the largest length scale, and the right most index corresponds to sigma_R, which describes the smallest length scale. The indices sigma_1 ldots sigma_R in the fourier transformed QTT are aligned in the inverse order; that is,  the left most index corresponds to sigma_R, which describes the smallest length scale. This allows construction of an operator with small bond dimension (see reference 1). If necessary, a call to TCI.reverse(tt) can restore large-to-small index ordering.\n\nThe Fourier transform operator is implemented using a direct analytic construction of the tensor train by Chen and Lindsey (see reference 2). The tensor train thus obtained is then re-compressed to the user-given bond dimension and tolerance.\n\nArguments:\n\nR: number of bits of the fourier transform.\nsign: sign in the exponent exp(2ipi times mathrmsign times (k_boldsymbolsigma-1) (x_boldsymbolsigma-1)M), usually pm 1.\nmaxbonddim: bond dimension to compress the operator to. From observations, maxbonddim = 12 is generally big enough to reach an accuracy of 1e-12.\ntolerance: tolerance of the TT compression. Note that the error in the fourier transform is generally a bit larger than this error tolerance.\nK: bond dimension of the TT before compression, i.e. number of basis functions to approximate the Fourier transform with (see reference 2). The TT will become inaccurate for K < 22; higher values may be necessary for very high precision.\nmethod: method with which to compress the TT. Choose between :SVD and :CI.\nnormalize: whether or not to normalize the operator as an isometry.\n\ndetails: References\nJ. Chen, E. M. Stoudenmire, and S. R. White, Quantum Fourier Transform Has Small Entanglement, PRX Quantum 4, 040318 (2023).\nJ. Chen and M. Lindsey, Direct Interpolative Construction of the Discrete Fourier Transform as a Matrix Product Operator, arXiv:2404.03182.\n\n\n\n\n\n","category":"method"},{"location":"#QuanticsTCI.jl-user-guide","page":"Home","title":"QuanticsTCI.jl user guide","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = QuanticsTCI","category":"page"},{"location":"","page":"Home","title":"Home","text":"This module allows easy translation of functions to quantics representation. It meshes well with the TensorCrossInterpolation.jl module, together with which it provides quantics TCI functionality.","category":"page"},{"location":"#Quickstart","page":"Home","title":"Quickstart","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The easiest way to construct a quantics tensor train is the quanticscrossinterpolate function. For example, the function f(x y) = (cos(x) - cos(x - 2y)) * abs(x + y) can be interpolated as follows.","category":"page"},{"location":"","page":"Home","title":"Home","text":"using QuanticsTCI\nf(x, y) = (cos(x) - cos(x - 2y)) * abs(x + y)\nxvals = range(-6, 6; length=256)\nyvals = range(-12, 12; length=256)\nqtt, ranks, errors = quanticscrossinterpolate(Float64, f, [xvals, yvals]; tolerance=1e-8)","category":"page"},{"location":"","page":"Home","title":"Home","text":"The output object qtt now represents a quantics tensor train. It can then be evaluated a function of indices enumerating the xvals and yvals arrays.","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Plots\nqttvals = qtt.(1:256, collect(1:256)')\ncontour(xvals, yvals, qttvals, fill=true)\nxlabel!(\"x\")\nylabel!(\"y\")\nsavefig(\"simple.svg\"); nothing # hide","category":"page"},{"location":"","page":"Home","title":"Home","text":"(Image: )","category":"page"},{"location":"","page":"Home","title":"Home","text":"The convergence criterion can be controlled using the keywords tolerance, pivottolerance, and maxbonddim.","category":"page"},{"location":"","page":"Home","title":"Home","text":"tolerance is the value of the error estimate at which the optimization algorithm will stop.\npivottolerance is the threshold at which each local optimization will truncate the bond.\nmaxbonddim sets the maximum bond dimension along the links.","category":"page"},{"location":"","page":"Home","title":"Home","text":"A common default setting is to control convergence using tolerance, and to set pivottolerance equal or slightly smaller than that. Specifying maxbonddim can be useful as a safety. However, if maxbonddim is set, one should check the error estimate for convergence afterwards.","category":"page"},{"location":"","page":"Home","title":"Home","text":"In the following example, we specify all 3 parameters, but set maxbonddim too small.","category":"page"},{"location":"","page":"Home","title":"Home","text":"qtt, ranks, errors = quanticscrossinterpolate(\n    Float64, f, [xvals, yvals];\n    tolerance=1e-8,\n    pivottolerance=1e-8,\n    maxbonddim=8)\nprint(last(errors))\nqttvals = qtt.(1:256, collect(1:256)')\ncontour(xvals, yvals, qttvals, fill=true)\nxlabel!(\"x\")\nylabel!(\"y\")\nsavefig(\"simpletrunc.svg\"); nothing # hide","category":"page"},{"location":"","page":"Home","title":"Home","text":"(Image: )","category":"page"},{"location":"","page":"Home","title":"Home","text":"The plot shows obvious noise due to the insufficient maximum bond dimension. Accordingly, the error estimate of 008 shows that convergence has not been reached, and an increase of the maximum bond dimension is necessary.","category":"page"},{"location":"#Further-reading","page":"Home","title":"Further reading","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"See the API Reference for all variants of calling quanticscrossinterpolate.\nIf you are having trouble with convergence / efficiency of the TCI, you might have to tweak some of its options. All keyword arguments are forwarded to TensorCrossInterpolation.crossinterpolate2() internally. See its documentation for further information.\nIf you intend to work directly with the quantics representation, QuanticsGrids.jl is useful for conversion between quantics and direct representations. More advanced use cases can be implemented directly using this library.","category":"page"}]
}
