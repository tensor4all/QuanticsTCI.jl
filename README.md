# QuanticsTCI

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://tensor4all.github.io/QuanticsTCI.jl/dev)
[![CI](https://github.com/tensor4all/QuanticsTCI.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/tensor4all/QuanticsTCI.jl/actions/workflows/CI.yml)

This module contains utilities for interpolations of functions in the quantics TCI / quantics tensor train (QTT) format. It is a small wrapper around [TensorCrossInterpolation.jl](https://github.com/tensor4all/TensorCrossInterpolation.jl) and [QuanticsGrids.jl](https://github.com/tensor4all/QuanticsGrids.jl) with more convenient functionality intended to cover the most common use cases. For more advanced or unusual use cases, it is likely that you will need to rely on those two libraries directly.

## Installation

This module has been registered in the General registry. It can be installed by typing the following in a Julia REPL:
```julia
using Pkg; Pkg.add("QuanticsTCI.jl")
```

This module depends on:
- [TensorCrossInterpolation.jl](https://github.com/tensor4all/TensorCrossInterpolation.jl)
- [QuanticsGrids.jl](https://github.com/tensor4all/QuanticsGrids.jl)

## Usage

*This section only contains the bare minimum to get you started. More examples, including more advanced use cases, can be found in the [T4F examples repository](https://tensors4fields.gitlab.io/T4FExamples.jl/dev/index.html). For a documentation of the API, see the [package documentation](https://tensors4fields.gitlab.io/quanticstci.jl/dev/index.html).*

The easiest way to construct a quantics tensor train is the `quanticscrossinterpolate` function. For example, the function `f(x, y) = (cos(x) - cos(x - 2y)) * abs(x + y)` can be interpolated as follows.

```julia
using QuanticsTCI
f(x, y) = (cos(x) - cos(x - 2y)) * abs(x + y)
xvals = range(-6, 6; length=256)
yvals = range(-12, 12; length=256)
qtt, ranks, errors = quanticscrossinterpolate(Float64, f, [xvals, yvals]; tolerance=1e-8)
```
The output object `qtt` now represents a quantics tensor train. It can then be evaluated a function of indices enumerating the `xvals` and `yvals` arrays:
```julia
@show qttvalue = qtt(212, 92)
@show truevalue = f(xvals[212], yvals[92])
@show error = abs(qttvalue - truevalue)
```
Output:
```
qttvalue = qtt(212, 92) = -0.2525252152789011
truevalue = f(xvals[212], yvals[92]) = -0.2525252152789314
error = abs(qttvalue - truevalue) = 3.0309088572266774e-14
```
The output shows that the approximation has an error of only `3 * 10^-14` at `[212, 92]`.

This example is continued in the [package documentation](https://tensors4fields.gitlab.io/quanticstci.jl/dev/index.html), and more examples can be found in the [T4F examples repository](https://tensors4fields.gitlab.io/T4FExamples.jl/dev/index.html).

## Related libraries
- [TensorCrossInterpolation.jl](https://gitlab.com/quanticstci/tensorcrossinterpolation.jl) to calculate tensor cross interpolations.
- [QuanticsGrids.jl](https://github.com/tensor4all/QuanticsGrids.jl) for conversion between quantics and direct representations. More advanced use cases can be implemented directly using this library.
- [ITensors.jl](https://github.com/ITensor/ITensors.jl) for MPS / MPO algorithms.

## References
- M. K. Ritter, Y. N. Fernández, M. Wallerberger, J. von Delft, H. Shinaoka, and X. Waintal, *Quantics Tensor Cross Interpolation for High-Resolution, Parsimonious Representations of Multivariate Functions in Physics and Beyond*, [arXiv:2303.11819](http://arxiv.org/abs/2303.11819).
