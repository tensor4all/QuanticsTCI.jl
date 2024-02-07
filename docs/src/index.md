# QuanticsTCI.jl user guide

```@meta
CurrentModule = QuanticsTCI
```

This module allows easy translation of functions to quantics representation. It meshes well with the `TensorCrossInterpolation.jl` module, together with which it provides quantics TCI functionality.

# Quickstart

The easiest way to construct a quantics tensor train is the `quanticscrossinterpolate` function. For example, the function ``f(x, y) = \exp(-x - 2y)`` can be interpolated as follows.

```@example simple
using QuanticsTCI
f(x, y) = (cos(x) - cos(x - 2y)) * abs(x + y)
xvals = range(-6, 6; length=256)
yvals = range(-12, 12; length=256)
qtt, ranks, errors = quanticscrossinterpolate(Float64, f, [xvals, yvals]; tolerance=1e-8)
```

The QTT can then be evaluated a function of indices enumerating the `xvals` and `yvals` arrays.

```@example simple
using Plots
qttvals = qtt.(1:256, collect(1:256)')
contour(xvals, yvals, qttvals, fill=true)
xlabel!("x")
ylabel!("y")
savefig("simple.svg"); nothing # hide
```

![](simple.svg)

# Quantics representation
<!---
Refer to [QuanticsGrids.jl](https://gitlab.com/tensors4fields/QuanticsGrids.jl).
--->