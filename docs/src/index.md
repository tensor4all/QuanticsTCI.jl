# QuanticsTCI.jl module documentation

```@meta
CurrentModule = QuanticsTCI
```

```@contents
```

## `xfac` library glue via python

Calls the `xfac` library via python bindings.
!!! warning
    The installation used by `PyCall`
    must be able to find `xfacpy`. You probably have to add the path to `xfac` to
    `$PYTHONPATH` before loading this module, like this:
    ```julia
    ENV["PYTHONPATH"] = "/somepath/xfac/python/"
    using QuanticsTCI
    ```
    For instructions on how to set up python, `PyCall` and `xfac` correctly, see
    [Setup of `xfacpy`](@ref).

```@docs
QuanticsTCI.qtt
QuanticsTCI.qtt_to_mps
```
