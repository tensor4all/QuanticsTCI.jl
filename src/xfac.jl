"""
Calls the xfac library via python bindings. The installation used by PyCall must
be able to find xfacpy. You probably have to add the path to xfac to PYTHONPATH
before loading this module, like this:
ENV["PYTHONPATH"] = "/somepath/xfac/python/"
"""

const xfacpy = PyNULL()

function __init__()
    try
        copy!(xfacpy, pyimport("xfacpy"))
    catch e
        if isa(e, PyCall.PyError)
            # For CI, documentation, etc.
            print("Did not find xfac. QTT functions will not be available.")
            print(e)
        else
            rethrow()
        end
    end
end

mutable struct myTf
    f::Function
    c::Int64
    function myTf(f)
        return new(f, 0)
    end
end

function (funcobj::myTf)(x)
    funcobj.c += 1
    return funcobj.f(x)
end

"""
    qtt(f::Function, localdim, firstpivot; cutoff, maxiter)

Get a Tensor Train representation of a quantics function.

 * `f`:             function accepting an array of binary digits
 * `localdim`:      local dimension of the quantics indices, generally 2^d
 * `firstpivot`:    first pivot point
 * `cutoff`:        iteration will be stopped at this precision
 * `maxiter`:       maximum number of iterations
"""
function qtt(
    f::Function, localdim, firstpivot;
    cutoff=1e-12, maxiter=200)

    params = xfacpy.TensorCIParam()
    # Convert to zero-indexing
    params.pivot1 = firstpivot .- 1
    ci = xfacpy.TensorCI(myTf(f), localdim, length(firstpivot), params)

    ranks = zeros(Int, maxiter)
    errors = zeros(maxiter)
    for i in 1:maxiter
        ci.iterate()
        ranks[i] = max(ci.rank()...)
        errors[i] = ci.pivotError()
        if errors[i] < cutoff
            ranks = ranks[1:i]
            errors = errors[1:i]
            break
        elseif i % 10 == 0
            print("$i\t$(ranks[i])\t$(errors[i])\n")
        end
    end

    return ci.get_TensorTrain().core, ranks, errors
end

"""
    qtt_to_mps(qtt, siteindices...)

Convert a quantics tensor train to an ITensor MPS

 * `qtt`            Tensor train as an array of tensors
 * `siteindices`    Arrays of ITensor Index objects
"""
function qtt_to_mps(qtt, siteindices...)
    n = length(qtt)

    if siteindices === nothing
        siteindices = [Index(size(t, 2), "site") for t in qtt]
    end

    qttmps = MPS(n)
    links =
        [
            Index(1, "link")
            [Index(size(t, 3), "link") for t in qtt]
        ]

    qttmps[1] = ITensor(
        qtt[1],
        [s[1] for s in siteindices]...,
        links[2]
    )
    for i in 2:(n-1)
        qttmps[i] = ITensor(
            qtt[i],
            links[i],
            [s[i] for s in siteindices]...,
            links[i+1]
        )
    end
    qttmps[n] = ITensor(
        qtt[n],
        links[n],
        [s[n] for s in siteindices]...
    )

    return qttmps
end
