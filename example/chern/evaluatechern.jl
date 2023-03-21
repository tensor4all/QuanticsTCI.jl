using PyPlot
using JLD2

include("chern.jl")

nq = parse(Int, ARGS[1])

result = chern.testberry(4, 0.2, nq)[1]

jldsave("chern_results/nq$nq.jld2";
    nq=result.nq,
    chernnumber=result.chernnumber,
    qtt=result.qtt,
    ranks=result.ranks,
    errors=result.errors,
    nevals=result.nevals,
    timeestimate=result.timeestimate,
    chernnumbermps=result.chernnumbermps,
    mps=result.mps,
    mpsranks=result.mpsranks,
    mpsnevals=result.mpsnevals,
    mpstimeestimate=result.mpstimeestimate)
