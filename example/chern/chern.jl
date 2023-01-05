module chern

using LinearAlgebra
using PyPlot
import TensorCrossInterpolation as TCI
using QuanticsTCI

include("honeycomb.jl")
include("kanemele.jl")
include("latticeplot.jl")
include("berry.jl")

function maxlinkdim(n::Integer, localdim::Integer=2)
    return 0:n-2, [min(localdim^i, localdim^(n - i)) for i in 1:(n-1)]
end

function sumqtt(qtt)
    return prod(sum(T, dims=2)[:, 1, :] for T in qtt)[1]
end

function getberryqtt(
    nquantics::Integer,
    kxvals::Vector{Float64},
    kyvals::Vector{Float64},
    n::Integer,
    lattice::TCI.IndexSet{HoneycombSite},
    q::Integer,
    lambdaSO::Float64,
    spinindex=1,
    tolerance=1e-12
)
    Hcached = TCI.CachedFunction{Array{Int},Matrix{ComplexF64}}(
        kindex -> get_H(
            q, lambdaSO, [kxvals[kindex[1]], kyvals[kindex[2]]], lattice
        )[:, spinindex, :, spinindex]
    )

    f(k) = berrycurvature_quantics(Hcached, n, k, nquantics)
    firstpivot = TCI.optfirstpivot(f, fill(4, nquantics))
    tci, ranks, errors = TCI.crossinterpolate(
        f,
        fill(4, nquantics),
        firstpivot,
        tolerance=tolerance,
        maxiter=200,
    )
    qtt = TCI.tensortrain(tci)
    return sumqtt(qtt) / 2pi, qtt, ranks, errors
end

struct BerryResult
    nq::Int
    chernnumber::Float64
    qtt::Vector{Array{Float64,3}}
    ranks::Vector{Int}
    errors::Vector{Float64}
end

function testberry(q::Integer, lambdaSO::Float64, nquantics=5:10)
    lattice = honeycomblattice(0, 1, 0, q - 1)

    BZedgex = pi / 1.5
    BZedgey = pi / sqrt(3)
    results = BerryResult[]
    for nq in nquantics
        ndiscretization = 2^nq
        kxvals = collect(range(-BZedgex, BZedgex; length=ndiscretization)) .+ (BZedgex / ndiscretization)
        kyvals = collect(range(-BZedgey, BZedgey; length=ndiscretization)) .+ (BZedgey / ndiscretization)
        push!(
            results,
            BerryResult(
                nq,
                getberryqtt(nq, kxvals, kyvals, 2q, lattice, q, lambdaSO, tolerance=1e-8)...
            ))
        println("Finished nq = $nq. Chern number is $(last(results).chernnumber)")
    end
    return results
end

end
