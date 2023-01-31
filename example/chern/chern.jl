module chern

using LinearAlgebra
using PyPlot
import TensorCrossInterpolation as TCI
using QuanticsTCI
using BenchmarkTools
using ITensors

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

function sum_quantics_mps(mps)
    m = mps[1] * ITensor(1, siteind(mps, 1))
    for i in 2:length(mps)
        m *= mps[i] * ITensor(1, siteind(mps, i))
    end
    return scalar(m)
end

function mps_to_array(mps)
    result = Vector{Array{Float64, 3}}()
    T1 = Array(mps[1], siteind(mps, 1), linkind(mps, 1))
    push!(result, reshape(T1, 1, size(T1)...))
    for i in 2:length(mps)-1
        push!(result, Array(mps[i], linkind(mps, i-1), siteind(mps, i), linkind(mps, i)))
    end
    Tlast = Array(mps[end], linkind(mps, length(mps)-1), siteind(mps, length(mps)))
    push!(result, reshape(Tlast, size(Tlast)..., 1))
    return result
end

mutable struct functioncounter
    f::Function
    n::Int
end

Base.broadcastable(m::functioncounter) = Ref(m)

function functioncounter(f::Function)
    return functioncounter(f, 0)
end

function (f::functioncounter)(k)
    f.n += 1
    return f.f(k)
end

function getberryqtt_dets(
    nquantics::Integer,
    kxvals::Vector{Float64},
    kyvals::Vector{Float64},
    n::Integer,
    lattice::TCI.IndexSet{HoneycombSite},
    q::Integer,
    lambdaSO::Float64;
    spinindex=1,
    tolerance=1e-12
)
    Hcached = TCI.CachedFunction{Matrix{ComplexF64}}(
        kindex -> get_H(
            q, lambdaSO, [kxvals[kindex[1]], kyvals[kindex[2]]], lattice
        )[:, spinindex, :, spinindex],
        [2^nquantics, 2^nquantics]
    )

    f = functioncounter(k -> berrycurvature_quantics_dets(Hcached, n, k, nquantics))
    firstpivot = TCI.optfirstpivot(f, fill(2, 2 * nquantics))
    f.n = 0
    tci, ranks, errors = TCI.crossinterpolate(
        Float64,
        f,
        fill(2, 2 * nquantics),
        firstpivot,
        tolerance=tolerance,
        maxiter=200,
        verbosity=1,
    )
    qtt = TCI.tensortrain(tci)
    return sumqtt(qtt) / 2pi, qtt, ranks, errors, f.n
end


function getberrymps_dets(
    nquantics::Integer,
    kxvals::Vector{Float64},
    kyvals::Vector{Float64},
    n::Integer,
    lattice::TCI.IndexSet{HoneycombSite},
    q::Integer,
    lambdaSO::Float64;
    spinindex=1,
    tolerance=1e-12
)
    H = [
        get_H(q, lambdaSO, [kx, ky], lattice)[:, spinindex, :, spinindex]
        for kx in kxvals, ky in kyvals
    ]
    quanticsindices = [
        Index(2, i % 2 == 0 ? "qx$(div(i, 2))" : "qy$(div(i, 2))") for i in 1:2nquantics
    ]
    A = ITensor(berrycurvature_dets(H, n), quanticsindices)
    mps = MPS(A, quanticsindices, cutoff=tolerance, maxdim=200)
    return sum_quantics_mps(mps) / 2pi, mps_to_array(mps), linkdims(mps), prod(size(A))
end

function getberryqtt_derivs(
    nquantics::Integer,
    kxvals::Vector{Float64},
    kyvals::Vector{Float64},
    n::Integer,
    lattice::TCI.IndexSet{HoneycombSite},
    q::Integer,
    lambdaSO::Float64;
    spinindex=1,
    tolerance=1e-12
)
    Hfunc(kindex) = get_H(q, lambdaSO, [kxvals[kindex[1]], kyvals[kindex[2]]], lattice)[:, spinindex, :, spinindex]
    Hderivfunc(kindex, derivdirection) = get_H(q, lambdaSO, [kxvals[kindex[1]], kyvals[kindex[2]]], lattice, derivative_direction=derivdirection)[:, spinindex, :, spinindex]
    f = functioncounter(k -> berrycurvature_quantics_derivatives(Hfunc, Hderivfunc, n, k))
    firstpivot = TCI.optfirstpivot(f, fill(2, 2 * nquantics))
    f.n = 0
    tci, ranks, errors = TCI.crossinterpolate(
        Float64,
        f,
        fill(2, 2 * nquantics),
        firstpivot,
        tolerance=tolerance,
        maxiter=200,
        verbosity=1,
    )
    qtt = TCI.tensortrain(tci)
    return sumqtt(qtt) / 2pi, qtt, ranks, errors, f.n
end

struct BerryResult
    nq::Int
    chernnumber::Float64
    qtt::Vector{Array{Float64,3}}
    ranks::Vector{Int}
    errors::Vector{Float64}
    nevals::Int
    timeestimate::Float64
    chernnumbermps::Float64
    mps::Vector{Array{Float64,3}}
    mpsranks::Vector{Int}
    mpsnevals::Int
    mpstimeestimate::Float64
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
        timeestimate = @elapsed result = getberryqtt_dets(nq, kxvals, kyvals, 2q, lattice, q, lambdaSO, tolerance=1e-5)
        mpstime = @elapsed mpsresult = getberrymps_dets(nq, kxvals, kyvals, 2q, lattice, q, lambdaSO, tolerance=1e-5)
        push!(
            results,
            BerryResult(
                nq,
                result...,
                timeestimate,
                mpsresult...,
                mpstime
            ))
        println(
            "Finished nq = $nq.
            Chern number from qtt is $(last(results).chernnumber).
            Time elapsed is $timeestimate for $(last(results).nevals) function evaluations.
            Chern number from mps is $(last(results).chernnumbermps).
            Time elapsed is $mpstime for $(last(results).mpsnevals) function evaluations.")
    end
    return results
end

end
