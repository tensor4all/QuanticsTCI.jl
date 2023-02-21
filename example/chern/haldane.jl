import TensorCrossInterpolation as TCI
using QuanticsTCI
using JLD2

include("chern.jl")

pauli0 = [1. 0.; 0. 1.]
pauli = [
    [0. 1.; 1. 0.],
    [0. -1.0im; 1.0im 0.],
    [1. 0.; 0. -1.]
]

antisymmetricproduct(u, v) = u[1] * v[2] - u[2] * v[1]

function haldane(k, t2, ϕ, m)
    a::Vector{Vector{Float64}} =
    [
        [1, 0],
        [-0.5, 0.5sqrt(3)],
        [-0.5, -0.5sqrt(3)]
    ]
    b::Vector{Vector{Float64}} = [a[2] - a[3], a[3] - a[1], a[1] - a[2]]

    return 2 * t2 * cos(ϕ) * sum(cos(k' * bi) for bi in b) * pauli0 +    # NNN hopping
        sum(cos(k' * ai) * pauli[1] + sin(k' * ai) * pauli[2] for ai in a) + # NN hopping
        (m - 2 * t2 * sin(ϕ) * sum(sin(k' * bi) for bi in b)) * pauli[3]    # staggered offset
end

function sumqtt(qtt)
    return prod(sum(T, dims=2)[:, 1, :] for T in qtt)[1]
end

function crossinterpolate_chern(
    ::Type{ValueType},
    f,
    localdims::Vector{Int},
    firstpivot::TCI.MultiIndex=ones(Int, length(localdims));
    tolerance::Float64=1e-8,
    maxiter::Int=200,
    sweepstrategy::TCI.SweepStrategies.SweepStrategy=TCI.SweepStrategies.back_and_forth,
    pivottolerance::Float64=1e-12,
    errornormalization::Union{Nothing,Float64}=nothing,
    verbosity::Int=0,
    additionalpivots::Vector{TCI.MultiIndex}=TCI.MultiIndex[]
) where {ValueType}
    tci = TCI.TensorCI{ValueType}(f, localdims, firstpivot)
    n = length(tci)
    errors = Float64[]
    cherns = Float64[]
    ranks = Int[]
    N::Float64 = isnothing(errornormalization) ? abs(f(firstpivot)) : abs(errornormalization)

    for pivot in additionalpivots
        println("Adding pivot $pivot")
        TCI.addglobalpivot!(tci, f, pivot, tolerance)
        println("Rank $(TCI.rank(tci))")
    end

    for iter in TCI.rank(tci)+1:maxiter
        foward_sweep = (
            sweepstrategy == TCI.SweepStrategies.forward ||
            (sweepstrategy != TCI.SweepStrategies.backward && isodd(iter))
        )

        if foward_sweep
            TCI.addpivot!.(tci, 1:n-1, f, pivottolerance)
        else
            TCI.addpivot!.(tci, (n-1):-1:1, f, pivottolerance)
        end

        push!(errors, TCI.lastsweeppivoterror(tci) / N)
        push!(ranks, maximum(TCI.rank(tci)))
        push!(cherns, sumqtt(TCI.tensortrain(tci)) / 4 / 2pi)

        if verbosity > 0 && (mod(iter, 10) == 0 || last(errors) < tolerance)
            println(
                "rank = $(last(ranks)), error = $(last(errors)), chern = $(last(cherns))")
        end
        if last(errors) < tolerance
            break
        end
    end

    return tci, ranks, errors, cherns
end

function evaluatechern_haldane(
    deltam::Float64,
    nquantics::Int;
    t2::Float64=1e-1,
    tolerance::Float64=1e-4,
)
    phi = pi/2
    m = 3sqrt(3) * t2 + deltam
    domainboundx = [-4pi/3, 4pi/3]
    domainboundy = [-6pi/(3sqrt(3)), 6pi/(3sqrt(3))]

    ndiscretization = 2^nquantics
    kxvals = range(domainboundx..., length=ndiscretization+1)#[2:end]
    kxvals = 0.5 .* (kxvals[1:ndiscretization] .+ kxvals[2:end])
    kyvals = range(domainboundy..., length=ndiscretization+1)#[2:end]
    kyvals = 0.5 .* (kyvals[1:ndiscretization] .+ kyvals[2:end])

    f(q) = chern.berrycurvature_quantics_dets(
        kindex -> haldane([kxvals[kindex[1]], kyvals[kindex[2]]], t2, phi, m),
        2, q, nquantics)

    localdims = fill(4, nquantics)
    cf = TCI.CachedFunction{Float64}(f, localdims)
    firstpivot = TCI.optfirstpivot(cf, localdims)
        # [2, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 2, 2,
        #     2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1][1:2*nquantics])
    additionalpivots = []
        # TCI.optfirstpivot(cf, localdims,
        # [2, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 2, 2,
        #     2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1][1:2*nquantics]),
        # TCI.optfirstpivot(cf, localdims),
        # TCI.optfirstpivot(cf, localdims, localdims)]
        # TCI.optfirstpivot(cf, localdims, repeat([1, 2], nquantics)),
        # TCI.optfirstpivot(cf, localdims, repeat([2, 1], nquantics))]
    # sort!(additionalpivots, by=abs ∘ cf, rev=true)
    # for a in additionalpivots
    #     println(a, cf(a))
    # end

    walltime = @elapsed tci, ranks, errors, cherns = crossinterpolate_chern(
        Float64,
        cf,
        localdims,
        firstpivot,
        tolerance=tolerance,
        maxiter=200,
        verbosity=1,
        pivottolerance=1e-16,
        #additionalpivots = additionalpivots
    )

    savepath::String="example/chern/haldane_results/nq$(nquantics)_deltam$(deltam).jld2"
    jldsave(
        savepath;
        deltam=deltam,
        nquantics=nquantics,
        cherns=cherns,
        chernnumber=last(cherns),
        tci=tci,
        ranks=ranks,
        errors=errors,
        nevals=length(cf.d),
        walltime=walltime
    )
end
