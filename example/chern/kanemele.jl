function peierls(B::Real, i::HoneycombSite, j::HoneycombSite)
    ri = realspacecoordinates(i)
    rj = realspacecoordinates(j)
    phase = -B * (ri[1] - rj[1]) * (ri[2] + rj[2]) / 2
    return mod(phase, 2pi)
    # return 0.0
end

function rashba(distance::Vector{Float64})
    pauli = [
        [0. 1.; 1. 0.],
        [0. -1.0im; 1.0im 0.],
        [1. 0.; 0. -1.]
    ]

    return pauli[1] .* distance[2] .- pauli[2] .* distance[1]
end

function get_Ht(
    q::Integer,
    k::Vector{Float64},
    lattice::TCI.IndexSet{HoneycombSite};
    mass::Float64=0.0,
    derivative_direction::Union{Nothing,Int}=nothing
)
    B = 4pi / sqrt(3) / q
    Ht = zeros(ComplexF64, 4q, 2, 4q, 2)

    for (i, s) in enumerate(lattice.fromint)
        # Alternating "mass term"
        Ht[i, 1, i, 1] += s.c == 1 ? mass : -mass

        # Hopping
        for n in neighbours(s)
            rs, rn = realspacecoordinates.([s, n])
            phase = -k' * (rn - rs) + peierls(B, s, n)
            j = TCI.pos(lattice, reducesite(n, 1, q))
            result = -exp(1im * phase)
            if !isnothing(derivative_direction)
                result *= rs[derivative_direction] - rn[derivative_direction]
            end

            Ht[i, 1, j, 1] += result
        end
    end
    Ht[:, 2, :, 2] = Ht[:, 1, :, 1]
    return Ht
end

function get_HR(
    q::Integer,
    k::Vector{Float64},
    lattice::TCI.IndexSet{HoneycombSite};
    derivative_direction::Union{Nothing,Int}=nothing
)
    @assert isnothing(derivative_direction)

    B = 4pi / sqrt(3) / q
    HR = zeros(ComplexF64, 4q, 2, 4q, 2)

    for (i, s) in enumerate(lattice.fromint)
        for n in neighbours(s)
            rs, rn = realspacecoordinates.([s, n])
            j = TCI.pos(lattice, reducesite(n, 1, q))
            HR[i, :, j, :] = 1im * exp(1im * peierls(B, s, n)) * rashba(rn - rs)
        end
    end

    return HR
end

antisymmetricproduct(u, v) = u[1] * v[2] - u[2] * v[1]

function get_Hlambda(
    q::Integer,
    k::Vector{Float64},
    lattice::TCI.IndexSet{HoneycombSite};
    derivative_direction::Union{Nothing,Int}=nothing
)
    B = 4pi / sqrt(3) / q
    Hlambda = zeros(ComplexF64, 4q, 2, 4q, 2)

    # "Spin-orbit coupling" term
    for (i, s) in enumerate(lattice.fromint)
        for n in neighbours(s)
            for nn in neighbours(n)
                j = TCI.pos(lattice, reducesite(nn, 1, q))
                rs, rn, rnn = realspacecoordinates.([s, n, nn])
                phase = -(k' * (rnn - rs)) + peierls(B, s, nn)
                nu = sign(antisymmetricproduct(rn - rs, rnn - rn))
                result = nu * exp(1im * phase)
                if !isnothing(derivative_direction)
                    result *= rs[derivative_direction] - rnn[derivative_direction]
                end
                Hlambda[i, 1, j, 1] += 1im * result
            end
        end
    end
    Hlambda[:, 2, :, 2] = -Hlambda[:, 1, :, 1]
    return Hlambda
end

function get_H(
    q::Integer,
    lambda_SO::Float64,
    lambda_R::Float64,
    k::Vector{Float64},
    lattice::TCI.IndexSet{HoneycombSite};
    mass::Float64=0.0,
    derivative_direction::Union{Nothing,Int}=nothing
)
    return get_Ht(q, k, lattice; mass, derivative_direction) .+
        lambda_SO .* get_Hlambda(q, k, lattice; derivative_direction) .+
        lambda_R .* get_HR(q, k, lattice)
end
