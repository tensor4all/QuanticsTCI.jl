function peierls(B::Real, i::HoneycombSite, j::HoneycombSite)
    ri = realspacecoordinates(i)
    rj = realspacecoordinates(j)
    phase = -B * (ri[1] - rj[1]) * (ri[2] + rj[2]) / 2
    return mod(phase, 2pi)
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
    #Ht[:, 1, :, 1] += diagm(fill(mass, 4q))

    for (i, s) in enumerate(lattice.fromint)
        Ht[i, 1, i, 1] += s.c == 1 ? mass : -mass

        for n in neighbours(s)
            rs, rn = realspacecoordinates.([s, n])
            phase = -k' * (rn - rs) + peierls(B, s, n)
            j = TCI.pos(lattice, reducesite(n, 1, q))
            result = -exp(1im * phase)
            if !isnothing(derivative_direction)
                result *= rs[derivative_direction] - rn[derivative_direction]
            end
            # result += mass * exp(-1im * k' * (rn - rs))

            Ht[i, 1, j, 1] += result
        end
    end
    Ht[:, 2, :, 2] = Ht[:, 1, :, 1]
    return Ht
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
    lambda::Float64,
    k::Vector{Float64},
    lattice::TCI.IndexSet{HoneycombSite};
    mass::Float64=0.0,
    derivative_direction::Union{Nothing,Int}=nothing
)
    return get_Ht(q, k, lattice; mass, derivative_direction) .+
        lambda .* get_Hlambda(q, k, lattice; derivative_direction)
end
