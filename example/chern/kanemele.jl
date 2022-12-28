function peierls(B::Real, i::HoneycombSite, j::HoneycombSite)
    ri = realspacecoordinates(i)
    rj = realspacecoordinates(j)
    phase = -B * (ri[1] - rj[1]) * (ri[2] + rj[2]) / 2
    return mod(phase, 2pi)
end

function get_Ht(
    q::Integer,
    k::Vector{Float64},
    lattice::TCI.IndexSet{HoneycombSite}
)
    B = 4pi / sqrt(3) / q
    Ht = zeros(ComplexF64, 4q, 2, 4q, 2)
    for (i, s) in enumerate(lattice.fromint)
        for n in neighbours(s)
            rs, rn = realspacecoordinates.([s, n])
            phase = -k' * (rn - rs) + peierls(B, s, n)
            j = TCI.pos(lattice, reducesite(n, 1, q))
            Ht[i, 1, j, 1] += -exp(1im * phase)
        end
    end
    Ht[:, 2, :, 2] = Ht[:, 1, :, 1]
    return Ht
end

antisymmetricproduct(u, v) = u[1] * v[2] - u[2] * v[1]

function get_Hlambda(
    q::Integer,
    k::Vector{Float64},
    lattice::TCI.IndexSet{HoneycombSite}
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

                Hlambda[i, 1, j, 1] += 1im * nu * exp(1im * phase)
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
    lattice::TCI.IndexSet{HoneycombSite}
)
    return get_Ht(q, k, lattice) .+ lambda .* get_Hlambda(q, k, lattice)
end
