function berrycurvature_dets(H::Array{Matrix{ComplexF64}}, n::Integer)
    Heigen = eigen.(H)
    vecs00 = [vectors[:, 1:n] for (vals, vectors) in Heigen]
    vecs01 = circshift(vecs00, (0, 1))
    vecs10 = circshift(vecs00, (1, 0))
    vecs11 = circshift(vecs00, (1, 1))

    function getdets(vecs1, vecs2)
        d = det(adjoint(vecs1) * vecs2)
        return d
    end

    dets = (
        getdets.(vecs00, vecs10) .*
        getdets.(vecs10, vecs11) .*
        getdets.(vecs11, vecs01) .*
        getdets.(vecs01, vecs00)
    )

    # bc = angle.(dets)
    bc = @. mod(angle(dets) + pi / 2, pi) - pi / 2
    return bc
end

function modindex(i::Integer, n::Integer)
    return mod(i - 1, n) + 1
end

function berrycurvature_quantics_dets(
    Hfunc,
    n::Integer,
    q::Vector{<:Integer},
    nquantics::Integer
)
    k = [quantics_to_index(qi)[1] for qi in split_dimensions(q, 2)]
    Hplaquette = [
        Hfunc(modindex.(k .+ [dkx, dky], 2^nquantics))
        for dkx in -1:0, dky in -1:0]
    return berrycurvature_dets(Hplaquette, n)[1, 1]
end

function berrycurvature_derivatives(
    H::Matrix{ComplexF64},
    Hderivative1::Matrix{ComplexF64},
    Hderivative2::Matrix{ComplexF64},
    n::Integer
)
    E, U = eigen(Hermitian(H))

    return -2 * sum(imag(
        (
            (U[:, v]' * Hderivative1 * U[:, c]) * (U[:, c]' * Hderivative2 * U[:, v]) -
            (U[:, v]' * Hderivative2 * U[:, c]) * (U[:, c]' * Hderivative1 * U[:, v])
        ) /
        (E[c] - E[v])^2
    ) for v in 1:n, c in n+1:length(E))

    # Ediff = E[n+1:end] .- E[1:n]'
    # v1 = U[:, 1:n]' * Hderivative1 * U[:, n+1:end] ./ Ediff'
    # v2 = U[:, n+1:end]' * Hderivative2 * U[:, 1:n] ./ Ediff
    # return 2 * tr(imag.(v1 * v2))
end

function berrycurvature_quantics_derivatives(
    Hfunc,
    Hderivfunc,
    n::Integer,
    q::Vector{<:Integer}
)
    k = [quantics_to_index(qi)[1] for qi in deinterleave_dimensions(q, 2)]
    return berrycurvature_derivatives(
        Hfunc(k),
        Hderivfunc(k, 1),
        Hderivfunc(k, 2),
        n)
end
