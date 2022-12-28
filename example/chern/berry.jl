function berrycurvature_dets(H::Array{Matrix{ComplexF64}}, n::Integer)
    Heigen = eigen.(H)
    vecs00 = [vectors[1:n] for (vals, vectors) in Heigen]
    vecs01 = circshift(vecs00, (0, 1))
    vecs10 = circshift(vecs00, (1, 0))
    vecs11 = circshift(vecs00, (1, 1))

    function getdets(vecs1, vecs2)
        return det(adjoint(vecs1) * vecs2)
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

function berrycurvature_quantics(
    Hfunc,
    n::Integer,
    q::Vector{<:Integer},
    nquantics::Integer
)
    k = quantics_to_index(q, d=2)
    Hplaquette = [
        Hfunc(modindex.(k + [dkx, dky], 2^nquantics))
        for dkx in -1:0, dky in -1:0]
    return berrycurvature_dets(Hplaquette, n)[1, 1]
end
