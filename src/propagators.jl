function G0(Lambda, nu)
    return -nu ./ (Lambda .^ 2 .+ nu .^ 2)
end

function G0_quantics(Lambda, numax, nu_q)
    dim = 2^length(nu_q)
    nu = (2 * quantics_to_index(nu_q)[1] / dim - 1) * numax
    return G0(Lambda, nu)
end

function G0_tensor(Lambda, numax, n)
    shape = fill(2, n)
    G0tensor = zeros(shape...)
    for index in CartesianIndices(G0tensor)
        G0tensor[index] = G0_quantics(Lambda, numax, Tuple(index))
    end
    return G0tensor
end

function G0_mps(Lambda, numax, n; cutoff=1e-6, maxdim=10)
    G0indices = [Index(2, "u,$i") for i in 1:n]
    return MPS(G0_tensor(Lambda, numax, n), G0indices;
        cutoff=cutoff, maxdim=maxdim)
end

function G0_qtt(Lambda, numax, n; cutoff=1e-6, maxdim=40)
    return qtt(
        nu_q -> G0_quantics(Lambda, numax, nu_q),
        2,
        index_to_quantics(Int(floor(Lambda * n / numax)), n);
        cutoff=cutoff, maxiter=maxiter
    )
end

function pair_propagator_mps(Lambda, numax, n)
    G0 = G0_mps(Lambda, numax, n)
    G0G0 = outer(G0, G0'; alg="naive", truncate=false)

    nuindices = [Index(2, "nu,$i") for i in 1:n]
    omegaindices = [Index(2, "omega,$i") for i in 1:n]

    plusmpo = binary_addition_mpo(siteinds(G0))
    minusmpo = binary_subtraction_mpo(siteinds(G0)')

    nusplice = kroneckerdelta_mpo(
        nuindices, siteinds(plusmpo; tags="a"), siteinds(minusmpo; tags="a"))
    omegasplice = kroneckerdelta_mpo(
        omegaindices, siteinds(plusmpo; tags="b"), siteinds(minusmpo; tags="b"))

    spiderweb = contract(
        contract(plusmpo, nusplice),
        contract(minusmpo, omegasplice))

    return contract(spiderweb, G0G0; alg="naive", truncate=false)
end
