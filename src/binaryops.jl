function binary_op_mpo(
    singletensor,
    outindices;
    firstindices=[Index(2, "a,$i") for i in eachindex(outindices)],
    secondindices=[Index(2, "b,$i") for i in eachindex(outindices)],
    removeends=true
)
    if isa(outindices, Int)
        n = outindices
        outindices = [Index(2, "out,$i") for i in 1:n]
    else
        n = length(outindices)
    end

    plusmpo = MPO(n)
    linkindices = [Index(2, "l,$i") for i in 0:n]
    for i in 1:n
        plusmpo[i] = ITensor(
            singletensor,
            outindices[i],
            linkindices[i+1],
            firstindices[i],
            secondindices[i],
            linkindices[i]
        )
    end

    if removeends
        plusmpo[1] *= state(linkindices[1], 1)
        plusmpo[n] *= state(linkindices[n+1], 1)
    end

    return plusmpo
end

function binary_addition_mpo(
    outindices,
    firstindices=[Index(2, "a,$i") for i in eachindex(outindices)],
    secondindices=[Index(2, "b,$i") for i in eachindex(outindices)];
    removeends=true)

    singleplus = [
        a + b + carryin == 2 * carryout + out
        # Order matters!
        for out in 0:1, carryout in 0:1, a in 0:1, b in 0:1, carryin in 0:1
    ]

    return binary_op_mpo(
        singleplus,
        outindices;
        firstindices=firstindices,
        secondindices=secondindices,
        removeends=removeends)
end

function binary_subtraction_mpo(
    outindices,
    firstindices=[Index(2, "a,$i") for i in eachindex(outindices)],
    secondindices=[Index(2, "b,$i") for i in eachindex(outindices)];
    removeends=true)

    singleminus = [
        2 * carryout + a - b - carryin == out
        # Order matters!
        for out in 0:1, carryout in 0:1, a in 0:1, b in 0:1, carryin in 0:1
    ]

    return binary_op_mpo(
        singleminus,
        outindices;
        firstindices=firstindices,
        secondindices=secondindices,
        removeends=removeends)
end

function kroneckerdelta_mpo(
    indexlists...;
    removeends=true)
    n = length(indexlists[1])

    nlegs = length(indexlists)
    allsame(x) = all(y -> y == first(x), x)
    delta = [
        allsame(indices) for indices in
        Iterators.product([1:dim(first(i)) for i in indexlists]...)
    ]

    result = MPO(n)
    for i in 1:n
        result[i] = ITensor(
            delta,
            [indexlist[i] for indexlist in indexlists]...
        )
    end
    return result
end

# function bitshift_mpo(
#     outindices,
#     firstindices=[Index(2, "a,$i") for i in eachindex(outindices)],
#     secondindices=[Index(2, "b,$i") for i in eachindex(outindices)];
#     removeends=true)
# end
