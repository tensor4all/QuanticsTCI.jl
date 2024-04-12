
function fouriertto(
    R::Int,
    sign::Int=-1;
    tolerance=1e-8,
    pivottolerance=tolerance,
    normalization=1.0,
    kwargs...
)::TCI.TensorTrain{ComplexF64, 4}
    function fkm(k::BigInt, m::BigInt, R::Int)::ComplexF64
        # cispi(x) is exp(i * pi * x)
        cispi(sign * k * m / big(2)^R) #/ 2^(R÷2)
    end

    # Necessary for the BigInt conversion
    function quantics_to_int(quantics_index::Vector{Int})::BigInt
        R = length(quantics_index)
        sum(big(2) .^ (0:R-1) .* (quantics_index .- 1))
    end

    function fq(fused_quantics_index::Vector{Int})::ComplexF64
        kq = mod.(fused_quantics_index .- 1, 2) .+ 1
        mq = div.(fused_quantics_index .- 1, 2) .+ 1
        reverse!(kq)
        return fkm(quantics_to_int(kq), quantics_to_int(mq), length(fused_quantics_index))
    end

    qfttci, ranks, errors = TCI.crossinterpolate2(
        ComplexF64,
        fq,
        fill(4, R);
        pivotsearch=:full,
        tolerance, pivottolerance,
        kwargs...
    )

    # split k and m indices
    qfttt = TCI.tensortrain([reshape(T, (size(T, 1), 2, 2, size(T, 3))) for T in qfttci])
    multiply!(qfttt, normalization)
    return qfttt
end

function applytto(
    tto::TCI.TensorTrain{V, 4},
    tt::Union{TCI.TensorTrain{V, 3}, TCI.TensorCI1{V}, TCI.TensorCI2{V}};
    tolerance=1e-8, maxbonddim=200
)::TCI.TensorTrain{V, 3} where {V}
    qttcontraction = TCIA.contract(
        tto,
        TCI.tensortrain([reshape(T, (size(T, 1), size(T, 2), 1, size(T, 3))) for T in tt]);
        tolerance, maxbonddim
    )
    return TCI.tensortrain(
        [reshape(T, (size(T, 1), size(T, 2), size(T, 4))) for T in qttcontraction])
end

function applytto(
    tt::Union{TCI.TensorTrain{V, 3}, TCI.TensorCI1{V}, TCI.TensorCI2{V}},
    tto::TCI.TensorTrain{V, 4};
    tolerance=1e-8, maxbonddim=200
)::TCI.TensorTrain{V, 3} where {V}
    qttcontraction = TCIA.contract(
        TCI.tensortrain([reshape(T, (size(T, 1), 1, size(T, 2), size(T, 3))) for T in tt]),
        tto;
        tolerance, maxbonddim
    )
    return TCI.tensortrain(
        [reshape(T, (size(T, 1), size(T, 3), size(T, 4))) for T in qttcontraction])
end


function swaphalves!(tt::TCI.TensorTrain{V, 3}) where {V}
    tt.sitetensors[1][:, :, :] = tt.sitetensors[1][:, [2, 1], :]
    nothing
end

function swaphalves(tt::TCI.AbstractTensorTrain{V}) where {V}
    ttcopy = TCI.tensortrain(deepcopy(TCI.sitetensors(tt)))
    swaphalves!(ttcopy)
    return ttcopy
end
