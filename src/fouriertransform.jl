module fourierimpl
    import TensorCrossInterpolation as TCI

    function fkm(sign::ComplexF64, k::IntType, m::IntType, R::Int)::ComplexF64 where {IntType}
        # cispi(x) is exp(i * pi * x)
        cispi(sign * 2 * (k - 1) * (m - 1) / IntType(2)^R) #/ 2^(R÷2)
    end

    function quantics_to_int(::Type{IntType}, quantics_index::Vector{Int}) where {IntType}
        R = length(quantics_index)
        sum(IntType(2) .^ (R-1:-1:0) .* (quantics_index .- 1)) + 1
    end

    function fquantics(
        ::Type{IntType}, sign::ComplexF64, fused_quantics_index::Vector{Int}
    )::ComplexF64 where {IntType}
        kq = mod.(fused_quantics_index .- 1, 2) .+ 1
        mq = div.(fused_quantics_index .- 1, 2) .+ 1
        reverse!(mq)
        return fkm(
            sign,
            quantics_to_int(IntType, kq),
            quantics_to_int(IntType, mq),
            length(fused_quantics_index)
        )
    end

    function getqtci(
        ::Type{IntType},
        R::Int,
        sign::ComplexF64=-1;
        tolerance::Float64=1e-8,
        pivottolerance::Float64=tolerance,
        kwargs...
    )::TCI.TensorCI2{ComplexF64} where {IntType}
        function fq(q::Vector{Int})::ComplexF64
            fquantics(IntType, sign, q)
        end
        qfttci, = TCI.crossinterpolate2(
            ComplexF64, fq, fill(4, R);
            pivotsearch=:full,
            tolerance, pivottolerance,
            kwargs...
        )
        return qfttci
    end
end

function quanticsfouriertto(
    R::Int,
    sign=-1;
    tolerance::Float64=1e-8,
    pivottolerance::Float64=tolerance,
    normalization::Symbol=:none,
    kwargs...
)::TCI.TensorTrain{ComplexF64, 4}
    qfttci::TCI.TensorCI2 =
        if R > 20   # 62 because signed int and 1-based indexing.
            fourierimpl.getqtci(
                BigInt, R, ComplexF64(sign);
                tolerance, pivottolerance, kwargs...
            )
        else
            fourierimpl.getqtci(
                Int64, R, ComplexF64(sign);
                tolerance, pivottolerance, kwargs...
            )
        end

    # split k and m indices
    qfttt = TCI.TensorTrain{4}(qfttci, fill((2, 2), R))

    if normalization === :full
        qfttt.sitetensors ./= 2.0
    elseif normalization === :sqrt
        qfttt.sitetensors ./= sqrt(2)
    elseif normalization !== :none
        error("Unknown normalization $normalization. Choose between :none (1/1), :full (1/2^R) and :sqrt (1/sqrt(2^R)).")
    end

    return qfttt
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
