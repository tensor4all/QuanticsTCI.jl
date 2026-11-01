import TensorCrossInterpolation as TCI
import QuanticsTCI as QTCI

function generate_reversed_cartesian_indices(dims)
    # generate CartesianIndices
    dims = tuple(dims...)
    indices = CartesianIndices(dims)
    # reverse each intex
    reversed_indices = [CartesianIndex(reverse(Tuple(idx))) for idx in indices]
    return reversed_indices
end

function qtt_tomat(tto::TCI.TensorTrain{T,4}) where {T}
    sitedims = TCI.sitedims(tto)
    localdims1 = [s[1] for s in sitedims]
    localdims2 = [s[2] for s in sitedims]
    mat = Matrix{T}(undef, prod(localdims1), prod(localdims2))
    for (i, inds1) in enumerate(generate_reversed_cartesian_indices(localdims1)[:])
        for (j, inds2) in enumerate(generate_reversed_cartesian_indices(localdims2)[:])
            mat[i, j] = TCI.evaluate(tto, collect(zip(Tuple(inds1), Tuple(inds2))))
        end
    end
    return mat
end

function qtt_tovec(tt::TCI.TensorTrain{T,3}) where {T}
    sitedims = TCI.sitedims(tt)
    localdims1 = [s[1] for s in sitedims]
    return TCI.evaluate.(Ref(tt), generate_reversed_cartesian_indices(localdims1)[:])
end

@testset "linear_function" begin
    mps_linear_ = QTCI.linear_mps(1.0, 8.0, 0.0, 3)
    mps_linear = TCI.TensorTrain(mps_linear_)
    @test qtt_tovec(mps_linear) == [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]

end

@testset "first_order_central_difference_periodic" begin
    R = 3
    mpo_center_ = QTCI.first_order_central_difference_periodic(R)
    mpo_center = TCI.TensorTrain(mpo_center_)
    mat_first_order_central_difference_periodic = (1 / (2 * 2^R)) * [
        0 1 0 0 0 0 0 -1;
        -1 0 1 0 0 0 0 0;
        0 -1 0 1 0 0 0 0;
        0 0 -1 0 1 0 0 0;
        0 0 0 -1 0 1 0 0;
        0 0 0 0 -1 0 1 0;
        0 0 0 0 0 -1 0 1;
        1 0 0 0 0 0 -1 0
    ]
    @test qtt_tomat(mpo_center) == mat_first_order_central_difference_periodic

    mps_linear_ = QTCI.linear_mps(1.0, 8.0, 0.0, 3)
    mps_linear = TCI.TensorTrain(mps_linear_)
    res = TCI.contract(mpo_center, mps_linear) # should be mpo * mps since the order is important
    qtt_tovec(res) == mat_first_order_central_difference_periodic * qtt_tovec(mps_linear)
end

@testset "second_order_central_difference_periodic" begin
    R = 3
    mpo_center_second_ = QTCI.second_order_central_difference_periodic(R)
    mpo_center_second = TCI.TensorTrain(mpo_center_second_)
    mat_second_order_central_difference_periodic = (1 / (2^(2R))) * [
        -2 1 0 0 0 0 0 1;
        1 -2 1 0 0 0 0 0;
        0 1 -2 1 0 0 0 0;
        0 0 1 -2 1 0 0 0;
        0 0 0 1 -2 1 0 0;
        0 0 0 0 1 -2 1 0;
        0 0 0 0 0 1 -2 1;
        1 0 0 0 0 0 1 -2
    ]
    @test qtt_tomat(mpo_center_second) == mat_second_order_central_difference_periodic
    mps_linear_ = QTCI.linear_mps(1.0, 8.0, 0.0, 3)
    mps_linear = TCI.TensorTrain(mps_linear_)

    res = TCI.contract(mpo_center_second, mps_linear) # should be mpo * mps since the order is important
    qtt_tovec(res) == mat_second_order_central_difference_periodic * qtt_tovec(mps_linear)
end