# The reference papers for the follwing code are:
# Quantized tensor networks for solving the Vlasov-Maxwell equations: https://arxiv.org/pdf/2311.07756
# A quantum-inspired method for solving the Vlasov-Poisson equations: https://journals.aps.org/pre/abstract/10.1103/PhysRevE.106.035208

function S(plus_minus::Symbol)::Matrix{Float64}
    if plus_minus == :plus
        return [0 0; 1 0]
    elseif plus_minus == :minus
        return [0 1; 0 0]
    else
        error("Invalid argument")
    end
end

function I()::Matrix{Float64}
    return [1 0;
        0 1]
end

# The first derivative using centered differences with periodic boundary conditions
function first_order_central_difference_periodic(R::Int64)::Vector{Array{Float64,4}}
    c1 = 1 / 2
    t_left = zeros(1, 2, 2, 3)
    t_left[1, :, :, 1] = I() * 1 / 2^R
    t_left[1, :, :, 2] = (S(:plus) + S(:minus)) * 1 / 2^R
    t_left[1, :, :, 3] = (S(:plus) + S(:minus)) * 1 / 2^R

    t_right = zeros(3, 2, 2, 1)
    t_right[1, :, :, 1] = -c1 * (S(:plus) - S(:minus))
    t_right[2, :, :, 1] = c1 * S(:plus)
    t_right[3, :, :, 1] = -c1 * S(:minus)

    t_center = zeros(3, 2, 2, 3)
    t_center[1, :, :, 1] = I()
    t_center[1, :, :, 2] = S(:minus)
    t_center[1, :, :, 3] = S(:plus)
    t_center[2, :, :, 2] = S(:plus)
    t_center[3, :, :, 3] = S(:minus)

    mpo = map(1:R) do i
        if i == 1
            t_left
        elseif i == R
            t_right
        else
            t_center
        end
    end
    mpo
end

# The second-order derivative using centered differences with periodic boundary conditions
function second_order_central_difference_periodic(R::Int64)::Vector{Array{Float64,4}}
    c1 = 1.0
    c0 = -2.0
    t_left = zeros(1, 2, 2, 3)
    t_left[1, :, :, 1] = I() * 1 / 2^(2R)
    t_left[1, :, :, 2] = (S(:plus) + S(:minus)) * 1 / 2^(2R)
    t_left[1, :, :, 3] = (S(:plus) + S(:minus)) * 1 / 2^(2R)

    t_right = zeros(3, 2, 2, 1)
    t_right[1, :, :, 1] = (c0 * I() + c1 * (S(:plus) + S(:minus)))
    t_right[2, :, :, 1] = c1 * S(:minus)
    t_right[3, :, :, 1] = c1 * S(:plus)

    t_center = zeros(3, 2, 2, 3)
    t_center[1, :, :, 1] = I()
    t_center[1, :, :, 2] = S(:plus)
    t_center[1, :, :, 3] = S(:minus)
    t_center[2, :, :, 2] = S(:minus)
    t_center[3, :, :, 3] = S(:plus)

    mpo = map(1:R) do i
        if i == 1
            t_left
        elseif i == R
            t_right
        else
            t_center
        end
    end
    mpo
end

# linear function av + b on a uniform grid,  where a, b are constants
# Grid on [−d/2, d/2), N=2^R points, v_i = −d/2 + d_i/R. QTT representation: R cores
function linear_mps(a::Float64, d::Float64, b::Float64, R::Int64)::Vector{Array{Float64,3}}
    t_first = zeros(1, 2, 2)
    t_end = zeros(2, 2, 1)
    nvec = [0; 1]
    t_first[1, :, 1] = ones(2)
    t_first[1, :, 2] = 1 / 2 * a * d * nvec + (-a * d / 2 + b) * ones(2)
    t_end[1, :, 1] = 1 / 2^R * a * d * nvec
    t_end[2, :, 1] = ones(2)

    function t_center(i)
        t_center = zeros(2, 2, 2)
        t_center[1, :, 1] = ones(2)
        t_center[2, :, 2] = ones(2)
        t_center[1, :, 2] = 1 / 2^i * a * d * nvec
        return t_center
    end

    mps = map(1:R) do i
        if i == 1
            t_first
        elseif i == R
            t_end
        else
            t_center(i)
        end
    end

    return mps
end