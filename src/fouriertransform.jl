module fourierimpl
import TensorCrossInterpolation as TCI
struct LagrangePolynomials{T}
    grid::Vector{T}
    baryweights::Vector{T}
end

function (P::LagrangePolynomials{T})(alpha::Int, x::T)::T where {T}
    if abs(x - P.grid[alpha+1]) >= 1e-14
        return prod(x .- P.grid) * P.baryweights[alpha+1] / (x - P.grid[alpha+1])
    else
        return one(T)
    end
end

function getChebyshevGrid(K::Int)::LagrangePolynomials{Float64}
    chebgrid = 0.5 * (1.0 .- cospi.((0:K) / K))
    baryweights = [
        prod(j == m ? 1.0 : 1.0 / (chebgrid[j+1] - chebgrid[m+1]) for m in 0:K)
        for j in 0:K
    ]
    return LagrangePolynomials{Float64}(chebgrid, baryweights)
end

function dftcoretensor(
    P::LagrangePolynomials{Float64},
    alpha::Int, beta::Int, sigma::Int, tau::Int;
    sign::Float64
)::ComplexF64
    x = (sigma + P.grid[beta+1]) / 2
    return P(alpha, x) * cispi(2 * sign * x * tau)
end

end


@doc raw"""
    function quanticsfouriermpo(
        R::Int;
        sign::Float64=-1.0,
        maxbonddim::Int=12,
        tolerance::Float64=1e-14,
        K::Int=25,
        method::Symbol=:SVD,
        normalize::Bool=false
    )::TCI.TensorTrain{ComplexF64}

Generate a quantics Fourier transform operator in tensor train form. When contracted with a quantics tensor train ``F_{\boldsymbol{\sigma}}`` representing a function, the result will be the fourier transform of the function in quantics tensor train form, ``\tilde{F}_{\boldsymbol{\sigma}'} = \sum_{\boldsymbol{\sigma}} F_{\boldsymbol{\sigma}} \exp(-2\pi i (k_{\boldsymbol{\sigma'}}-1) (m_{\boldsymbol{\sigma}} - 1)/M)``, where ``k_{\boldsymbol{\sigma}} = \sum_{\ell=1}^R 2^{R-\ell} \sigma_\ell``, ``m_{\boldsymbol{\sigma}'} = \sum_{\ell=1}^R 2^{R-\ell} \sigma'_\ell``, and ``M=2^R``.

!!! note "Index ordering"
    Before the Fourier transform, the left most index corresponds to ``\sigma_1``, which describes the largest length scale, and the right most index corresponds to ``\sigma_R``, which describes the smallest length scale.
    The indices ``\sigma_1' \ldots \sigma_{R}'`` in the fourier transformed QTT are aligned in the *inverse* order; that is,  the left most index corresponds to ``\sigma'_R``, which describes the smallest length scale.
    This allows construction of an operator with small bond dimension (see reference 1). If necessary, a call to `TCI.reverse(tt)` can restore large-to-small index ordering.

The Fourier transform operator is implemented using a direct analytic construction of the tensor train by Chen and Lindsey (see reference 2). The tensor train thus obtained is then re-compressed to the user-given bond dimension and tolerance.

Arguments:
- `R`: number of bits of the fourier transform.
- `sign`: sign in the exponent ``\exp(2i\pi \times \mathrm{sign} \times (k_{\boldsymbol{\sigma'}}-1) (x_{\boldsymbol{\sigma}}-1)/M)``, usually ``\pm 1``.
- `maxbonddim`: bond dimension to compress the operator to. From observations, `maxbonddim = 12` is generally big enough to reach an accuracy of `1e-12`.
- `tolerance`: tolerance of the TT compression. Note that the error in the fourier transform is generally a bit larger than this error tolerance.
- `K`: bond dimension of the TT before compression, i.e. number of basis functions to approximate the Fourier transform with (see reference 2). The TT will become inaccurate for `K < 22`; higher values may be necessary for very high precision.
- `method`: method with which to compress the TT. Choose between `:SVD` and `:CI`.
- `normalize`: whether or not to normalize the operator as an isometry.

!!! details "References"
    1. [J. Chen, E. M. Stoudenmire, and S. R. White, Quantum Fourier Transform Has Small Entanglement, PRX Quantum 4, 040318 (2023).](https://link.aps.org/doi/10.1103/PRXQuantum.4.040318)
    2. [J. Chen and M. Lindsey, Direct Interpolative Construction of the Discrete Fourier Transform as a Matrix Product Operator, arXiv:2404.03182.](http://arxiv.org/abs/2404.03182)
"""
function quanticsfouriermpo(
    R::Int;
    sign::Float64=-1.0,
    tolerance::Float64=1e-14,
    maxbonddim::Int=12,
    K::Int=25, method::Symbol=:SVD,
    normalize::Bool=true
)::TCI.TensorTrain{ComplexF64}
    P = fourierimpl.getChebyshevGrid(K)
    A = [
        fourierimpl.dftcoretensor(P, alpha, beta, sigma, tau; sign)
        for alpha in 0:K, tau in [0, 1], sigma in [0, 1], beta in 0:K
    ]
    Afirst = reshape(sum(A, dims=1), (1, 2, 2, K + 1))
    Alast = reshape(A[:, :, :, 1], (K + 1, 2, 2, 1))
    tt = TCI.TensorTrain{ComplexF64,4}([Afirst, fill(A, R - 2)..., Alast])
    TCI.compress!(tt, method; tolerance, maxbonddim)
    if normalize
        for t in tt.sitetensors
            t ./= sqrt(2.0)
        end
    end
    return tt
end

function swaphalves!(tt::TCI.TensorTrain{V,3}) where {V}
    tt.sitetensors[1][:, :, :] = tt.sitetensors[1][:, [2, 1], :]
    nothing
end

function swaphalves(tt::TCI.AbstractTensorTrain{V}) where {V}
    ttcopy = TCI.tensortrain(deepcopy(TCI.sitetensors(tt)))
    swaphalves!(ttcopy)
    return ttcopy
end
