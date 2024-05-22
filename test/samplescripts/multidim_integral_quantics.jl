import QuanticsGrids as QG
import TensorCrossInterpolation as TCI

N = 5               # Number of dimensions <@$\cN$@>
tolerance = 1e-10   # Tolerance of the internal TCI
R = 40              # Number of bits <@$\cR$@>

f(x) = 2^N / (1 + 2 * sum(x))    # Integrand <@$f(\vec{x})$@>

# Discretization grid with <@$2^{\scN \scR}$@> points
grid = QG.DiscretizedGrid{N}(R, Tuple(fill(0.0, N)), Tuple(fill(1.0, N)), unfoldingscheme=:interleaved)
quanticsf(sigma) = f(QG.quantics_to_origcoord(grid, sigma)) # <@$f(\vec{x}(\bsigma))$@>

# Obtain the QTCI representation and evaluate the integral via factorized sum
tci, ranks, errors = TCI.crossinterpolate2(Float64, quanticsf, QG.localdimensions(grid); tolerance)

# Integral is sum multiplied with discretization volumne
integralvalue = TCI.sum(tci) * prod(QG.grid_step(grid))

# Exact value of integral for <@$\cN = 5$@>
i5 = (-65205 * log(3) - 6250 * log(5) + 24010 * log(7) + 14641 * log(11)) / 24
error = abs(integralvalue - i5)     # Error for <@$\cN = 5$@>

@info "Quantics TCI integration with R=$R: " integralvalue i5 error
