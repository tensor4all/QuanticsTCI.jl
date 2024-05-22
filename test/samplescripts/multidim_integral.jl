import TensorCrossInterpolation as TCI

N = 5              # Number of dimensions <@$\cN$@>
tolerance = 1e-10   # Tolerance of the internal TCI
GKorder = 15        # Order of the Gauss-Kronrod rule to use

f(x) = 2^N / (1 + 2 * sum(x))   # Integrand
integralvalue = TCI.integrate(Float64, f, fill(0.0, N), fill(1.0, N); tolerance, GKorder)

# Exact value of integral for <@$\cN = 5$@>
i5 = (-65205 * log(3) - 6250 * log(5) + 24010 * log(7) + 14641 * log(11)) / 24
error = abs(integralvalue - i5)

@info "TCI integration with GK$GKorder: " integralvalue i5 error
