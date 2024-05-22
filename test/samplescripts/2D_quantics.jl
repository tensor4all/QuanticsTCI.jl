using QuanticsTCI
import QuanticsGrids as QG

R = 40  # Number of bits <@$\cR$@>
xygrid = QG.DiscretizedGrid{2}(R, (-5.0, -5.0), (5.0, 5.0)) # Discretization grid <@$\vec{x}(\bsigma)$@>

function f(x, y) # Function of interest <@$f(x)$@>
    return exp(-0.4*(x^2 + y^2)) + 1 + sin(x * y) * exp(-x^2) +
        cos(3*x*y) * exp(-y ^ 2) + cos(x+y)
end

# Construct and optimize quantics TCI <@$\tf_\bsigma$@>
f_tci, ranks, errors = quanticscrossinterpolate(Float64, f, xygrid; tolerance=1e-10)

# Print a table to compare <@$f(x)$@> and <@$\tF_\bsigma$@> on some regularly spaced points
println("x\t y\t f(x)\t\t\t f_tt(x)")
for index in CartesianIndices((10, 10))
    m = Tuple(index) .* div(2^R, 10)
    x, y = QG.grididx_to_origcoord(xygrid, m)
    println("$x\t$y\t$(f(x, y))\t$(f_tci(m))")
end

println("Value of the integral: $(integral(f_tci))")
