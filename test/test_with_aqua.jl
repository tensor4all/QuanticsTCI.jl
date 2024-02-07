using Aqua
import QuanticsTCI

@testset "Aqua" begin
    Aqua.test_all(QuanticsTCI; ambiguities = false, unbound_args = false, deps_compat = false)
end
