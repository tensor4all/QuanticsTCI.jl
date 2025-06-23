using Aqua
import QuanticsTCI

@testset "Aqua" begin
    Aqua.test_all(QuanticsTCI; deps_compat=false)
end
