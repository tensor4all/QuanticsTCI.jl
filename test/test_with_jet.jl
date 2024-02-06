using JET
import QuanticsTCI

@testset "JET" begin
    if VERSION ≥ v"1.9"
        JET.test_package(QuanticsTCI; target_defined_modules=true)
    end
end
