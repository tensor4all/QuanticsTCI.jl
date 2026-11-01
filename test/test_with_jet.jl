using JET
import QuanticsTCI

@testset "JET" begin
    if VERSION ≥ v"1.10"
        JET.test_package(QuanticsTCI; target_modules=(QuanticsTCI,))
    end
end
