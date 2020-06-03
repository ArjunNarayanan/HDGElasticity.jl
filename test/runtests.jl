using SafeTestsets

@safetestset "Test Utils" begin
    include("test_utils.jl")
end

# @safetestset "Test Local Operator" begin
#     include("test_local_operator.jl")
# end
#
# @safetestset "Test Local Hybrid Operator" begin
#     include("test_local_hybrid_coupling.jl")
# end
#
# @safetestset "Test Displacement Component BC" begin
#     include("test_displacement_component_bc.jl")
# end
#
# @safetestset "Test Assembly" begin
#     include("test_assembly.jl")
# end
#
# @safetestset "Test Isotropic Hooke Matrix" begin
#     include("test_isotropic_elasticity.jl")
# end
