using SafeTestsets

@safetestset "Test Local Operator" begin
    include("test_local_operator.jl")
end

@safetestset "Test Local Hybrid Operator" begin
    include("test_local_hybrid_coupling.jl")
end
