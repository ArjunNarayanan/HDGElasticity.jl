using SafeTestsets

@safetestset "Test Affine Map" begin
    include("test_affine_map.jl")
end

@safetestset "Test Utils" begin
    include("test_utils.jl")
end

@safetestset "Test Curved Face Area Scaling" begin
    include("test_curved_facescale.jl")
end

@safetestset "Test Variational Form Utils" begin
    include("test_variational_form_utils.jl")
end

@safetestset "Test DGMesh" begin
    include("test_dg_mesh.jl")
end

@safetestset "Test Interface Fitting" begin
    include("test_fit_interface_hybrid_element.jl")
end

@safetestset "Test Function Space Construction" begin
    include("test_function_space.jl")
end

@safetestset "Test Isotropic Hooke Matrix" begin
    include("test_isotropic_elasticity.jl")
end

@safetestset "Test Local Operator" begin
    include("test_local_operator.jl")
end

@safetestset "Test Local Hybrid Operator" begin
    include("test_local_hybrid_coupling.jl")
end

@safetestset "Test Hybrid Operator" begin
    include("test_hybrid_coupling.jl")
end

@safetestset "Test Hybrid Local Operator" begin
    include("test_hybrid_local_coupling.jl")
end

@safetestset "Test Full Displacement BC on single element" begin
    include("test_full_displacement_bc_on_cells.jl")
end

@safetestset "Test Traction BCs" begin
    include("test_traction_bc.jl")
end

@safetestset "Test Local Solver Initialization" begin
    include("test_local_solver.jl")
end

@safetestset "Test Assembly" begin
    include("test_assembly.jl")
end

@safetestset "Test Assembly on Boundary Cells" begin
    include("test_boundary_mesh_assembly.jl")
end

@safetestset "Test Incoherent Interface" begin
    include("test_incoherent_interface.jl")
end
