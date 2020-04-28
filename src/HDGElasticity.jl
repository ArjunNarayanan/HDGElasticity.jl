module HDGElasticity

using LinearAlgebra
using StaticArrays
using CartesianMesh
using ImplicitDomainQuadrature

include("local_operator.jl")
include("local_hybrid_coupling.jl")
include("normal_displacement_bc.jl")
include("hdge_assembly.jl")
include("isotropic_elasticity.jl")

end # module
