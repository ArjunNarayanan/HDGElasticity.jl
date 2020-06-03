module HDGElasticity

using LinearAlgebra
using StaticArrays
using LightGraphs
using CartesianMesh
using PolynomialBasis
using ImplicitDomainQuadrature
import ImplicitDomainQuadrature: extend

IDQ = ImplicitDomainQuadrature

include("utils.jl")
include("dg_mesh.jl")
# include("local_operator.jl")
# include("local_hybrid_coupling.jl")
# include("displacement_component_bc.jl")
# include("assembly.jl")
# include("isotropic_elasticity.jl")

end # module
